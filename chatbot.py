import os
import re
import json
import time
import ollama
import chromadb
import pandas as pd
import networkx as nx

from re import Match
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from spacy.language import Language
from graph_visualization import GraphVisualizer
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from dotenv import load_dotenv
from query_routing import rag_is_needed
from translation import to_english_query
from translation import to_english_response
from translation import to_chinese_response
from query_refinement import to_search_query
from query_validation import is_valid_search_query
from umls_concept_linking import get_umls_concepts
from graph_data_retrieval import retrieve_subgraph
from query_abbreviation_expansion import expand_abbreviations

from pubmed_papers import PMID_TO_PAPER

load_dotenv()

TOPIC_TO_NAME: dict[str, str] = \
{
    "Children Allergy": "*Children Allergy*", "MRT": "*miRNA-target Relationship Tracker* (MRT)", "MSC": "*Mesenchymal Stem Cell* (MSC)",
    "TRIP": "*Taiwan Regenerative medicine and Cell Therapy Information Portal* (TRIP)", "TWHM": "*Taiwan Han Medicine* (TWHM)"
}

PMIDS: set[str] = set(PMID_TO_PAPER.keys())

CHROMA = chromadb.PersistentClient("./chroma_db")

OPENAI = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

OPENAI_EF = OpenAIEmbeddingFunction(os.environ.get("OPENAI_API_KEY"), "text-embedding-3-large")

MODEL_NAME_TO_ID: dict[str, str] = {"GPT-5.2": "gpt-5.2", "GPT-4.1": "gpt-4.1"}

PROMPT = \
"""
### INSTRUCTION

You are a specialized research assistant in {TOPIC}. You must use only the information contained in the CONTEXT below and logical combinations of that information.

The CONTEXT contains two parts:

1. A set of PubMed papers (titles and abstracts).
2. A biomedical knowledge graph built from these papers, including nodes, edges, and multi-hop paths (e.g., A → B → C).

You should:
- First, provide an evidence-based answer grounded in the PubMed papers.
- Then, when appropriate, use the knowledge graph structure (e.g., chains such as A → B → C, shared neighbors, hubs) to suggest **possible but unproven** relationships or research directions.

---

### CONTEXT

{CONTEXT}

---

### RULES

1. **Evidence-Based First:**

   * First, check whether the PubMed papers in the CONTEXT contain explicit information that answers the user's query.
   * When you state something that is directly supported by the papers (titles/abstracts), clearly indicate that it is **evidence-based**.

2. **Use of the Knowledge Graph:**

   * The graph section of the CONTEXT may list:
     - Nodes (e.g., genes, drugs, diseases, pathways, cell types).
     - Edges with relation labels (e.g., "upregulates", "inhibits", "associated_with").
     - Multi-hop paths such as: A → B → C, with supporting PMIDs.
   * You may use the graph structure to perform **multi-hop reasoning**, for example:
     - If A is strongly linked to B, and B is strongly linked to C, it may **suggest** an indirect relationship between A and C.
     - If two entities share many common neighbors, it may suggest a functional or mechanistic relationship.
   * However, such A–C relationships are **not** considered proven unless they are explicitly stated in one or more PubMed abstracts.

3. **Hypotheses and Speculative Inference (When Direct Evidence Is Limited):**

   * If the CONTEXT does not contain a clear or complete direct answer, you may **carefully infer or hypothesize** based on:
     - Patterns and relationships described in the PubMed abstracts, and
     - The knowledge graph structure (e.g., multi-hop paths, shared neighbors).
   * These inferences must be **grounded only in the provided papers and graph**. Do NOT use any knowledge or assumptions from outside the CONTEXT.
   * Clearly separate such content into a section titled **"Hypotheses / Speculative Inference"**.
   * Explicitly state that these are **not directly confirmed by the papers**.
   * Use cautious language (e.g., "may", "might", "could", "is consistent with", "suggests", "plausibly") and avoid making definitive claims about unproven relationships.

4. **When No Meaningful Inference Is Possible:**

   * If the CONTEXT is "N/A", or if the papers and graph are clearly unrelated and do not even allow a reasonable hypothesis about the query, then output exactly:
     *"The provided papers do not contain information regarding this query."*

5. **Strict Citation:**

   * Every **evidence-based** claim must be cited with its corresponding PMID from the CONTEXT.
   * For **hypotheses / speculative inference**, you must still cite the PMIDs that provide the underlying observations (e.g., the A–B and B–C edges), but make it clear that the final A–C conclusion itself is not directly stated in any paper.
   * **Format:** Place citations at the end of the sentence using brackets.
     - Single citation: `[PMID]` (e.g., `[12345678]`).
     - Multiple citations: `[PMID1, PMID2]` (e.g., `[12345678, 87654321]`).
   * Do not combine information from papers not listed in the CONTEXT.

6. **No External Knowledge:**

   * Do NOT use your internal training data, external databases, or general domain knowledge beyond what is explicitly represented in the CONTEXT.
   * All reasoning and hypotheses must be derivable from the information in the provided papers and graph.

---

### OUTPUT STRUCTURE

If possible, structure your answer into the following sections:

1. **Evidence-Based Answer**  
   - Summarize what the papers directly state that is relevant to the user's query, with PMIDs.

2. **Hypotheses / Speculative Inference** (optional)  
   - Only if there is no clear direct answer, or if the direct evidence is incomplete.
   - Propose possible explanations, mechanisms, or future research directions that are **consistent with** but not **proven by** the papers and graph.
   - Make it explicit that these points are speculative and indicate which PMIDs support the underlying edges or observations.

If there is absolutely no relevant information or reasonable inference, return only the fixed sentence described in Rule 4.

---

### TASK

*User:* {QUERY}

*Answer:*
"""

class Chatbot:

    def __init__(self, chat_topic: str, ner_pipeline: Language, topic_to_pmid_to_node_weights: dict[str, dict[str, dict[str, int]]],
                 topic_to_graphs: dict[str, dict[str, nx.DiGraph | nx.Graph]]) -> None:

        self.chat_topic = chat_topic
        self.ner_pipeline = ner_pipeline

        self.chat_topic_name = TOPIC_TO_NAME[chat_topic]

        self.collection = CHROMA.get_collection(chat_topic, OPENAI_EF)

        if (chat_topic in topic_to_pmid_to_node_weights) and (chat_topic in topic_to_graphs):

            self.pmid_to_node_weights: dict[str, dict[str, int]] = topic_to_pmid_to_node_weights[chat_topic]

            self.digraph: nx.DiGraph = topic_to_graphs[chat_topic]["digraph"]
            self.graph: nx.Graph = topic_to_graphs[chat_topic]["graph"]

            self.graph_visualizer = GraphVisualizer(ner_pipeline, self.digraph, self.graph)
        
    def get_response(self, chat_id: str, query: str, rag_scheme: str, n_max_refs: int, chat_model: str, prefer_lan: str) -> dict[str, str]:

        chat_log: dict[str, str] = \
        {
            "query": query, "english_query": "N/A", "search_query": "N/A", "rag_scheme": "N/A", 
            "pmids": "N/A", "chat_model": "N/A", "response": "N/A", "digraph_html_path": "N/A"
        }

        if not re.match(r'^[a-zA-Z0-9\u4e00-\u9fff\u0370-\u03ff_\s\W]+$', query):
            
            chat_log["response"] = "Unsupported language detected. Please use English or Chinese."
            
            return chat_log
        
        chat_history, chat_log["english_query"] = self._retrieve_chat_history(chat_id), to_english_query(expand_abbreviations(query))

        chat_log["search_query"] = to_search_query(chat_history, chat_log["english_query"])

        if not is_valid_search_query(chat_log["search_query"]):

            chat_log["response"] = f"Your request is outside my scope, as I am designed to assist exclusively with {self.chat_topic_name}."
            
            return chat_log
        
        if rag_is_needed(chat_log["search_query"]):

            if rag_scheme[0] == "T":
                data = self._retrieve_data_from_vectordb(chat_log["search_query"], n_max_refs)
                graph_search_nodes, graph_filtered_edges = set(), {}
            elif rag_scheme[0] == "G":
                data = self._retrieve_data_from_graphdbs(chat_log["search_query"], n_max_refs)
                graph_search_nodes, graph_filtered_edges = data[3], data[4]
            elif rag_scheme[0] == "H":
                data = self._retrieve_data_from_hybriddb(chat_log["search_query"], n_max_refs)
                graph_search_nodes, graph_filtered_edges = data[3], data[4]
            else:
                data = ("N/A", "N/A", "Unknown")
                graph_search_nodes, graph_filtered_edges = set(), {}

            chat_log["pmids"], context, chat_log["rag_scheme"] = data[0], data[1], data[2]

            messages: list[dict[str, str]] = \
            [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        QUERY=chat_log["search_query"],
                        TOPIC=self.chat_topic_name, 
                        CONTEXT=context
                    )
                }
            ]

            model: str = MODEL_NAME_TO_ID[chat_model]

            response = OPENAI.chat.completions.create(model=model, messages=messages, temperature=0)

            chat_log["response"] = self._linkify_pmids(response.choices[0].message.content)

        else:

            chat_log["chat_model"] = "Ministral-3-8B"

            messages: list[dict[str, str]] = \
            [
                {
                    "role": "user",
                    "content": (
                        f"You are a helpful research assistant specialized in {self.chat_topic_name}.\n\n"
                        f"{chat_log['search_query']}"
                    )
                }
            ]

            response = ollama.chat("ministral-3:8b", messages, options={"temperature": 0}, keep_alive=-1)

            chat_log["response"] = response["message"]["content"]
        
        chat_log_path = Path(f"./chat_logs/{chat_id}/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json")
        chat_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if chat_log["rag_scheme"][0] in ["G", "H"]:

            self.graph_visualizer.generate_pyvis_html(graph_search_nodes, graph_filtered_edges, str(chat_log_path))

            chat_log["digraph_html_path"] = str(chat_log_path.with_suffix(".html"))
        
        if prefer_lan == "English": chat_log["response"] = to_english_response(chat_log["response"])
        if prefer_lan == "Chinese": chat_log["response"] = to_chinese_response(chat_log["response"])

        chat_log_path.write_text(json.dumps(chat_log, ensure_ascii=False, indent=4), encoding="utf-8")

        return chat_log

    def _retrieve_chat_history(self, chat_id: str) -> list[tuple[str, str]]:

        chat_history: list[dict[str, str]] = []

        chat_history_dir = Path(f"./chat_logs/{chat_id}")

        if not chat_history_dir.exists(): return chat_history

        for filepath in sorted(chat_history_dir.glob("*.json"))[-100:]:
            
            chat_log: dict[str, str] = json.load(open(filepath, "r", encoding="utf-8"))

            chat_history.append((chat_log["search_query"], chat_log["response"]))
            
        return chat_history
    
    def _retrieve_data_from_vectordb(self, search_query: str, n_max_refs: int) -> tuple[str, str, str]:

        chromadb_results = self.collection.query(query_texts=[search_query], include=["documents"], n_results=n_max_refs)

        pmids: list[str] = []
        texts: list[str] = []

        for pmid in chromadb_results["ids"][0]:

            pmids.append(pmid)

            title = PMID_TO_PAPER[pmid]["Title"]
            abstract = PMID_TO_PAPER[pmid]["Abstract"]
            
            texts.append(f"*PMID:* {pmid}\n*Title*: {title}\n*Abstract:* {abstract}")

        return ";".join(pmids), "\n\n".join(texts), "Text"

    def _retrieve_data_from_graphdbs(self, search_query: str, n_max_refs: int) -> tuple:

        search_nodes_df: pd.DataFrame = get_umls_concepts(search_query, self.ner_pipeline)

        search_nodes: set[str] = {n for n in search_nodes_df["UMLS_Concept_Name"] if n in self.digraph}

        if not search_nodes:
            pmids_str, context, scheme = self._retrieve_data_from_vectordb(search_query, n_max_refs)
            return pmids_str, context, scheme, set(), {}

        pmids: list[str] = []
        texts: list[str] = []

        start_time = time.time()
        print("\nRetrieving Papers from Graph ...")

        retrieved_pmids, retrieved_edges = retrieve_subgraph(
            search_nodes, n_max_refs, self.pmid_to_node_weights, self.digraph, self.graph
        )

        print(f"Done. {time.time() - start_time:.2f} seconds taken.")

        for pmid in retrieved_pmids:

            pmids.append(pmid)

            title = PMID_TO_PAPER[pmid]["Title"]
            abstract = PMID_TO_PAPER[pmid]["Abstract"]

            texts.append(f"*PMID:* {pmid}\n*Title*: {title}\n*Abstract:* {abstract}")

        context = "\n\n".join(texts) + self._format_graph_context(retrieved_edges)

        return ";".join(pmids), context, "Graph", search_nodes, retrieved_edges
    
    def _format_graph_context(self, edges: dict[tuple[str, str, str, str, str], set[str]]) -> str:

        if not edges: return ""
        
        lines = ["\n\n### KNOWLEDGE GRAPH EDGES ###"]
        for (u, u_type, label, v, v_type), pmids in edges.items():

            pmid_str = ", ".join(sorted(list(pmids)))

            lines.append(f"- {u} ({u_type}) --[{label}]--> {v} ({v_type}) | Supported by PMIDs: {pmid_str}")
            
        return "\n".join(lines)

    def _retrieve_data_from_hybriddb(self, english_query: str, n_max_refs: int) -> tuple:

        vectordb_data: tuple[str, str, str] = self._retrieve_data_from_vectordb(english_query, n_max_refs)
        graphdbs_data: tuple = self._retrieve_data_from_graphdbs(english_query, n_max_refs)

        graph_search_nodes: set = graphdbs_data[3]
        graph_filtered_edges: dict = graphdbs_data[4]

        if vectordb_data[0] == graphdbs_data[0]:
            return (*vectordb_data, graph_search_nodes, graph_filtered_edges)

        pmids: list[str] = []
        texts: list[str] = []

        all_pmids = set(vectordb_data[0].split(";") + graphdbs_data[0].split(";"))
        all_pmids.discard("")

        for pmid in all_pmids:
            pmids.append(pmid)
            title = PMID_TO_PAPER[pmid]["Title"]
            abstract = PMID_TO_PAPER[pmid]["Abstract"]
            texts.append(f"*PMID:* {pmid}\n*Title*: {title}\n*Abstract:* {abstract}")

        graph_context = ""
        if "### KNOWLEDGE GRAPH EDGES ###" in graphdbs_data[1]:
            graph_context = "\n\n### KNOWLEDGE GRAPH EDGES ###" + graphdbs_data[1].split("### KNOWLEDGE GRAPH EDGES ###")[1]

        context = "\n\n".join(texts) + graph_context

        return ";".join(pmids), context, "Hybrid", graph_search_nodes, graph_filtered_edges

    def _linkify_pmids(self, response: str) -> str:

        def replacer(match: Match) -> str:

            number: str = match.group(1)

            if number not in PMIDS: return number

            return f"[{number}](https://pubmed.ncbi.nlm.nih.gov/{number})"

        return re.compile(r'\b(\d+)\b').sub(replacer, response)
