import os
import json
import time
import random
import asyncio
import warnings

from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from google.oauth2.service_account import Credentials
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer

from dotenv import load_dotenv

from pubmed_papers import PMID_TO_PAPER, PMID_TO_TOPIC

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

PROMPT: str = Path("./prompts/graph_constructor_prompt.txt").read_text(encoding="utf-8")

ALLOWED_NODES: list[str] = json.load(Path("./datasets/allowed_node_types.json").open(encoding="utf-8"))
ALLOWED_RELATIONSHIPS: list[str] = json.load(Path("./datasets/allowed_relationship_types.json").open(encoding="utf-8"))

CREDS = Credentials.from_service_account_file(
    filename=Path("./datasets/google_credentials.json"), 
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

GEMINI_2_5_PRO = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", project="mscare-469417", location="us-central1", 
    credentials=CREDS, vertexai=True, temperature=0.0
)

GEMINI_2_5_FLASH = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", project="mscare-469417", location="us-central1", 
    credentials=CREDS, vertexai=True, temperature=0.0
)

PMID_NODES_MAP: dict[str, dict[str, dict[str, str]]] = {}

print("\nLoading nodes ...", end=" ")

nodes_dir = Path("./nodes")
for json_file_path in nodes_dir.glob("*.json"):
    PMID_NODES_MAP.update(json.load(json_file_path.open(encoding="utf-8")))

print(f"{len(PMID_NODES_MAP)} nodes were loaded.")

def is_rate_limit_error(e: Exception) -> bool:

    keywords: list[str] = \
    [
        "429", "rate limit", "ratelimit", "resourceexhausted", "quota",
        "too many requests", "temporarily unavailable", "service unavailable",
        "deadline exceeded", "timeout", "api connection", "connection error"
    ]

    return any(w in f"{type(e).__name__}: {e}".lower() for w in keywords)

async def async_retry(coroutine_func, retries: int, max_delay: float) -> None:
    
    last_error_messege = None
    for attempt in range(retries + 1):
        try: return await coroutine_func()

        except Exception as e:
            last_error_messege = e
            print(f"Last error messege: '{last_error_messege}'")
            if attempt >= retries or not is_rate_limit_error(e): raise

            delay = min(max_delay, 2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25))
            await asyncio.sleep(max(0.0, delay))

    raise last_error_messege

class GraphConstructor:

    def __init__(self, llm_name: str, retries: int, max_delay: float) -> None:

        self.llm_name: str = llm_name
        self.llm: BaseChatModel = None

        if llm_name == "Gemini 2.5 Pro": self.llm = GEMINI_2_5_PRO
        if llm_name == "Gemini 2.5 Flash": self.llm = GEMINI_2_5_FLASH
        if self.llm is None: raise ValueError(f"Invalid LLM option: '{llm_name}'.")

        self.retries: int = retries
        self.max_delay: float = max_delay

        self.file_lock = asyncio.Lock()

    async def construct_knowledge_graph(self, pmid: str, save_to_disk: bool) -> list[dict[str, str]]:

        additional_instructions: str = self._get_additional_instructions(PMID_NODES_MAP[pmid])

        llm_graph_transformer = LLMGraphTransformer(
            llm=self.llm, additional_instructions=additional_instructions,
            allowed_nodes=ALLOWED_NODES, allowed_relationships=ALLOWED_RELATIONSHIPS,
            relationship_properties=["evidence", "confidence"],
        )

        title: str = PMID_TO_PAPER[pmid]["Title"]
        abstract: str = PMID_TO_PAPER[pmid]["Abstract"]

        async def _run_once():

            return await llm_graph_transformer.aconvert_to_graph_documents(
                [Document(page_content=f"## {title}\n\nPMID: {pmid}\n\n### Abstract\n\n{abstract}")]
            )

        start_time = time.time()

        try:
            graph_docs: list[GraphDocument] = await async_retry(
            _run_once, self.retries, self.max_delay
        )
        except Exception as e:
            print(f"[FAIL!] {pmid}: {type(e).__name__}: {e}"); return []

        print(f"[DONE!] {pmid}: {(time.time() - start_time):.2f} secs.")

        edges: list[dict[str, str]] = self._format_edges(PMID_NODES_MAP[pmid], graph_docs)
        
        if save_to_disk: await self._save_edges(pmid, edges)

        return edges

    async def batch_construct(self, pmids: list[str], max_concurrent: int, save_to_disk: bool) -> dict[str, list[dict[str, str]]]:
        
        semaphore = asyncio.Semaphore(max_concurrent)

        pmid_edges_map: dict[str, list[dict[str, str]]] = {}

        async def _task(pmid: str) -> None:

            async with semaphore:

                edges = await self.construct_knowledge_graph(
                    pmid, save_to_disk=False
                )
                pmid_edges_map[pmid] = edges

                if save_to_disk: await self._save_edges(pmid, edges)

        await asyncio.gather(*[_task(p) for p in pmids])

        return pmid_edges_map

    def _get_additional_instructions(self, nodes: dict[str, dict[str, str]]) -> str:

        node_idx = 0
        lines: list[str] = []

        for node_attributes in nodes.values():

            if node_attributes["Node_Name"] == "selenomethylselenocysteine":

                node_name, node_type, linked_term = "Mesenchymal Stem Cells", "Anatomy", "MSC"
                node_definition = "An undifferentiated stromal cell with the ability to develop into the cells that form distinct mesenchymal tissues; such as bone, muscle, connective tissue, blood vessels, and lymphatic tissue."
            
            else:

                node_name = node_attributes["Node_Name"]
                node_type = node_attributes["Node_Type"]
                node_definition = node_attributes["Node_Definition"]
                linked_term = node_attributes["Corresponding_Term_in_the_Paper"]

            node_idx += 1

            line: str = (
                f"*Candidate {node_idx}:*\n\n"
                f"* *Node Name:* **{node_name}**\n"
                f"* *Node Type:* **{node_type}**\n"
                f"* *Node Definition:* {node_definition}\n"
                f"* *Corresponding Term in the Paper:* {linked_term}"
            )
            lines.append(line.replace("{", "{{").replace("}", "}}"))

        return PROMPT.format(umls_concepts="\n\n".join(lines))

    def _format_edges(self, nodes: dict[str, dict[str, str]], graph_docs: list[GraphDocument]) -> list[dict[str, str]]:
        
        if not nodes or not graph_docs: return []
        
        lower_node_info_map: dict[str, dict[str, str]] = {}

        for node_attributes in nodes.values():

            node_key = node_attributes["Node_Name"].lower()

            lower_node_info_map[node_key] = \
            {
                "node_name": node_attributes["Node_Name"],
                "node_type": node_attributes["Node_Type"]
            }

        edges: list[dict[str, str]] = []

        for relationship in graph_docs[0].relationships:

            lower_source_name: str = relationship.source.id.strip().lower()
            lower_target_name: str = relationship.target.id.strip().lower()

            if lower_source_name not in lower_node_info_map: continue
            if lower_target_name not in lower_node_info_map: continue

            source_info: dict[str, str] = lower_node_info_map[lower_source_name]
            target_info: dict[str, str] = lower_node_info_map[lower_target_name]

            properties = getattr(relationship, "properties", None) or {}
            properties = {str(k).strip().lower(): v for k, v in properties.items()}

            edges.append(
                {
                    "Source_Node": source_info["node_name"],
                    "Source_Type": source_info["node_type"],
                    "Relationship_Type": relationship.type.upper(),
                    "Target_Node": target_info["node_name"],
                    "Target_Type": target_info["node_type"],
                    "Evidence": properties.get("evidence", ""),
                    "Confidence": properties.get("confidence", ""),
                }
            )
        
        return edges

    async def _save_edges(self, pmid: str, edges: list[dict[str, str]]) -> None:
        
        async with self.file_lock:

            save_dir = Path("edges")

            for topic in PMID_TO_TOPIC[pmid].split(";"):

                filepath = save_dir / f"{topic}.json"
                save_dir.mkdir(parents=True, exist_ok=True)

                if filepath.exists(): pmid_edges_map = json.loads(filepath.read_text(encoding="utf-8"))
                else: pmid_edges_map: dict[str, list[dict[str, str]]] = {}

                pmid_edges_map[pmid] = edges

                filepath.write_text(json.dumps(pmid_edges_map, ensure_ascii=False, indent=4), encoding="utf-8")

async def run_graph_construction(llm_name: str, pmids: list[str] = None, 
                                 max_concurrent: int = 10, retries: int = 5, max_delay: float = 30.0, 
                                 save_to_disk: bool = True) -> dict[str, list[dict[str, str]]]:

    graph_constructor = GraphConstructor(llm_name, retries=retries, max_delay=max_delay)

    if not pmids: pmids = sorted(PMID_NODES_MAP.keys(), key=int, reverse=True)

    pmid_edges_map = await graph_constructor.batch_construct(
        pmids, max_concurrent=max_concurrent, 
        save_to_disk=save_to_disk
    )
    
    return pmid_edges_map

if __name__ == "__main__":

    edges_path = Path("./edges/TWHM.json")

    existing_pmids: set[str] = set()

    if edges_path.exists():

        edges: dict[str, list[dict[str, str]]] = json.loads(edges_path.read_text(encoding="utf-8"))

        existing_pmids: set[str] = set(edges.keys())

    target_pmids: list[str] = []

    for pmid, topic in PMID_TO_TOPIC.items():

        if "TWHM" in topic.split(";") and pmid not in existing_pmids:
            
            if pmid in PMID_NODES_MAP: target_pmids.append(pmid)

    target_pmids: list[str] = sorted(target_pmids, key=int, reverse=True)

    asyncio.run(
        run_graph_construction(
            llm_name="Gemini 2.5 Pro",
            pmids=target_pmids,
            max_concurrent=100,
        )
    )