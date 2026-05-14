import json
import time
import spacy
import ollama
import logging
import warnings
import networkx as nx
import streamlit as st

from pathlib import Path
from collections import Counter
from collections import defaultdict
from spacy.language import Language
from scispacy.linking import EntityLinker
from scispacy.linking_utils import KnowledgeBase
from scispacy.linking_utils import UmlsKnowledgeBase
from scispacy.candidate_generation import LinkerPaths
from scispacy.abbreviation import AbbreviationDetector

from scispacy.candidate_generation import DEFAULT_PATHS
from scispacy.candidate_generation import DEFAULT_KNOWLEDGE_BASES

logging.getLogger("scispacy").setLevel(logging.ERROR)
logging.getLogger("scispacy.candidate_generation").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@st.cache_resource
def load_cache_resource() -> tuple[Language, dict[str, dict[str, dict[str, int]]], dict[str, dict[str, nx.DiGraph | nx.Graph]]]:
    
    ner_pipeline: Language = load_ner_pipeline()

    topic_to_pmid_to_node_weights: dict[str, dict[str, dict[str, int]]] = {}
    topic_to_graphs: dict[str, dict[str, nx.DiGraph | nx.Graph]] = {}

    start_time = time.time()
    print("\nLoading Nodes, Edges, and GraphMLs ...")

    for topic in ["CHA", "MRT", "MSC", "TRIP", "TWHM"]:

        nodes_path = Path(f"./nodes/{topic}.json")
        edges_path = Path(f"./edges/{topic}.json")
        graph_path = Path(f"./knowledge_graphs/{topic}.graphml")

        if not graph_path.exists() or not nodes_path.exists() or not edges_path.exists(): continue

        # --- 1. Edges --- #

        pmid_to_node_weights: dict[str, dict[str, int]] = {}

        with open(edges_path, "r", encoding="utf-8") as f:
            pmid_to_edges: dict[str, list[dict[str, str]]] = json.load(f)

        for pmid, edges in pmid_to_edges.items():
            node_counter = Counter()

            for edge in edges:
                node_counter[edge["Source_Node"]] += 1
                node_counter[edge["Target_Node"]] += 1
            
            pmid_to_node_weights[pmid] = dict(node_counter)
        
        topic_to_pmid_to_node_weights[topic] = pmid_to_node_weights

        # --- 2. Graphs --- #

        digraph: nx.DiGraph = nx.read_graphml(graph_path)

        topic_to_graphs[topic] = {"digraph": digraph, "graph": digraph.to_undirected()}

    print(f"Done. {time.time() - start_time:.2f} seconds taken.")

    return ner_pipeline, topic_to_pmid_to_node_weights, topic_to_graphs

def load_ner_pipeline() -> Language:

    linker_data_dir = Path("./umls_2025ab")

    custom_linker_paths_2025ab = LinkerPaths(
        ann_index=str(linker_data_dir / "nmslib_index.bin"),
        tfidf_vectors=str(linker_data_dir / "tfidf_vectors_sparse.npz"),
        tfidf_vectorizer=str(linker_data_dir / "tfidf_vectorizer.joblib"),
        concept_aliases_list=str(linker_data_dir / "concept_aliases.json"),
    )

    class UMLS2025ABKnowledgeBase(KnowledgeBase):
        def __init__(self): super().__init__(str(linker_data_dir / "umls_2025ab.jsonl"))

    DEFAULT_PATHS["umls2025ab"] = custom_linker_paths_2025ab
    DEFAULT_KNOWLEDGE_BASES["umls2025ab"] = UMLS2025ABKnowledgeBase

    start_time = time.time()
    print("\nInitializing NER pipeline ...")

    ner_pipeline: Language = spacy.load("en_core_sci_scibert")

    ner_pipeline.add_pipe("abbreviation_detector")
    ner_pipeline.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls2025ab"})

    print(f"Done. {time.time() - start_time:.2f} seconds taken.")

    return ner_pipeline

@st.cache_resource
def warmup_ollama_models() -> None:

    for model in ["translategemma:12b", "gpt-oss:20b", "ministral-3:8b"]:

        ollama.chat(model, [], keep_alive=-1)