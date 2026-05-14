import json
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from spacy.language import Language

from streamlit_functions import load_ner_pipeline
from umls_concept_linking import get_umls_concepts

from pubmed_papers import TOPIC_TO_PMID_TO_PAPER

TOPIC: str = "TWHM"

nodes_dir = Path("./nodes")
nodes_dir.mkdir(exist_ok=True)

nodes_path = nodes_dir / f"{TOPIC}.json"

if nodes_path.exists():

    pmid_to_nodes: dict[str, list[dict[str, str]]] = json.loads(
        nodes_path.read_text(encoding="utf-8")
    )

else:

    pmid_to_nodes: dict[str, list[dict[str, str]]] = {}

NLP_PIPELINE: Language = load_ner_pipeline()

pmid_to_paper: dict[str, dict[str, str]] = TOPIC_TO_PMID_TO_PAPER[TOPIC]

for pmid, paper in tqdm(list(pmid_to_paper.items()), desc="Linking UMLS concepts", unit="paper"):
    
    if pmid in pmid_to_nodes: continue

    text: str = paper["Title"] + "\n\n" + paper["Abstract"]

    try: umls_concept_df: pd.DataFrame = get_umls_concepts(text, NLP_PIPELINE)
    except Exception: continue

    if umls_concept_df.empty: continue

    if pmid not in pmid_to_nodes: pmid_to_nodes[pmid] = []

    for _, row in umls_concept_df.iterrows():

        pmid_to_nodes[pmid].append({
            "Node_ID": row["UMLS_CUI"],
            "Node_Name": row["UMLS_Concept_Name"],
            "Node_Type": row["UMLS_Semantic_Group"],
            "Node_Definition": row["UMLS_Definition"],
            "Corresponding_Term_in_Paper": row["Linked_Term"],
        })

        node_type: str = pmid_to_nodes[pmid][row["UMLS_CUI"]]["Node_Type"]
        pmid_to_nodes[pmid][row["UMLS_CUI"]]["Node_Type"] = node_type.split(";")[0]

nodes_path.write_text(json.dumps(pmid_to_nodes, ensure_ascii=False, indent=4), encoding="utf-8")
