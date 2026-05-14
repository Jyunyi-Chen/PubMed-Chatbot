import json
import pathlib

TOPIC_TO_PMID_TO_PAPER: dict[str, dict[str, dict[str, str]]] = {}
PMID_TO_PAPER: dict[str, dict[str, str]] = {}
PMID_TO_TOPIC: dict[str, str] = {}

pubmed_dir = pathlib.Path("./pubmed_papers")
for json_file_path in pubmed_dir.glob("*.json"):
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        topic_papers: dict[str, dict[str, str]] = json.load(f)

    topic: str = json_file_path.stem # MRT | MSC | ...
    TOPIC_TO_PMID_TO_PAPER[topic] = topic_papers
    PMID_TO_PAPER.update(topic_papers)

    for pmid in topic_papers:
        if pmid not in PMID_TO_TOPIC: PMID_TO_TOPIC[pmid] = topic
        else: PMID_TO_TOPIC[pmid] += ";" + topic
