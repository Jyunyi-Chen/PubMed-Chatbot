import os
import json
import spacy
import warnings
import lemminflect

spacy.require_gpu(0)

os.environ["NUMEXPR_MAX_THREADS"] = "16"

warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from spacy.tokens.span import Span
from collections import defaultdict
from spacy.language import Language

from abbreviation_extractor import extract_abbreviation_definition_pairs

NLP_PIPELINE: Language = spacy.load("en_core_sci_scibert", disable=["ner", "parser"])

PAPERS: dict[str, dict[str, str]] = {}

for json_file_path in Path("./pubmed_papers").glob("*.json"):
    PAPERS.update(json.loads(json_file_path.read_text(encoding="utf-8")))

def _extract_abbreviations_from_paper(pmid: str) -> dict[str, str]:

    text: str = PAPERS[pmid]["Abstract"]

    abbreviation_to_definition: dict[str, str] = {}

    for pair in extract_abbreviation_definition_pairs(text):

        if len(pair.abbreviation) < len(pair.definition.split()):

            if not _is_allowed_pos(NLP_PIPELINE(pair.definition)[:]): continue

        if len(pair.abbreviation.split()) >= 2 or ", " in pair.definition: continue

        definition = " ".join(w.lower() if w.istitle() else w for w in pair.definition.split())

        if pair.abbreviation not in abbreviation_to_definition: abbreviation_to_definition[pair.abbreviation] = definition

    return abbreviation_to_definition

def _is_allowed_pos(span: Span) -> bool:

    token_pos_list: list[str] = [span[i].pos_ for i in range(len(span))]

    if not set(token_pos_list).issubset({"PUNCT", "VERB", "ADJ", "ADP", "NOUN", "PROPN"}): return False

    first_pos, last_pos = token_pos_list[0], token_pos_list[-1]

    if first_pos == "ADP" or last_pos == "ADP": return False

    if last_pos not in ["NOUN", "PROPN"]: return False

    if len(token_pos_list) >= 2:

        if token_pos_list[0] == "VERB" and token_pos_list[1] != "ADJ": return False
        
        if "VERB" in token_pos_list[1:]: return False

    return True

def _extract_abbreviations_from_paper_dbs() -> dict[str, dict[str, int]]:
    
    raw_abbreviation_to_definition_to_count: dict[str, Counter] = defaultdict(Counter)

    for pmid in tqdm(PAPERS.keys(), desc="Extracting Abbreviations", unit="paper"):

        abbreviation_to_definition: dict[str, str] = _extract_abbreviations_from_paper(pmid)

        for abbreviation, definition in abbreviation_to_definition.items():
            
            raw_abbreviation_to_definition_to_count[abbreviation][definition] += 1

    abbreviation_to_definition_to_count: dict[str, dict[str, int]] = {}

    for abbreviation in sorted(raw_abbreviation_to_definition_to_count.keys()):

        definitions_counts = raw_abbreviation_to_definition_to_count[abbreviation].items()

        definitions_counts = sorted([(d, c) for d, c in definitions_counts if c >= 10], key=lambda kv: kv[1], reverse=True)

        if definitions_counts: abbreviation_to_definition_to_count[abbreviation] = dict(definitions_counts)

    content: str = json.dumps(abbreviation_to_definition_to_count, ensure_ascii=False, indent=4)

    Path("./datasets/abbreviations.json").write_text(content, encoding="utf-8")

    return abbreviation_to_definition_to_count

if __name__ == "__main__":

    _extract_abbreviations_from_paper_dbs()
