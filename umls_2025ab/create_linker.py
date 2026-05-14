import os
import json
import argparse

from scispacy.linking_utils import KnowledgeBase
from scispacy.candidate_generation import create_tfidf_ann_index

def main(kb_path: str, output_path: str):

    os.makedirs(output_path, exist_ok=True)
    
    print(f"\nLoading KnowledgeBase from {kb_path} ...")

    kb = KnowledgeBase(kb_path)

    print("Patching canonical names into alias map ...")

    with open(kb_path, "r", encoding="utf-8") as f:

        for line in f:
            entry = json.loads(line)
            concept_id = entry.get("concept_id")
            canonical_name = entry.get("canonical_name")
            
            if concept_id and canonical_name:
                if canonical_name not in kb.alias_to_cuis:
                    kb.alias_to_cuis[canonical_name] = [concept_id]
                elif concept_id not in kb.alias_to_cuis[canonical_name]:
                    kb.alias_to_cuis[canonical_name].append(concept_id)

    print("Starting to create TF-IDF ANN index ...")
    
    create_tfidf_ann_index(output_path, kb=kb)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_path", help="Path to the KB file.", required=True)
    parser.add_argument("--output_path", help="Path to the output directory.", required=True)

    args = parser.parse_args()
    main(args.kb_path, args.output_path)