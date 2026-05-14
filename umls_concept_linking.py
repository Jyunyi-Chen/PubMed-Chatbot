import os
import re
import json
import logging
import warnings

os.environ["NUMEXPR_MAX_THREADS"] = "16"

logging.getLogger().setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch as T
import numpy as np
import pandas as pd

from pathlib import Path
from spacy.tokens import Doc
from spacy.tokens.span import Span
from spacy.language import Language
from spacy.tokens.token import Token

from scispacy.linking import EntityLinker
from scispacy.linking_utils import Entity
from scispacy.linking_utils import UmlsKnowledgeBase
from scispacy.candidate_generation import MentionCandidate

from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

ABBREVIATIONS: dict[str, dict[str, int]] = json.loads(Path("./datasets/abbreviations.json").read_text(encoding="utf-8"))

CODER_TOKENIZER: BertTokenizerFast = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")

CODER_MODEL: BertModel = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")

CODER_MODEL.to(DEVICE).eval()

TUIS: dict[str, dict[str, str]] = json.loads(Path("./datasets/umls_tuis.json").read_text(encoding="utf-8"))

ALLOW_NODES: list[str] = json.loads(Path("./datasets/allowed_node_types.json").read_text(encoding="utf-8"))

HARD_GROUPS: list[str] = ["Chemicals & Drugs", "Genes & Molecular Sequences"]

def get_umls_concepts(text: str, ner_pipeline: Language, max_window_size: int = 10) -> pd.DataFrame:

    linker: EntityLinker = ner_pipeline.get_pipe("scispacy_linker")
    knowledge_base: UmlsKnowledgeBase = linker.kb

    doc: Doc = ner_pipeline(text)

    founded_abbrs: set[str] = set()
    founded_terms: set[str] = set()

    founded_concepts: dict[str, dict[str, str]] = {}

    for abbr in [abbr.text for abbr in doc._.abbreviations]:
        founded_abbrs.add(span_to_singular_term(ner_pipeline(abbr)[:]))
    
    n_tokens = len([tok.text for tok in doc])
    for window_size in range(max_window_size, 0, -1):
        for idx in range(n_tokens - window_size + 1):
            
            span: Span = doc[idx : idx + window_size]
            
            if not is_allowed_pos(span): continue
            singular_term = span_to_singular_term(span)
            term = " ".join(w.lower() if w.istitle() else w for w in singular_term.split())

            if any(not re.search(r"[A-Za-z0-9]", word) for word in term.split()): continue
            if term.isalnum() and term[0].isdigit() and (len(term) <= 3): continue
            if not re.search(r"[a-zA-Z]", term): continue
            if term in founded_terms: continue
            if len(term) <= 2: continue

            skip_this = False

            for abbr in founded_abbrs:
                remain = term.replace(f"{abbr}", "")
                if not re.search(r"[A-Za-z0-9]", remain):
                    skip_this = True
                    break

            for founded_term in founded_terms:
                if term in founded_term:
                    skip_this = True
                    break
                
            if skip_this: continue

            if ABBREVIATIONS.get(term):

                best_definition: str = max(ABBREVIATIONS[term], key=ABBREVIATIONS[term].get)

                singular_term = span_to_singular_term(ner_pipeline(best_definition)[:])
                
                term = " ".join(w for w in singular_term.split())

            # scispaCy's default similarity calculation method (TF-IDF + n-gram)
            candidates: list[MentionCandidate] = linker.candidate_generator([term], k=20)[0]
            prior_candidates = [c for c in candidates if c.similarities[0] >= 0.95]
            if not candidates: continue

            if not prior_candidates: # this implies that the character makeup of the term might be special
                for candidate in candidates: # for example, a gene name or some chemical compound
                    entity: Entity = knowledge_base.cui_to_entity[candidate.concept_id]
                    if not any(TUIS[t]["Semantic_Group"] in HARD_GROUPS for t in entity.types): continue
                    if _check_nums_overlap(term, entity.canonical_name):
                        if term[0] != entity.canonical_name[0]: continue
                        prior_candidates.append(candidate)

            best_candidate: MentionCandidate = None
            best_similarity_1: float = -1.0
            best_similarity_2: float = -1.0

            if prior_candidates: # skip UMLS-BERT to save time
                # it means that we already found some good concepts for that term

                best_candidate = max(prior_candidates, key=lambda c: c.similarities[0])
                best_similarity_1 = best_candidate.similarities[0]

            else: # use UMLS-BERT to enhance search ability
                term_vector: np.ndarray = get_entities_embeddings([term])[0]

                for candidate in candidates:
                    entity: Entity = knowledge_base.cui_to_entity[candidate.concept_id]

                    candidate_names: list[str] = []
                    for alias in [entity.canonical_name] + entity.aliases:
                        if len(alias.split()) == window_size: candidate_names.append(alias)
                    if not candidate_names: continue

                    candidate_vectors: np.ndarray = get_entities_embeddings(candidate_names)
                    max_similarity_2 = max(get_cos_similarity(term_vector, v) for v in candidate_vectors)

                    if not any(TUIS[t]["Semantic_Group"] in HARD_GROUPS for t in entity.types):
                        if max_similarity_2 < 0.98: continue
                    else: # to ensure we have a chance to resolve the difficult terms
                        if window_size == 1 and max_similarity_2 < 0.95: continue
                        if window_size >= 1 and max_similarity_2 < 0.98: continue

                    if max_similarity_2 > best_similarity_2:
                        best_similarity_1 = candidate.similarities[0]
                        best_similarity_2 = max_similarity_2
                        best_candidate = candidate

            if not best_candidate: continue
            
            entity: Entity = knowledge_base.cui_to_entity[best_candidate.concept_id]

            if not all(TUIS[t]["Semantic_Group"] in ALLOW_NODES for t in entity.types): continue

            # to avoid results like:
            # stem -> Scanning Transmission Electron Microscopy Procedures
            if term.islower() and window_size == 1 and best_similarity_1 > 0.95:
                if len(entity.canonical_name) > len(term) * 5 and term not in entity.aliases:
                    if term.title() not in entity.aliases and term.upper() in entity.aliases: continue

            concept_data = {
                "Linked_Term": term,
                "UMLS_CUI": best_candidate.concept_id,
                "UMLS_Concept_Name": entity.canonical_name,
                "Cos_Similarity_1": f"{best_similarity_1:.3f}",
                "Cos_Similarity_2": f"{best_similarity_2:.3f}",
                "UMLS_Semantic_Type": ";".join([TUIS[t]["Semantic_Type"] for t in entity.types]),
                "UMLS_Semantic_Group": ";".join([TUIS[t]["Semantic_Group"] for t in entity.types]),
                "UMLS_Aliases": ";".join(entity.aliases),
                "UMLS_Definition": entity.definition
            }
            if entity.canonical_name in founded_concepts:
                if len(term.split()) == len(entity.canonical_name.split()):
                    founded_concepts[entity.canonical_name] = concept_data
                    founded_terms.add(term)
            else:
                founded_concepts[entity.canonical_name] = concept_data
                founded_terms.add(term)

    if not founded_concepts: return pd.DataFrame()
    
    umls_concepts_df = pd.DataFrame(list(founded_concepts.values()))

    return umls_concepts_df.sort_values(by="Linked_Term").reset_index(drop=True)

def is_allowed_pos(span: Span) -> bool:

    token_pos_list: list[str] = [span[i].pos_ for i in range(len(span))]

    if not set(token_pos_list).issubset({"PUNCT", "VERB", "ADJ", "ADP", "NOUN", "PROPN"}): return False

    first_pos, last_pos = token_pos_list[0], token_pos_list[-1]

    if first_pos == "ADP" or last_pos == "ADP": return False

    if last_pos not in ["NOUN", "PROPN"]: return False

    if len(token_pos_list) >= 2:

        if token_pos_list[0] == "VERB" and token_pos_list[1] != "ADJ": return False
        
        if "VERB" in token_pos_list[1:]: return False

    return True

def span_to_singular_term(span: Span) -> str:

    last_plural_token: Token = None
    token_idx = len(span) - 1 # from end to start
    while (not last_plural_token) and (token_idx >= 0):
        if span[token_idx].tag_ not in ["NNS", "NNPS"]: token_idx -= 1
        else: last_plural_token: Token = span[token_idx] # then exit the loop
    
    if not last_plural_token: return span.text # no change

    last_plural_vocab: str = last_plural_token.text # letter-case not fixed
    lemma_: str = last_plural_token.lemma_ # are all lowercase
    
    idx = 0
    while idx < min(len(last_plural_vocab), len(lemma_)):
        if last_plural_vocab[idx].lower() != lemma_[idx].lower(): break
        else: idx += 1

    last_singular_vocab: str = last_plural_vocab[:idx] + lemma_[idx:]

    prefix_r_idx: int = last_plural_token.idx - span.start_char # prefix is "stem "
    suffix_l_idx: int = prefix_r_idx + len(last_plural_token.text) # suffix is ", human"

    # For example, suppose our span.text is "stem cells, human"

    prefix: str = span.text[:prefix_r_idx] # this will be: "stem "
    suffix: str = span.text[suffix_l_idx:] # this will be: ", human"
    
    return prefix + last_singular_vocab + suffix # "stem cells, human" -> "stem cell, human"

def _check_nums_overlap(term: str, concept_name: str) -> bool:

    # find all contiguous digit-sequences
    longest_num_in_term: str = max(re.findall(r"\d+", term), key=len, default="###")
    if len(longest_num_in_term) == 1 or longest_num_in_term == "###": return False

    # For example, suppose our term is miR-15a-5p, we will check
    # whether the string "15" is contained in the concept name

    nums_in_concept: str = re.findall(r"\d+", concept_name)
    if not nums_in_concept: return False

    if longest_num_in_term in nums_in_concept: return True
    else: return False

def get_entities_embeddings(entities: list[str]) -> np.ndarray:

    encoded_inputs: BatchEncoding = CODER_TOKENIZER(entities, padding=True, truncation=False, return_tensors="pt")
    encoded_inputs = {a: b.to(DEVICE) for a, b in encoded_inputs.items()}

    with T.no_grad(): outputs: BaseModelOutputWithPoolingAndCrossAttentions = CODER_MODEL(**encoded_inputs)

    token_embeddings: T.Tensor = outputs.last_hidden_state
    attention_mask: T.Tensor = encoded_inputs["attention_mask"]

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    sum_embeddings = T.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = T.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return (sum_embeddings / sum_mask).cpu().numpy()

def get_cos_similarity(v1: np.ndarray, v2: np.ndarray) -> float:

    raw_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return float((raw_similarity + 1) / 2) # range: (0.0, 1.0)

if __name__ == "__main__":

    from streamlit_functions import load_ner_pipeline

    from pubmed_papers import PMID_TO_PAPER

    ner_pipeline: Language = load_ner_pipeline()

    text: str = PMID_TO_PAPER["38602231"]["Title"] + "\n\n" + PMID_TO_PAPER["38602231"]["Abstract"]

    umls_concepts: pd.DataFrame = get_umls_concepts(text, ner_pipeline)

    umls_concepts = umls_concepts[["Linked_Term", "UMLS_CUI", "UMLS_Concept_Name"]]

    print(f"\n{umls_concepts}\n")

    text: str = PMID_TO_PAPER["23124998"]["Title"] + "\n\n" + PMID_TO_PAPER["23124998"]["Abstract"]

    umls_concepts: pd.DataFrame = get_umls_concepts(text, ner_pipeline)

    umls_concepts = umls_concepts[["Linked_Term", "UMLS_CUI", "UMLS_Concept_Name"]]

    print(f"\n{umls_concepts}\n")