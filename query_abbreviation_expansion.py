import re
import json
import ollama

from pathlib import Path
from collections import defaultdict

from spacy.lang.en.stop_words import STOP_WORDS

CUSTOM_STOP_WORDS = STOP_WORDS.union({"hi", "hello", "hey", "dear", "thanks", "regards", "best"})

PROMPT = \
"""
### INSTRUCTION

You are a query refinement specialist for a biomedical search engine.
Your objective is to rewrite the user's search query by expanding valid abbreviations into their full names based **strictly** on the provided definitions.

---

### RULES

1. **Strict Mapping:** Use ONLY the provided "Abbreviation Definitions" to expand terms.
2. **Context Awareness:** - Choose the most appropriate full name from the provided list based on the query context.
   - If the most frequent definition fits, use it.
   - Replace the abbreviation with its full name.
   - **Exception:** If the full name is already present in the query (e.g., "MSC (Mesenchymal Stem Cell)"), DO NOT expand it. Leave it exactly as is.
3. **Output Format:** Output **ONLY** the rewritten query string. Do not include prefixes.

---

### EXAMPLES

**Example 1:**

*Abbreviation Definitions:*

- *PC:* pancreatic cancer; prostate cancer

*User:* What are the symptoms of PC?
*Assistant:* What are the symptoms of prostate cancer?

**Example 2:**

*Abbreviation Definitions:*

- *BMSCs:* bone marrow mesenchymal stem cells; bone marrow stromal cells; bone marrow-derived mesenchymal stem cells

*User:* How to isolate bone marrow mesenchymal stem cells (BMSCs) from mice?
*Assistant:* How to isolate bone marrow mesenchymal stem cells (BMSCs) from mice?

**Example 3:**

*Abbreviation Definitions:*

- *MSCs:* mesenchymal stem cells; mesenchymal stromal cells; mesenchymal stem/stromal cells

*User:* Please list the mscs as a table with pmids which derived from various tissues.
*Assistant:* Please list the mesenchymal stem cells as a table with pmids which derived from various tissues.

---

### TASK

*Abbreviation Definitions:*

{ABBREVIATION_DEFINITIONS}

*User:* {QUERY}
*Assistant:*
"""

load_path = Path("./datasets/abbreviations.json")

ABBREVIATIONS: dict[str, dict[str, int]] = json.loads(load_path.read_text(encoding="utf-8"))

temp_grouping: dict[str, list[tuple[str, dict[str, int]]]] = defaultdict(list)

for original_key, definitions in ABBREVIATIONS.items():

    temp_grouping[original_key.lower()].append((original_key, definitions))

ABBREVIATIONS_LOWER: dict[str, tuple[str, str]] = {}

for lower_key, entries in temp_grouping.items():
    
    if len(entries) > 1:

        merged_definitions: dict[str, int] = defaultdict(int)

        for _, defs in entries:
            for full_name, freq in defs.items():
                merged_definitions[full_name] += freq
        
        representative_key, final_defs_map = entries[0][0], merged_definitions

    else:

        representative_key, final_defs_map = entries[0]

    sorted_defs = sorted(final_defs_map.items(), key=lambda x: x[1], reverse=True)[:3]

    top_3_names_str = "; ".join([item[0] for item in sorted_defs])
    
    ABBREVIATIONS_LOWER[lower_key] = (representative_key, top_3_names_str)

def expand_abbreviations(english_query: str, model: str = "ministral-3:8b") -> str:

    tokens: list[str] = re.findall(r'\b[A-Za-z0-9][A-Za-z0-9-]*\b', english_query)

    relevant_abbrs: dict[str, str] = {}

    for token in tokens:
        token_lower = token.lower()

        if (token_lower in ABBREVIATIONS_LOWER) and (token_lower not in CUSTOM_STOP_WORDS):

            original_key, full_names_str = ABBREVIATIONS_LOWER[token_lower]
            relevant_abbrs[original_key] = full_names_str

    if not relevant_abbrs: return english_query

    abbreviations_definitions = "\n".join([f"- {k}: {v}" for k, v in relevant_abbrs.items()])

    messages: list[dict[str, str]] = \
    [
        {
            "role": "user",
            "content": PROMPT.format(
                ABBREVIATION_DEFINITIONS=abbreviations_definitions,
                QUERY=english_query
            )
        }
    ]

    response = ollama.chat(model, messages, options={"temperature": 0}, keep_alive=-1)

    message_content: str = response["message"]["content"]
    
    expanded_query = message_content.strip()

    return expanded_query

if __name__ == "__main__":

    test_queries: list[str] = \
    [
        "Can you explain the detailed mechanism of MSC in tissue regeneration?",
        "I am looking for the latest methods using ngs for genomic analysis.",
        "Please find papers about PC (Prostate Cancer) treatments effectively.",
        "How do MSC interact with PC cells in the tumor microenvironment?",
        "Could you help me analyze the gene expression levels in liver tissue?",
        "Hi, how are you?",
        "Hi.",
        "What is ipsc?"
    ]

    for query in test_queries:

        print(f"\n{'-' * 80}\n\n{query}\n---> {expand_abbreviations(query)}")

    print()