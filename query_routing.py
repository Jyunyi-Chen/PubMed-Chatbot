import ollama

PROMPT = \
"""
### INSTRUCTION

You are an intelligent query router for a biomedical research assistant. 

Your objective is to classify whether a user's input requires retrieving external scientific literature to provide an evidence-based answer.

---

### RULES

1. **Output Format:**

   * Return strictly **one word**: "YES" or "NO".
   * Do not include punctuation, markdown, explanations, or any other text.

2. **Classification Criteria - "YES" (Retrieval Needed):**

   * The query asks for specific facts, definitions, biological mechanisms, relationships, or experimental data.
   * The query seeks verification or evidence (e.g., "What papers support this?", "Is there evidence for ...").
   * The query is a technical question that requires domain-specific accuracy.

3. **Classification Criteria - "NO" (Retrieval Not Needed):**

   * **Social & Meta:** Greetings, farewells, gratitude (e.g., "Thanks", "Hi"), or questions about your identity ("Who are you?").

---

### EXAMPLES

**Example 1:**
*User:* "Hello!"
*Assistant:* NO

**Example 2:**
*User:* "Thank you."
*Assistant:* NO

**Example 3:**
*User:* "Who are you?"
*Assistant:* NO

---

### TASK

*User:* "{QUERY}"
*Assistant:*
"""

def rag_is_needed(search_query: str, model: str = "ministral-3:8b") -> bool:

    messages: list[dict[str, str]] = \
    [
        {
            "role": "user",
            "content": PROMPT.format(
                QUERY=search_query
            )
        }
    ]

    response = ollama.chat(model, messages, options={"temperature": 0}, keep_alive=-1)

    message_content: str = response["message"]["content"]

    return "YES" in message_content.upper()