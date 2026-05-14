import ollama

PROMPT = \
"""
### INSTRUCTION

You are an expert Query Refiner. Your task is to rewrite the user's follow-up question into a **complete, standalone question** based on the conversation history.

### CRITICAL RULES

1. **Resolve Context:** Replace pronouns ("it", "they", "this") with specific entities (genes, drugs, methods) from the Chat History.

2. **Keep Constraints:** **DO NOT REMOVE** specific instructions such as:
   - Quantities (e.g., "5 genes", "top 3 methods")
   - Formats (e.g., "list", "table", "compare")
   - Specific attributes (e.g., "human", "in vitro", "failed clinical trials")

3. **No Answering:** Do NOT answer the question. Do NOT summarize the history. Just output the refined question.

4. **Meta-Commands:** If the user says "Try again" or "Elaborate", create a query for the underlying topic of the previous turn.

---

### EXAMPLES

**Example 1**

=== Chat History ===

*User:* What does p53 do?
*Assistant:* It is a tumor suppressor protein ...

=== Chat History ===

*User:* How is it regulated?
*Assistant:* How is the p53 protein regulated?

**Example 2**

=== Chat History ===

*User:* Tell me about antibiotics.
*Assistant:* Penicillin is a common one ...

=== Chat History ===

*User:* List 3 side effects of it.
*Assistant:* List 3 side effects of Penicillin.

**Example 3**

=== Chat History ===

*User:* What works for headache?
*Assistant:* Aspirin is commonly used ...

=== Chat History ===

*User:* I need more details.
*Assistant:* Detailed mechanism of action and usage of Aspirin for headache.

**Example 4**

=== Chat History ===

*User:* What are the main types of cellular transport?
*Assistant:* The main types are: **1. Passive transport (diffusion):** ... **2. Active transport**: ... **3. Vesicle transport (endocytosis and exocytosis)**: ...

=== Chat History ===

*User:* Please elaborate on the second point.
*Assistant:* Elaborate on the mechanism and examples of active cellular transport.

---

### TASK

=== Chat History ===

{CHAT_HISTORY}

=== Chat History ===

*User:* {QUERY}
*Assistant:*
"""

def to_search_query(chat_history: list[tuple[str, str]], english_query: str, model: str = "gpt-oss:20b") -> str:

    if not chat_history: return english_query

    chat_logs: list[str] = []

    for query, response in chat_history:

        chat_logs.append(f"*User:* {query}\n*Assistant:* {response}")
        
    chat_history: str = "\n\n".join(chat_logs)

    messages: list[dict[str, str]] = \
    [
        {
            "role": "system",
            "content": PROMPT.format(
                CHAT_HISTORY=chat_history, 
                QUERY=english_query
            )
        }
    ]

    response = ollama.chat(model, messages, options={"temperature": 0}, keep_alive=-1)

    message_content: str = response["message"]["content"]

    return message_content.strip()

if __name__ == "__main__":

    chat_history: list[tuple[str, str]] = [("What are the primary sources for isolating Mesenchymal Stem Cells (MSCs)?", "MSCs can be isolated from various tissues. The most common sources are: 1. Bone Marrow (BM-MSCs), 2. Adipose Tissue (AT-MSCs), 3. Umbilical Cord Tissue (UC-MSCs), and 4. Dental Pulp (DP-MSCs).")]

    english_query = "Compare the yield and differentiation potential of the third one against the first one in a table."
    
    refined_query = to_search_query(chat_history, english_query)
    
    print(f"\nResult: {refined_query}\n")
    