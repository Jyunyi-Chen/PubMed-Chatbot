import re
import ollama
import opencc

PROMPT = \
"""
You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:


{TEXT}
"""

CH_TO_TW = opencc.OpenCC("s2twp.json")

def to_english_query(raw_query: str, model: str = "translategemma:12b") -> str:

    if not re.search(r'[\u4e00-\u9fff]', raw_query): return raw_query

    messages: list[dict[str, str]] = \
    [
        {
            "role": "user",
            "content": PROMPT.format(
                SOURCE_LANG="Chinese", SOURCE_CODE="zh-TW",
                TARGET_LANG="English", TARGET_CODE="en",
                TEXT=CH_TO_TW.convert(raw_query)
            )
        }
    ]

    response = ollama.chat(model, messages, options={"temperature": 0}, keep_alive=-1)

    message_content: str = response["message"]["content"]
    
    return message_content.strip()

def to_english_response(response: str, model: str = "translategemma:12b") -> str:

    if not re.search(r'[\u4e00-\u9fff]', response): return response

    messages: list[dict[str, str]] = \
    [
        {
            "role": "user",
            "content": PROMPT.format(
                SOURCE_LANG="Chinese", SOURCE_CODE="zh-TW",
                TARGET_LANG="English", TARGET_CODE="en",
                TEXT=CH_TO_TW.convert(response)
            )
        }
    ]

    response = ollama.chat(model, messages, options={"temperature": 0}, keep_alive=-1)

    message_content: str = response["message"]["content"]
    
    return message_content.strip()

def to_chinese_response(response: str, model: str = "translategemma:12b") -> str:

    messages: list[dict[str, str]] = \
    [
        {
            "role": "user",
            "content": PROMPT.format(
                SOURCE_LANG="English", SOURCE_CODE="en",
                TARGET_LANG="Chinese", TARGET_CODE="zh-TW", 
                TEXT=response
            )
        }
    ]

    response = ollama.chat(model, messages, options={"temperature": 0}, keep_alive=-1)

    message_content: str = CH_TO_TW.convert(response["message"]["content"])

    return message_content.strip()
