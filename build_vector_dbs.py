import os
import chromadb

from tqdm import tqdm
from chromadb.utils import embedding_functions

from dotenv import load_dotenv

from pubmed_papers import TOPIC_TO_PMID_TO_PAPER

load_dotenv()

DB_CLIENT = chromadb.PersistentClient("./chroma_db")

OPENAI_EF = embedding_functions.OpenAIEmbeddingFunction(os.environ.get("OPENAI_API_KEY"), "text-embedding-3-large")

if __name__ == "__main__":

    batch_size = 100

    for topic, topic_papers in TOPIC_TO_PMID_TO_PAPER.items():

        print(f"\n--- Processing topic: {topic} ---")
        
        collection = DB_CLIENT.get_or_create_collection(topic, embedding_function=OPENAI_EF)

        existing_data = collection.get(include=[])
        existing_pmids: set[str] = set(existing_data["ids"])

        new_pmids: list[str] = []
        for pmid in topic_papers.keys():
            if pmid not in existing_pmids: new_pmids.append(pmid)

        n_new_pmids = len(new_pmids)

        if n_new_pmids == 0:
            print(f"Topic '{topic}': all papers already indexed.")
            continue

        print(f"Topic '{topic}': processing {n_new_pmids} papers ...")

        for idx in tqdm(range(0, n_new_pmids, batch_size), desc=f"Indexing {topic}"):
            
            batch_metadatas: list[dict[str, str]] = []
            batch_documents: list[str] = []
            
            for pmid in new_pmids[idx:idx + batch_size]:

                batch_metadatas.append(
                    {
                        "topic": topic, "pmid": pmid, 
                        "title": topic_papers[pmid]["Title"],
                        "abstract": topic_papers[pmid]["Abstract"]
                    }
                )
                batch_documents.append(topic_papers[pmid]["Title"] + "\n" + topic_papers[pmid]["Abstract"])

            collection.add(new_pmids[idx:idx + batch_size], metadatas=batch_metadatas, documents=batch_documents)

        print(f"Finished topic '{topic}'.")