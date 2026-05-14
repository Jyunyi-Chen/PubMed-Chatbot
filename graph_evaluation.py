import time
import json
import asyncio
import datetime

from pathlib import Path
from collections.abc import Coroutine
from graph_construction import GraphConstructor
from langchain_core.messages.ai import AIMessage
from google.oauth2.service_account import Credentials
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

from pubmed_papers import PMID_TO_PAPER
from graph_construction import PMID_NODES_MAP

load_dotenv()

SAVE_DIR = Path("evaluation_logs")

CREDS = Credentials.from_service_account_file(
    filename=Path("./datasets/google_credentials.json"), 
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

GEMINI_3_PRO = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview", project="mscare-469417", response_mime_type="application/json",
    location="global", credentials=CREDS, vertexai=True, temperature=0.0
)

PROMPT: str = Path("./prompts/graph_evaluator_prompt.txt").read_text(encoding="utf-8")

class KGEvaluator:

    async def _evaluate_edge(self, pmid: str, edge: dict[str, dict[str, str] | str]) -> dict[str, str]:
    
        prompt: str = PROMPT.format(
            title=PMID_TO_PAPER[pmid]["Title"],
            abstract=PMID_TO_PAPER[pmid]["Abstract"],
            relationship_type=edge["Relationship_Type"],
            source_node_name=edge["Source_Node"]["Node_Name"],
            target_node_name=edge["Target_Node"]["Node_Name"],
            source_node_type=edge["Source_Node"]["Node_Type"],
            target_node_type=edge["Target_Node"]["Node_Type"],
            source_node_definition=edge["Source_Node"]["Node_Definition"],
            target_node_definition=edge["Target_Node"]["Node_Definition"],
            source_node_linked_term=edge["Source_Node"]["Corresponding_Term_in_the_Paper"],
            target_node_linked_term=edge["Target_Node"]["Corresponding_Term_in_the_Paper"]
        )

        triplet: str = (
            f"({edge['Source_Node']['Node_Name']})"
            f" --[{edge['Relationship_Type']}]--> "
            f"({edge['Target_Node']['Node_Name']})"
        )
    
        for attempt in range(3):

            try:
                response: AIMessage = await GEMINI_3_PRO.ainvoke(prompt)
         
                response_content: dict[str, str] = json.loads(response.content[0]["text"])

                results: dict[str, str] = {
                    "pmid": pmid, "triplet": triplet,
                    "result": response_content["result"],
                    "reasoning": response_content["reasoning"]
                }
                
                return results
            
            except Exception as e:

                print(f"⚠️ KGEvaluator error ({pmid}): {e}")
                
                if attempt == 2:

                    results: dict[str, str] = {
                        "pmid": pmid, "triplet": triplet,
                        "result": "ERROR", "reasoning": f"Exception: {str(e)}"
                    }

                    return results
                
                await asyncio.sleep(1)

        return {"pmid": pmid, "triplet": triplet, "evaluation": "ERROR", "reasoning": "Max retries reached."}

    async def _evaluate_paper_edges(self, pmid: str, edges: list[dict[str, str]], 
                                    semaphore: asyncio.Semaphore) -> list[dict[str, str]]:
        
        nodes: dict[str, dict[str, str]] = {}

        for node_attributes in PMID_NODES_MAP[pmid].values():
            nodes[node_attributes["Node_Name"]] = node_attributes

        async with semaphore:

            tasks: list[dict[str, str]] = []

            for edge in edges:

                tasks.append(
                    self._evaluate_edge(
                        pmid=pmid, edge={
                            "Source_Node": nodes[edge["Source_Node"]],
                            "Relationship_Type": edge["Relationship_Type"],
                            "Target_Node": nodes[edge["Target_Node"]]
                        }
                    )
                )
            
            return await asyncio.gather(*tasks)

    async def run_evaluation(self, llm_name: str, pmid_edges_map: dict[str, list[dict[str, str]]], 
                             max_concurrent: int) -> None:
        
        semaphore = asyncio.Semaphore(max_concurrent)

        tasks: list[Coroutine[None, None, list[dict[str, str]]]] = [
            self._evaluate_paper_edges(pmid, edges, semaphore) 
            for pmid, edges in pmid_edges_map.items()
        ]

        edges_results: list[dict[str, str]] = []
        
        print(f"Starting evaluation of {sum(len(e) for e in pmid_edges_map.values())} edges ...")

        for idx, coroutine in enumerate(asyncio.as_completed(tasks)):
            paper_results: list[dict[str, str]] = await coroutine
            edges_results.extend(paper_results)
            
            if (idx + 1) % 10 == 0: print(f"Progress: [{idx + 1}/{len(tasks)}] papers done.")

        if edges_results: print()
        else: return None

        statistics: dict[str, int] = \
        {
            "CORRECT": 0,
            "NO_RELATIONSHIP": 0,
            "WRONG_RELATIONSHIP_TYPE": 0, 
            "WRONG_EDGE_DIRECTION": 0,
            "EVALUATION_ERROR": 0,
            "UNKNOWN_RESULT": 0
        }

        for results in edges_results:
            result_type: str = results.get("result", "UNKNOWN_RESULT")
            if result_type in statistics: statistics[result_type] += 1
            else: statistics["UNKNOWN_RESULT"] += 1

        n_edges = len(edges_results)
        n_correct = statistics["CORRECT"]
        precision = n_correct / n_edges if n_edges > 0 else 0

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename = SAVE_DIR / f"{time_str}.json"

        print("=================================")
        print("|       EVALUATION REPORT       |")
        print("=================================")
        print(f"| Total Edges:             {n_edges:4d} |")
        print(f"| Precision:               {precision:4.2f} |")
        print("---------------------------------")
        print(f"| Correct:                 {statistics['CORRECT']:4d} |")
        print(f"| No Relationship:         {statistics['NO_RELATIONSHIP']:4d} |")
        print(f"| Wrong Relationship Type: {statistics['WRONG_RELATIONSHIP_TYPE']:4d} |")
        print(f"| Wrong Edge Direction:    {statistics['WRONG_EDGE_DIRECTION']:4d} |")
        print(f"| Evaluation Error:        {statistics['EVALUATION_ERROR']:4d} |")
        print(f"| Unknown Result:          {statistics['UNKNOWN_RESULT']:4d} |")
        print("=================================")

        report_data: dict[str, any] = \
        {
            "meta": {
                "timestamp": time_str,
                "construction_llm": llm_name,
                "total_papers": len(pmid_edges_map),
                "total_edges": n_edges,
                "precision": precision,
                "stats": statistics
            },
            "details": edges_results
        }

        filename.write_text(json.dumps(report_data, indent=4, ensure_ascii=False), encoding="utf-8")
        
        print(f"\nDetailed report saved to: {filename}\n")

async def run_pipeline_evaluation(llm_name: str, pmids: list[str] = None, max_concurrent: int = 10, 
                                  retries: int = 5, max_delay: float = 30.0) -> None:
    
    if not pmids: pmids = sorted(PMID_NODES_MAP.keys(), key=int, reverse=True)

    print(f"\n--- Phase 1: Constructing Graphs for {len(pmids)} papers (Constructor LLM: {llm_name}) ---\n")
    
    graph_constructor = GraphConstructor(llm_name, retries=retries, max_delay=max_delay)

    pmid_edges_map = await graph_constructor.batch_construct(pmids, max_concurrent, save_to_disk=False)
    
    print(f"\nGenerated graphs for {len(pmid_edges_map)} papers.")

    start_time = time.time()

    print(f"\n--- Phase 2: Evaluating Edges (Evaluator LLM: Gemini 3 Pro) ---\n")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    kg_evaluator = KGEvaluator()

    await kg_evaluator.run_evaluation(llm_name, pmid_edges_map, max_concurrent)
    
    print(f"Done. Total time: {int(time.time() - start_time)} secs.\n")

if __name__ == "__main__":

    async def main() -> None:

        pmids = sorted(PMID_NODES_MAP.keys(), key=int, reverse=True)[:5]

        await run_pipeline_evaluation(llm_name="Gemini 2.5 Flash", pmids=pmids, max_concurrent=5)

    asyncio.run(main())
    