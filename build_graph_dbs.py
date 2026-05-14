import json
import networkx as nx

from pathlib import Path
from collections import Counter

def convert_edges_to_graphml(topic: str):

    G = nx.DiGraph()

    edges_path = Path(f"./edges/{topic}.json")
    
    pmid_to_edges: dict[str, list[dict[str, str]]] = json.loads(edges_path.read_text(encoding="utf-8"))

    edge_buffer: dict[tuple[str, str], dict[str, Counter | set[str]]] = {}

    for pmid, edges in pmid_to_edges.items():

        for edge in edges:

            source_node: str = edge["Source_Node"]
            target_node: str = edge["Target_Node"]

            if source_node == target_node: continue

            G.add_node(source_node, type=edge["Source_Type"].split(";")[0])
            G.add_node(target_node, type=edge["Target_Type"].split(";")[0])

            adjacent_nodes: tuple[str, str] = (source_node, target_node)

            if adjacent_nodes not in edge_buffer: edge_buffer[adjacent_nodes] = {"relationships": Counter(), "pmids": set()}

            edge_buffer[adjacent_nodes]["relationships"][edge["Relationship_Type"]] += 1
            edge_buffer[adjacent_nodes]["pmids"].add(pmid)

    print(f"\nConstructing graph edges (topic: {topic}) ...")

    for (source_node, target_node), attributes in edge_buffer.items():

        most_common_relationship, count = attributes["relationships"].most_common(1)[0]
        if (count / attributes["relationships"].total()) <= 0.8: continue

        pmids = ";".join(sorted(attributes["pmids"], key=int, reverse=True))

        G.add_edge(source_node, target_node, label=most_common_relationship, pmids=pmids)

    G.remove_nodes_from(list(nx.isolates(G)))

    graph_path = Path(f"./knowledge_graphs/{topic}.graphml")
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    nx.write_graphml(G, graph_path, encoding="utf-8")

    print(f"Successfully wrote GraphML to: {graph_path.absolute()}")
    print(f"Total Nodes: {G.number_of_nodes()} | Total Edges: {G.number_of_edges()}\n")

if __name__ == "__main__":

    convert_edges_to_graphml(topic="CHA")
    convert_edges_to_graphml(topic="MRT")
    convert_edges_to_graphml(topic="TRIP")
