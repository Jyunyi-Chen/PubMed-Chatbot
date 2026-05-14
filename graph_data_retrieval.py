import itertools
import networkx as nx

from collections import defaultdict

EdgeTuple = tuple[str, str, str, str, str]

def retrieve_subgraph(search_nodes: set[str], n_max_refs: int, pmid_to_node_weights: dict[str, dict[str, int]], 
                      digraph: nx.DiGraph, graph: nx.Graph) -> tuple[list[str], dict[EdgeTuple, set[str]]]:

    subgraph_edge_to_pmids: dict[EdgeTuple, set[str]] = defaultdict(set)

    if len(search_nodes) >= 2:

        for source_node, target_node in itertools.combinations(search_nodes, 2):

            edge_to_pmids = _get_node_pair_paths(digraph, graph, source_node, target_node)

            for edge, pmids in edge_to_pmids.items(): subgraph_edge_to_pmids[edge].update(pmids)
    else:

        source_node = list(search_nodes)[0]

        for target_node in graph.neighbors(source_node):
            for u, v in [(source_node, target_node), (target_node, source_node)]:

                u_type = digraph.nodes[u]["type"]
                v_type = digraph.nodes[v]["type"]

                if digraph.has_edge(u, v):

                    edge_data: dict[str, str] = digraph.get_edge_data(u, v)

                    edge: EdgeTuple = (u, u_type, edge_data["label"], v, v_type)

                    subgraph_edge_to_pmids[edge].update(edge_data["pmids"].split(";"))

    pmid_to_edges: dict[str, set[EdgeTuple]] = defaultdict(set)

    for edge, pmids in subgraph_edge_to_pmids.items():
        for pmid in pmids: pmid_to_edges[pmid].add(edge)

    def _sorting_pmids(pmid: str) -> tuple[int, int, int]:

        total_search_nodes_weight: int = sum(pmid_to_node_weights[pmid].get(n, 0) for n in search_nodes)

        n_edges_in_subgraph: int = len(pmid_to_edges[pmid])

        return (total_search_nodes_weight, n_edges_in_subgraph, int(pmid))

    pmids: list[str] = sorted(pmid_to_edges.keys(), key=_sorting_pmids, reverse=True)[:n_max_refs]

    selected_pmids_set: set[str] = set(pmids)

    filtered_edges: dict[EdgeTuple, set[str]] = {}

    for edge, edge_pmids in subgraph_edge_to_pmids.items():

        intersect = edge_pmids.intersection(selected_pmids_set)

        if intersect: filtered_edges[edge] = intersect

    return pmids, filtered_edges

def _get_node_pair_paths(digraph: nx.DiGraph, graph: nx.Graph, source_node: str, target_node: str) -> dict[EdgeTuple, set[str]]:
    
    edge_to_pmids: dict[EdgeTuple, set[str]] = defaultdict(set)

    try:
        path_generator = nx.all_shortest_paths(graph, source_node, target_node)
        for path in path_generator:
            for u, v in zip(path, path[1:]):
                u_type = digraph.nodes[u]["type"]
                v_type = digraph.nodes[v]["type"]
                if digraph.has_edge(u, v):
                    edge_data: dict[str, str] = digraph.get_edge_data(u, v)
                    edge: EdgeTuple = (u, u_type, edge_data["label"], v, v_type)
                    edge_to_pmids[edge].update(edge_data["pmids"].split(";"))
                if digraph.has_edge(v, u):
                    edge_data: dict[str, str] = digraph.get_edge_data(v, u)
                    edge: EdgeTuple = (v, v_type, edge_data["label"], u, u_type)
                    edge_to_pmids[edge].update(edge_data["pmids"].split(";"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    return edge_to_pmids