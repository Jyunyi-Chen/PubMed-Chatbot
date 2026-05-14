import math

import pandas as pd
import networkx as nx

from pyvis.network import Network
from spacy.language import Language

from umls_concept_linking import get_umls_concepts

# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_CLUSTER_RADIUS = 300
BASE_INNER_RING_RADIUS = 120
PIXELS_PER_CHAR = 4

MIN_SIZE = 20
MAX_SIZE = 50


class GraphVisualizer:

    def __init__(self, ner_model: Language, digraph: nx.DiGraph, graph: nx.Graph) -> None:

        self.ner_model = ner_model
        self.digraph = digraph
        self.graph = graph

        self.node_type_to_color: dict[str, str] = \
        {
            "Anatomy": "#BF092F",
            "Chemicals & Drugs": "#F87B1B",
            "Disorders": "#FFC400",
            "Genes & Molecular Sequences": "#B0CE88",
            "Living Beings": "#9ECFD4",
            "Physiology": "#F5D2D2",
            "Procedures": "#D78FEE"
        }

    # =====================================================================
    # Main entry point
    # =====================================================================
    def generate_pyvis_html(self, search_nodes: set[str], filtered_edges: dict, save_path: str) -> None:

        # Build local subgraph directly from filtered_edges
        sub_digraph: nx.DiGraph = nx.DiGraph()
        for (u, u_type, label, v, v_type), pmids in filtered_edges.items():
            sub_digraph.add_node(u, type=u_type)
            sub_digraph.add_node(v, type=v_type)
            sub_digraph.add_edge(u, v, label=label, pmids=";".join(sorted(pmids)))

        # Ensure search nodes always appear (even if isolated)
        for n in search_nodes:
            if n not in sub_digraph and n in self.digraph:
                sub_digraph.add_node(n, type=self.digraph.nodes[n]["type"])

        all_nodes: set[str] = set(sub_digraph.nodes())
        other_nodes: set[str] = all_nodes - search_nodes

        # -- Cluster assignment --
        clusters, orphans = self._assign_clusters(
            search_nodes=list(search_nodes),
            response_nodes=list(other_nodes),
            subgraph=sub_digraph,
        )

        # -- Compute positions --
        positions = self._compute_cluster_positions(clusters, orphans, sub_digraph)

        # -- Compute node sizes (linearly mapped from degree to MIN_SIZE ~ MAX_SIZE) --
        node_to_degree: dict[str, int] = dict(sub_digraph.degree())
        max_degree: int = max(max(node_to_degree.values(), default=1), 1)
        min_degree: int = min(node_to_degree.values(), default=0)

        node_to_size: dict[str, float] = {}
        for node, deg in node_to_degree.items():
            if max_degree == min_degree:
                node_to_size[node] = (MIN_SIZE + MAX_SIZE) / 2
            else:
                ratio = (deg - min_degree) / (max_degree - min_degree)
                node_to_size[node] = MIN_SIZE + ratio * (MAX_SIZE - MIN_SIZE)

        # -- Build Pyvis Network --
        pyvis_network = Network(
            height="1000px",
            width="100%",
            directed=True,
            bgcolor="#2B2B2B",
            font_color="white",
        )

        # Physics: WEAK stabilization only to fix edge label positions,
        # then freeze. Strong values (e.g. -8000) would override pre-computed positions.
        pyvis_network.set_options("""
        {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000,
                    "updateInterval": 25
                },
                "barnesHut": {
                    "gravitationalConstant": -500,
                    "springLength": 400,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.1
                }
            }
        }
        """)

        # -- Add nodes --
        for node in all_nodes:

            is_search_node = node in search_nodes
            shape = "star" if is_search_node else "triangle"
            size_ = node_to_size.get(node, MIN_SIZE)
            node_type = sub_digraph.nodes[node]["type"]
            color = self.node_type_to_color.get(node_type, "#AAAAAA")
            title = f"Node Name: {node}\nNode Type: {node_type}\nDegree: {node_to_degree.get(node, 0)}"
            x, y = positions.get(node, (0, 0))

            pyvis_network.add_node(
                n_id=node,
                label=node,
                shape=shape,
                size=size_,
                color={
                    "background": color,
                    "border": color,
                    "highlight": {"background": "#FFFFFF", "border": "#FFFFFF"},
                },
                title=title,
                borderWidth=0,
                borderWidthSelected=3,
                x=x,
                y=y,
                font={"color": "white", "size": 12, "face": "Arial"},
                mass=8 if is_search_node else 1,
            )

        # -- Add edges (keep ALL edges from subgraph, no filtering) --
        edge_pair_count: dict[tuple[str, str], int] = {}

        for src_node, tgt_node, edge_data in sub_digraph.edges(data=True):

            edge_label: str = edge_data.get("label", "")
            edge_pmids: str = str(edge_data.get("pmids", ""))
            edge_title: str = f"PMIDs: {edge_pmids.replace(';', ', ')}." if edge_pmids else ""

            # Assign different roundness for parallel edges between the same node pair
            pair_key = (min(src_node, tgt_node), max(src_node, tgt_node))
            idx = edge_pair_count.get(pair_key, 0)
            edge_pair_count[pair_key] = idx + 1
            roundness = 0.15 + idx * 0.15

            pyvis_network.add_edge(
                src_node,
                tgt_node,
                label=edge_label,
                title=edge_title,
                length=200,
                color={"color": "#777C6D", "opacity": 0.9},
                width=2,
                font={"color": "#CCCCCC", "size": 10, "align": "top", "strokeWidth": 0},
                arrows="to",
                smooth={"type": "curvedCW", "roundness": roundness},
            )

        # -- Save HTML --
        html_path = save_path.replace(".json", ".html")
        pyvis_network.save_graph(html_path)

        # -- Inject legend and stabilization-freeze JS --
        self._inject_legend_and_stabilize_js(html_path)

    # =====================================================================
    # Clustering: assign response nodes to search node clusters by connectivity
    # =====================================================================
    def _assign_clusters(
        self,
        search_nodes: list[str],
        response_nodes: list[str],
        subgraph: nx.DiGraph,
    ) -> tuple[dict[str, list[str]], list[str]]:
        """
        Rules:
        1. Treat the subgraph as undirected; check whether each response node
           is in the same connected component as a search node.
        2. Connected to exactly one search node -> assign to that cluster.
        3. Connected to multiple search nodes -> assign to the one with a direct edge (priority).
        4. Connected to no search node -> assign to the orphan group.
        """
        undirected = subgraph.to_undirected()
        components = list(nx.connected_components(undirected))

        search_comp: dict[str, set[str]] = {}
        for s in search_nodes:
            for comp in components:
                if s in comp:
                    search_comp[s] = comp
                    break

        clusters: dict[str, list[str]] = {s: [] for s in search_nodes}
        orphans: list[str] = []

        for r in response_nodes:
            connected_searches = [
                s for s in search_nodes
                if s in search_comp and r in search_comp[s]
            ]

            if len(connected_searches) == 0:
                orphans.append(r)
            elif len(connected_searches) == 1:
                clusters[connected_searches[0]].append(r)
            else:
                best = max(
                    connected_searches,
                    key=lambda s: subgraph.has_edge(s, r) + subgraph.has_edge(r, s),
                )
                clusters[best].append(r)

        return clusters, orphans

    # =====================================================================
    # Compute cluster layout positions (radii dynamically adjusted by label length)
    # =====================================================================
    def _compute_cluster_positions(
        self,
        clusters: dict[str, list[str]],
        orphans: list[str],
        subgraph: nx.DiGraph,
    ) -> dict[str, tuple[float, float]]:

        positions: dict[str, tuple[float, float]] = {}
        n_clusters = len(clusters)
        if n_clusters == 0:
            return positions

        # -- Helper functions --
        def node_label_len(n: str) -> int:
            return len(n)  # node id itself is the label

        def edge_label_len(u: str, v: str) -> int:
            if subgraph.has_edge(u, v):
                return len(subgraph.edges[u, v].get("label", ""))
            return 0

        # -- Compute ring radius for each cluster --
        cluster_ring_radii: dict[str, float] = {}

        for s, members in clusters.items():
            all_in_cluster = [s] + members
            max_node_chars = max(node_label_len(n) for n in all_in_cluster)

            max_edge_chars = 0
            cluster_set = set(all_in_cluster)
            for u, v in subgraph.edges():
                if u in cluster_set and v in cluster_set:
                    max_edge_chars = max(max_edge_chars, edge_label_len(u, v))

            label_padding = max(max_node_chars, max_edge_chars) * PIXELS_PER_CHAR
            n_members = max(len(members), 1)
            min_radius = (n_members * label_padding * 1.2) / (2 * math.pi)
            cluster_ring_radii[s] = max(BASE_INNER_RING_RADIUS, min_radius)

        # -- Compute inter-cluster distance --
        max_ring = max(cluster_ring_radii.values()) if cluster_ring_radii else 0

        node_to_cluster: dict[str, str] = {}
        for s, members in clusters.items():
            for n in [s] + members:
                node_to_cluster[n] = s

        max_cross_edge_chars = 0
        for u, v in subgraph.edges():
            cu = node_to_cluster.get(u)
            cv = node_to_cluster.get(v)
            if cu is not None and cv is not None and cu != cv:
                max_cross_edge_chars = max(max_cross_edge_chars, edge_label_len(u, v))

        cross_edge_padding = max_cross_edge_chars * PIXELS_PER_CHAR
        cluster_radius = max(BASE_CLUSTER_RADIUS, max_ring * 1.8 + cross_edge_padding)

        # -- Layout each cluster --
        for idx, (search_node, resp_nodes) in enumerate(clusters.items()):
            angle = 2 * math.pi * idx / n_clusters
            cx = cluster_radius * math.cos(angle)
            cy = cluster_radius * math.sin(angle)
            positions[search_node] = (cx, cy)

            ring_r = cluster_ring_radii[search_node]
            n_resp = len(resp_nodes)
            for j, rn in enumerate(resp_nodes):
                a = 2 * math.pi * j / max(n_resp, 1)
                positions[rn] = (cx + ring_r * math.cos(a), cy + ring_r * math.sin(a))

        # -- Orphan nodes placed at the center --
        if orphans:
            max_orphan_chars = max(node_label_len(n) for n in orphans)
            orphan_padding = max_orphan_chars * PIXELS_PER_CHAR
            n_orph = len(orphans)
            orphan_radius = max(80, (n_orph * orphan_padding * 2) / (2 * math.pi)) if n_orph > 1 else 0
            for j, on in enumerate(orphans):
                a = 2 * math.pi * j / max(n_orph, 1)
                positions[on] = (orphan_radius * math.cos(a), orphan_radius * math.sin(a))

        return positions

    # =====================================================================
    # Inject legend + post-stabilization freeze JS
    # =====================================================================
    def _inject_legend_and_stabilize_js(self, html_path: str) -> None:

        legend_items = ""
        for t, c in self.node_type_to_color.items():
            legend_items += (
                f'<div style="display:flex;align-items:center;margin:4px 0;">'
                f'<span style="display:inline-block;width:14px;height:14px;'
                f"background:{c};border-radius:2px;margin-right:8px;\"></span>"
                f'<span style="color:#EEE;font-size:13px;">{t}</span></div>'
            )

        legend_html = f"""
        <div id="legend" style="
            position:fixed; bottom:20px; right:20px;
            background:rgba(30,30,30,0.92); border:1px solid #555;
            border-radius:8px; padding:14px 18px;
            font-family:Arial,sans-serif; z-index:9999;
            box-shadow: 0 2px 12px rgba(0,0,0,0.5);
        ">
            <div style="color:#FFF;font-weight:bold;margin-bottom:8px;font-size:14px;">
                Node Types
            </div>
            {legend_items}
            <hr style="border-color:#555;margin:10px 0;">
            <div style="display:flex;align-items:center;margin:4px 0;">
                <span style="color:#FFD700;font-size:16px;margin-right:8px;">&#9733;</span>
                <span style="color:#EEE;font-size:13px;">Search Node</span>
            </div>
            <div style="display:flex;align-items:center;margin:4px 0;">
                <span style="color:#AAA;font-size:16px;margin-right:8px;">&#9650;</span>
                <span style="color:#EEE;font-size:13px;">Response Node</span>
            </div>
        </div>
        """

        stabilize_js = """
        <script>
        network.once("stabilizationIterationsDone", function () {
            network.setOptions({ physics: { enabled: false } });
        });
        </script>
        """

        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = content.replace("</body>", legend_html + stabilize_js + "\n</body>")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)

    # =====================================================================
    # Extract response nodes from response text
    # =====================================================================
    def _get_response_nodes(self, response: str, max_nodes: int = 15, window_size: int = 100, overlap: int = 10) -> set[str]:

        node_dfs: list[pd.DataFrame] = []

        words: list[str] = response.split()

        for idx in range(0, len(words), window_size - overlap):

            chunk: str = " ".join(words[idx:idx + window_size])

            node_dfs.append(get_umls_concepts(chunk, self.ner_model))

        if not node_dfs: return set()

        response_nodes_df = pd.concat(node_dfs, ignore_index=True)

        if "UMLS_Concept_Name" not in response_nodes_df.columns: return set()

        response_nodes_df = response_nodes_df.drop_duplicates(subset=["UMLS_Concept_Name"])

        valid_nodes: list[str] = [n for n in response_nodes_df["UMLS_Concept_Name"] if n in self.digraph]

        valid_nodes.sort(key=lambda n: self.digraph.degree(n), reverse=True)

        return set(valid_nodes[:max_nodes])