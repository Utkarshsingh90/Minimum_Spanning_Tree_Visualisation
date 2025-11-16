"""
Spanning Trees Explorer — Streamlit app (step-by-step visualizer)
(Updated: fixes for sliders, missing function, and robustness)
"""
from typing import List, Tuple, Dict
import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import io
import itertools
import heapq

st.set_page_config(page_title="Spanning Trees Explorer (Step-by-step)", layout="wide")

# ------------------------ Utilities ------------------------

def parse_edge_list(text: str) -> nx.Graph:
    G = nx.Graph()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in lines:
        # accept formats: u v  or u,v  or u - v or u->v
        sep = None
        for s in ["->", "-", ",", " "]:
            if s in ln:
                sep = s
                break
        if sep is None:
            # single vertex
            try:
                v = int(ln)
            except ValueError:
                v = ln
            G.add_node(v)
        else:
            parts = [p.strip() for p in ln.split(sep) if p.strip()]
            if len(parts) >= 2:
                u, v = parts[0], parts[1]
                # try int conversion
                try:
                    u = int(u)
                except:
                    pass
                try:
                    v = int(v)
                except:
                    pass
                G.add_edge(u, v)
    return G


def laplacian_matrix(G: nx.Graph, nodes: List) -> np.ndarray:
    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}
    L = np.zeros((n, n), dtype=object)  # object so Bareiss stays exact
    for u in nodes:
        i = idx[u]
        L[i, i] = G.degree(u)
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        L[i, j] = -1
        L[j, i] = -1
    # ensure zeros are ints (not numpy floats)
    for i in range(n):
        for j in range(n):
            if L[i, j] == 0:
                L[i, j] = 0
    return L


def bareiss_det(A: np.ndarray) -> int:
    # Bareiss algorithm for exact integer determinant (fraction-free)
    n = A.shape[0]
    if n == 0:
        return 1
    M = np.array(A, copy=True, dtype=object)
    if n == 1:
        return int(M[0, 0])
    prev = 1
    for k in range(n - 1):
        pivot = M[k, k]
        if pivot == 0:
            swapped = False
            for i in range(k + 1, n):
                if M[i, k] != 0:
                    M[[k, i]] = M[[i, k]]
                    pivot = M[k, k]
                    prev = -prev
                    swapped = True
                    break
            if not swapped:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                # exact fraction-free update
                M[i, j] = (M[i, j] * pivot - M[i, k] * M[k, j]) // prev
        prev = pivot
    det = int(M[n - 1, n - 1])
    return det


def matrix_tree_count(G: nx.Graph) -> int:
    if not nx.is_connected(G):
        return 0
    nodes = list(G.nodes())
    L = laplacian_matrix(G, nodes)
    Lc = L[:-1, :-1]
    det = bareiss_det(Lc)
    return det

# ------------------------ Prüfer helpers ------------------------

def prufer_random_tree(n: int, rng=None) -> nx.Graph:
    """Generate a random labelled tree on nodes 1..n via a random Prüfer sequence."""
    if rng is None:
        rng = random
    if n <= 0:
        return nx.Graph()
    if n == 1:
        G = nx.Graph()
        G.add_node(1)
        return G
    seq = [rng.randint(1, n) for _ in range(n - 2)]
    T, _ = prufer_decode_steps(seq)
    return T


def prufer_encode_steps(tree: nx.Graph) -> Tuple[List[int], List[Tuple]]:
    """
    Returns (final sequence, steps).
    Each step: (removed_leaf, neighbor_recorded, current_nodes_list)
    Uses a heap keyed by stringified label to be robust when labels are mixed types.
    """
    G = tree.copy()
    nodes = sorted(list(G.nodes()), key=lambda x: str(x))
    n = len(nodes)
    if n <= 2:
        return ([], [])
    deg = {v: G.degree(v) for v in nodes}
    # heap of (str(label), label) to avoid type compare between ints & strings
    leaves = [(str(v), v) for v in nodes if deg[v] == 1]
    heapq.heapify(leaves)
    seq = []
    steps = []
    for _ in range(n - 2):
        if not leaves:
            break
        _, leaf = heapq.heappop(leaves)
        nbrs = list(G.neighbors(leaf))
        if not nbrs:
            # isolated somehow
            continue
        nbr = nbrs[0]
        seq.append(nbr)
        steps.append((leaf, nbr, sorted(list(G.nodes()), key=lambda x: str(x))))
        G.remove_node(leaf)
        deg.pop(leaf, None)
        deg[nbr] -= 1
        if deg[nbr] == 1:
            heapq.heappush(leaves, (str(nbr), nbr))
    return seq, steps


def prufer_decode_steps(seq: List[int]) -> Tuple[nx.Graph, List[Tuple[int, int, List[int]]]]:
    """
    Decode a Prüfer sequence into a tree.
    Returns (tree_graph, steps) where each step is (leaf_connected, connected_to, current_node_list)
    Nodes are 1..n (standard Prüfer labels).
    """
    m = len(seq)
    n = m + 2
    nodes = list(range(1, n + 1))
    deg = {i: 1 for i in nodes}
    for x in seq:
        if x in deg:
            deg[x] += 1
    leaves = [i for i in nodes if deg[i] == 1]
    heapq.heapify(leaves)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    steps = []
    for x in seq:
        leaf = heapq.heappop(leaves)
        G.add_edge(leaf, x)
        steps.append((leaf, x, sorted([v for v in nodes if deg[v] > 0])))
        deg[leaf] -= 1
        deg[x] -= 1
        if deg[x] == 1:
            heapq.heappush(leaves, x)
    a = heapq.heappop(leaves)
    b = heapq.heappop(leaves)
    G.add_edge(a, b)
    steps.append((a, b, sorted([v for v in nodes if deg[v] > 0])))
    return G, steps

# ------------------------ Wilson's algorithm (step-by-step) ------------------------

def loop_erased_path(path: List) -> List:
    pos = {}
    res = []
    for v in path:
        if v in pos:
            idx = pos[v]
            res = res[:idx + 1]
            pos = {res[i]: i for i in range(len(res))}
        else:
            pos[v] = len(res)
            res.append(v)
    return res


def wilson_steps(G: nx.Graph, root=None, rng=None) -> Tuple[nx.Graph, List[Dict]]:
    """
    Return (tree, steps) where steps is a list of dicts:
      {'start', 'walk' (list of snapshots), 'loop_erased', 'before_tree_nodes', 'added_edges'}
    """
    if not nx.is_connected(G):
        raise ValueError("Wilson's algorithm requires a connected graph")
    if rng is None:
        rng = random
    G_nodes = list(G.nodes())
    if root is None:
        root = G_nodes[0]
    tree_nodes = {root}
    tree_edges = []
    neighbors = {v: list(G.neighbors(v)) for v in G_nodes}
    steps = []
    for v in G_nodes:
        if v in tree_nodes:
            continue
        path = [v]
        cur = v
        walk_steps = []
        # random walk until hit tree
        while cur not in tree_nodes:
            nbrs = neighbors.get(cur, [])
            if not nbrs:
                # isolated / dead end -- shouldn't happen due to connectivity check
                break
            nbr = rng.choice(nbrs)
            path.append(nbr)
            cur = nbr
            walk_steps.append(list(path))
        le = loop_erased_path(path)
        added_edges = [(le[i], le[i+1]) for i in range(len(le)-1)]
        steps.append({
            'start': v,
            'walk': walk_steps,
            'loop_erased': le.copy(),
            'before_tree_nodes': sorted(list(tree_nodes), key=lambda x: str(x)),
            'added_edges': added_edges
        })
        for a, b in added_edges:
            tree_edges.append((a, b))
            tree_nodes.add(a)
            tree_nodes.add(b)
    T = nx.Graph()
    T.add_nodes_from(G_nodes)
    T.add_edges_from(tree_edges)
    return T, steps

# ------------------------ Enumerate spanning trees (small graphs) ------------------------

def enumerate_spanning_trees_bruteforce(G: nx.Graph, max_trees: int = 2000) -> List[nx.Graph]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m > 20:
        # Using st here is okay since function is used in UI, but keep message short
        st.warning('Graph has many edges; enumeration may be slow. Limit edges to <=20 for enumeration.')
    edges = list(G.edges())
    trees = []
    for combo in itertools.combinations(edges, n - 1):
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(combo)
        if nx.is_tree(H):
            trees.append(H)
            if len(trees) >= max_trees:
                break
    return trees

# ------------------------ Drawing helpers ------------------------

def draw_graph_highlight(G: nx.Graph, highlight_edges: List[Tuple] = None, highlight_nodes: List = None, figsize=(6,4), title: str = None):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=250)
    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), alpha=0.3)
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    if highlight_edges:
        # ensure edges are presentable (convert to list)
        nx.draw_networkx_edges(G, pos, edgelist=list(highlight_edges), width=3.0, edge_color='red')
    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(highlight_nodes), node_color='orange', node_size=300)
    if title:
        plt.title(title)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


def draw_graph_and_tree(G: nx.Graph, T: nx.Graph = None, edge_freq: Dict[Tuple, float] = None, title: str = None, figsize=(8,6)):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=250)
    base_edges = list(G.edges())
    if edge_freq is None:
        nx.draw_networkx_edges(G, pos, edgelist=base_edges, alpha=0.4)
    else:
        freqs = []
        keys = []
        for e in base_edges:
            a, b = e
            key = (a,b) if (a,b) in edge_freq else ((b,a) if (b,a) in edge_freq else (a,b))
            f = edge_freq.get(key, 0)
            freqs.append(f)
            keys.append(key)
        maxf = max(freqs) if len(freqs) > 0 else 1
        widths = [0.5 + 6 * (f / maxf) for f in freqs]
        nx.draw_networkx_edges(G, pos, edgelist=base_edges, width=widths)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=maxf))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Edge frequency')
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    if T is not None:
        tree_edges = list(T.edges())
        nx.draw_networkx_edges(G, pos, edgelist=tree_edges, width=2.5, alpha=0.9, edge_color='red')
    if title:
        plt.title(title)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# ------------------------ Streamlit UI ------------------------

st.title('Spanning Trees Explorer — Step-by-step')
st.markdown('Explore construction of spanning trees: Prüfer (for K_n) and Wilson (general) with interactive steps, and enumerate small graphs.')

with st.sidebar:
    st.header('Graph input')
    preset = st.selectbox('Choose preset graph', ['Custom', 'Complete graph K_n', 'Path P_n', 'Cycle C_n', 'Grid 4x4', 'Erdos-Renyi G(n,p)', 'Small example (cube)'])
    if preset == 'Custom':
        edge_text = st.text_area('Enter edge list (one per line, e.g. "1 2" or "1,2" or "a-b"). Single vertices allowed.', value='1 2 ')
        G = parse_edge_list(edge_text)
    elif preset == 'Complete graph K_n':
        nn = st.number_input('n', min_value=2, value=6, step=1, key='kn_n')
        G = nx.complete_graph(int(nn))
        G = nx.relabel_nodes(G, {i: i+1 for i in range(int(nn))})
    elif preset == 'Path P_n':
        nn = st.number_input('n', min_value=2, value=6, step=1, key='path_n')
        G = nx.path_graph(int(nn))
        G = nx.relabel_nodes(G, {i: i+1 for i in range(int(nn))})
    elif preset == 'Cycle C_n':
        nn = st.number_input('n', min_value=3, value=6, step=1, key='cycle_n')
        G = nx.cycle_graph(int(nn))
        G = nx.relabel_nodes(G, {i: i+1 for i in range(int(nn))})
    elif preset == 'Grid 4x4':
        G = nx.grid_2d_graph(4, 4)
        G = nx.convert_node_labels_to_integers(G, first_label=1)
    elif preset == 'Erdos-Renyi G(n,p)':
        nn = st.number_input('n', min_value=2, value=12, step=1, key='er_n')
        p = st.slider('p', min_value=0.0, max_value=1.0, value=0.15, key='er_p')
        G = nx.erdos_renyi_graph(int(nn), float(p), seed=42)
        G = nx.relabel_nodes(G, {i: i+1 for i in range(int(nn))})
    elif preset == 'Small example (cube)':
        G = nx.cubical_graph()
        G = nx.relabel_nodes(G, {i: i+1 for i in range(G.number_of_nodes())})

    st.markdown('---')
    st.header('Matrix-Tree')
    compute_count = st.button('Compute number of spanning trees (exact)')

    st.markdown('---')
    st.header('Sampling / Step-by-step')
    sample_mode = st.selectbox('Sampling method', ['Wilson (any connected graph)', 'Prüfer (complete graphs only)'])
    sample_seed = st.number_input('Random seed (0 to use Python RNG)', min_value=0, value=0, step=1)
    if sample_mode == 'Prüfer (complete graphs only)':
        st.write('Prüfer step-by-step shows the leaf removal sequence and decoded construction.')
        show_prufer_steps = st.checkbox('Show Prüfer encode steps', value=True)
        show_prufer_decode = st.checkbox('Show Prüfer decode steps', value=False)
        n_samples = 1
    else:
        st.write('Wilson step-by-step shows random walks and loop-erased paths added to the tree.')
        show_wilson_steps = st.checkbox('Show Wilson steps', value=True)
        n_samples = st.number_input('Number of Wilson samples to run (for final frequency heatmap)', min_value=1, value=100, step=1)

    st.markdown('---')
    st.header('Enumeration (small graphs)')
    enumerate_toggle = st.checkbox('Enumerate all spanning trees (small graphs only)', value=False)
    enum_limit = st.number_input('Max trees to enumerate', min_value=1, value=200, step=1)

# Basic info and preview
cols = st.columns([1,2])
with cols[0]:
    st.subheader('Graph info')
    st.write(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
    if not nx.is_connected(G):
        st.warning('Graph is not connected — Matrix-Tree count = 0 and Wilson requires a connected graph (you may sample per component).')
with cols[1]:
    st.subheader('Graph preview')
    buf0 = draw_graph_highlight(G, None, None, title='Input graph')
    st.image(buf0)

# Matrix-Tree count
if compute_count:
    with st.spinner('Computing Laplacian and determinant...'):
        try:
            count = matrix_tree_count(G)
            st.success(f'Number of spanning trees (τ(G)) = {count}')
        except Exception as e:
            st.error(f'Error computing count: {e}')

# Prüfer mode
if sample_mode == 'Prüfer (complete graphs only)':
    n = G.number_of_nodes()
    # require labeled 1..n complete graph for standard Prüfer
    if not nx.is_connected(G) or G.number_of_edges() != n * (n - 1) // 2:
        st.error('Prüfer requires complete graph K_n (labels 1..n). Choose preset K_n.')
    else:
        tree_sample = None
        # Show encode steps
        if show_prufer_steps:
            random.seed(sample_seed if sample_seed != 0 else None)
            T = prufer_random_tree(n)
            seq, steps = prufer_encode_steps(T)
            st.subheader('Prüfer encoding (leaf removals)')
            st.write('Sequence length =', len(seq))
            if len(steps) == 0:
                st.write('No encoding steps (n <= 2).')
            else:
                if len(steps) == 1:
                    step_idx = 0
                else:
                    step_idx = st.slider('Encoding step', min_value=0, max_value=len(steps)-1, value=0)
                rem_leaf, neighbor, nodes_list = steps[step_idx]
                st.write(f'Removed leaf: {rem_leaf} — neighbor recorded in sequence: {neighbor}')
                buf = draw_graph_highlight(T, highlight_edges=[(rem_leaf, neighbor)] if neighbor is not None else None, highlight_nodes=[rem_leaf], title=f'Encoding step {step_idx+1}')
                st.image(buf)
            st.write('Full Prüfer sequence:', seq)
            tree_sample = T
        # Show decode steps (constructive)
        if show_prufer_decode:
            random.seed(sample_seed if sample_seed != 0 else None)
            seq = [random.randint(1, n) for _ in range(n-2)] if n > 2 else []
            G_decoded, dsteps = prufer_decode_steps(seq)
            st.subheader('Prüfer decoding (constructive)')
            st.write('Sequence:', seq)
            if len(dsteps) == 0:
                st.write('No decoding steps (n <= 2).')
            else:
                if len(dsteps) == 1:
                    dec_idx = 0
                else:
                    dec_idx = st.slider('Decoding step', min_value=0, max_value=len(dsteps)-1, value=0)
                leaf, connect_to, nodes_now = dsteps[dec_idx]
                st.write(f'Connect leaf {leaf} to {connect_to}')
                buf = draw_graph_highlight(G_decoded, highlight_edges=[(leaf, connect_to)], title=f'Decoding step {dec_idx+1}')
                st.image(buf)
        # final tree display
        if 'tree_sample' in locals() and tree_sample is not None:
            st.subheader('Final sampled tree (from Prüfer)')
            buff = draw_graph_highlight(tree_sample, title='Sampled tree')
            st.image(buff)

# Wilson mode
if sample_mode == 'Wilson (any connected graph)':
    if nx.is_connected(G):
        random.seed(sample_seed if sample_seed != 0 else None)
        T_final, wilson_steps_list = wilson_steps(G, root=None)
        st.subheader('Wilson algorithm steps')
        if show_wilson_steps:
            st.write('Each step shows: random walk from a start vertex until it hits the existing tree, loop-erased path, and edges added.')
            if len(wilson_steps_list) == 0:
                st.write('No Wilson steps recorded (graph trivial or root covers all nodes).')
            else:
                # Step selector safe: if only one element, don't show a slider
                if len(wilson_steps_list) == 1:
                    step_choice = 0
                else:
                    step_choice = st.slider('Wilson step index', min_value=0, max_value=len(wilson_steps_list)-1, value=0)
                stepd = wilson_steps_list[step_choice]
                st.write(f"Start vertex: {stepd['start']}")
                st.write('Number of walk snapshots recorded:', len(stepd['walk']))
                # show snapshots of the walk
                if len(stepd['walk']) == 0:
                    st.write('No walk snapshots for this step (walk immediately hit tree).')
                    path_snapshot = stepd['loop_erased']
                    snap_idx = 0
                else:
                    if len(stepd['walk']) == 1:
                        snap_idx = 0
                    else:
                        snap_idx = st.slider('Walk snapshot index (0 = first)', min_value=0, max_value=len(stepd['walk'])-1, value=len(stepd['walk'])-1)
                    path_snapshot = stepd['walk'][snap_idx]
                st.write('Path snapshot:', path_snapshot)
                buf_walk = draw_graph_highlight(G, highlight_edges=[(path_snapshot[i], path_snapshot[i+1]) for i in range(len(path_snapshot)-1)] if len(path_snapshot)>1 else None, highlight_nodes=path_snapshot, title='Walk snapshot (before loop erasure)')
                st.image(buf_walk)
                # show loop-erased path
                st.write('Loop-erased path (to be added):', stepd['loop_erased'])
                buf_le = draw_graph_highlight(G, highlight_edges=[(stepd['loop_erased'][i], stepd['loop_erased'][i+1]) for i in range(len(stepd['loop_erased'])-1)] if len(stepd['loop_erased'])>1 else None, highlight_nodes=stepd['loop_erased'], title='Loop-erased path')
                st.image(buf_le)
                st.write('Edges added in this step:', stepd['added_edges'])
        # show final tree
        st.subheader('Final tree produced by Wilson (one sample)')
        buf_final = draw_graph_highlight(T_final, title='Wilson final tree')
        st.image(buf_final)
        # run multiple samples and show edge-frequency heatmap if requested
        run_many = st.button('Run many Wilson samples and show edge-frequency heatmap')
        if run_many:
            ns = int(n_samples)
            edge_freq = {}
            for i in range(ns):
                Ti, _ = wilson_steps(G, root=None)
                for a,b in Ti.edges():
                    key = (a,b) if (a,b) in edge_freq else ((b,a) if (b,a) in edge_freq else (a,b))
                    edge_freq[key] = edge_freq.get(key, 0) + 1
            buf_heat = draw_graph_and_tree(G, T=None, edge_freq=edge_freq, title=f'Wilson edge frequencies ({ns} samples)')
            st.image(buf_heat)
            st.write('Edge counts:', edge_freq)
    else:
        st.error('Graph not connected — Wilson requires a connected graph.')

# Enumeration
if enumerate_toggle:
    if G.number_of_edges() > 20:
        st.warning('Graph has many edges; enumeration may be slow or infeasible. Reduce edges or nodes for enumeration.')
    else:
        st.subheader('Enumerating spanning trees (brute-force)')
        with st.spinner('Enumerating...'):
            trees = enumerate_spanning_trees_bruteforce(G, max_trees=int(enum_limit))
        st.write(f'Found {len(trees)} spanning trees (showing up to {enum_limit}).')
        cols = st.columns(3)
        for i, T in enumerate(trees):
            buf = draw_graph_highlight(T, title=f'Tree {i+1}')
            cols[i % 3].image(buf)

st.markdown('---')
st.write('Notes: Step-by-step visuals are intended for educational demonstrations on small graphs. For large graphs, steps and enumerations may be slow.')
st.write('References: Kirchhoff (Matrix-Tree), Cayley (n^{n-2}), Prüfer sequence, Wilson (UST).')
# EOF