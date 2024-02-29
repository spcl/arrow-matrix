import igraph
import numpy
import numpy as np


class ArrowGraph:
    """
    Represents a graph whose edges are concentrated in an arrow shape.
    This graph has been obtained by permuting the vertices of an original graph.
    If permutation[i]==j, the i-th vertex of self.graph corresponds to the vertex j in the original graph.
    """
    graph: igraph.Graph
    permutation: list[int]
    nonzero_rows: int
    arrow_width: int

    def __init__(self, graph: igraph.Graph, permutation: list[int], arrow_width: int):
        self.graph = graph
        self.permutation = permutation
        self.nonzero_rows = len([d for d in graph.degree() if d == 0])
        self.arrow_width = arrow_width

    def __getitem__(self, item):
        if item == 0:
            return self.graph
        elif item == 1:
            return self.permutation
        else:
            raise IndexError()


def arrow_decomposition(G: igraph.Graph, arrow_width: int = 512, max_number_of_levels: int = 2, block_diagonal:bool = False, prune: bool = True) -> list[ArrowGraph]:
    """
    Computes a two-level arrow decomposition of G with the given arrow_width.
    :param prune: If true, arrow_width many highest-degree vertices are pruned (and placed first in the permutation).
    :param block_diagonal: If true, the nonzeros are in arrow_width x arrow_width blocks around the diagonal.
    If false, they are within a arrow_width band around the diagonal.
    :param max_number_of_levels: the maximum number of allowed matrices in the decomposition.
    Once reached, a best effort is made to fit the remaining vertices.
    :param G: a graph to decompose.
    :param arrow_width: desired number of columns/rows to include in the arrow-head.
    Note that the last matrix in the decomposition might have a larger arrow_width.
    Check the arrow_width of the returned ArrowGraph.
    :return:
    """
    assert arrow_width <= G.vcount()

    if not "original_id" in G.vertex_attributes():
        G.vs["original_id"] = range(0, G.vcount())

    decomposition = []
    _arrow_decomposition(G, arrow_width, decomposition, max_number_of_levels, block_diagonal, prune)

    return decomposition


def get_arrow_width(g: igraph.Graph, initial_width: int) -> int:
    width = initial_width
    for e in g.es:
        if e.source > width and e.target > width:
            width = max(width, abs(e.source-e.target))
    return width


def _arrow_decomposition(G: igraph.Graph, arrow_width: int, decomposition: list[ArrowGraph], max_level: int,
                         block_diagonal: bool = False, prune: bool = True) -> None:
    """
    Recursively computes
    :param G: graph to decompose
    :param arrow_width: the desired arrow_width / block size
    :param decomposition: the result is appended to this list
    :param max_level: After this many levels, make a best effort to fit the remaining edges and stop recursing
    :param block_diagonal: If true, instead of fitting into a band around the diagonal, it fits into blocks around the diagonal
    :return:
    """
    # Compute linearization of the current level
    l1_order = _arrow_linear_order(G, arrow_width, len(decomposition)+1 >= max_level)

    # Maps from Vertex id to position in the order
    inverse_permutation = np.argsort(l1_order)

    if len(decomposition) + 1 < max_level:

        # TODO Note These es.select calls now take a large fraction of the runtime

        # BAND Criterion
        if not block_diagonal:
            l1_edges = G.es.select(lambda e: abs(inverse_permutation[e.source] - inverse_permutation[e.target]) <= arrow_width
                                             or (prune and (
                                             inverse_permutation[e.source] < arrow_width
                                             or inverse_permutation[e.target] < arrow_width)))

        # BLOCK Criterion
        else:
            l1_edges = G.es.select(lambda e: ((inverse_permutation[e.source]//arrow_width - inverse_permutation[e.target]//arrow_width) == 0)
                                             or (prune and (
                                             inverse_permutation[e.source] < arrow_width
                                             or inverse_permutation[e.target] < arrow_width)))

        if len(l1_edges) == 0:
            l1_edges = G.es
            l2_edges = []
        else:
            if not block_diagonal:
                l2_edges = G.es.select(lambda e:
                                       abs(inverse_permutation[e.source] - inverse_permutation[e.target]) > arrow_width
                                       and (inverse_permutation[e.source] >= arrow_width or not prune)
                                       and (inverse_permutation[e.target] >= arrow_width or not prune))
            else:
                # BLOCK Criterion
                l2_edges = G.es.select(lambda e:
                                       (abs(inverse_permutation[e.source]//arrow_width - inverse_permutation[e.target]//arrow_width) >= 1)
                                       and (inverse_permutation[e.source] >= arrow_width or not prune)
                                       and (inverse_permutation[e.target] >= arrow_width or not prune))

        assert len(l1_edges) >= 1
        assert len(l1_edges) + len(l2_edges) == G.ecount()

        # We could for example return the tuples as a result instead
        l1_edges_list = [(e.source, e.target) for e in l1_edges]
        l2_edges_list = [(e.source, e.target) for e in l2_edges]

        actual_width = arrow_width

        l1Graph = igraph.Graph(n=G.vcount(), edges=l1_edges_list, directed=G.is_directed()).permute_vertices(list(inverse_permutation))
    else:

        l1Graph = G.permute_vertices(list(inverse_permutation))

        l2_edges = []
        l2_edges_list = []

        actual_width = get_arrow_width(l1Graph, arrow_width)
        print("Level ", len(decomposition), ": desired width: ", arrow_width, " -  Actual width:", actual_width)

    assert l1Graph.vcount() == G.vcount()
    decomposition.append(ArrowGraph(l1Graph, l1_order, actual_width))

    # assert(get_arrow_width(l1Graph, arrow_width) == max(arrow_width, band_width))

    if len(l2_edges):
        l2Graph = igraph.Graph(n=G.vcount(), edges=l2_edges_list, directed=G.is_directed())
        l2Graph.vs["original_id"] = range(0, G.vcount())
        _arrow_decomposition(l2Graph, arrow_width, decomposition, max_level, block_diagonal, prune)


def linearize_with_ck(g: igraph.Graph, order: list[int], base_size = 2)-> None:

    components = g.connected_components(mode='weak')

    for cc in components:
        # For very small components, do not bother to optimize as the bandwidth is bounded by the size of the component
        if len(cc) <= base_size:
            order.extend(g.vs[cc]["original_id"])
            continue

        g_cc = g.subgraph(cc)

        it = g_cc.bfsiter(0, mode='all')
        for v in it:
            order.append(v["original_id"])
            continue


def linearize_with_random_forest(g: igraph.Graph, order: list[int], base_size: int = 4) -> None:
    """
    Pushes the original_id's of the vertices of g onto the order, attempting to minimize the cost of the linear arrangment.
    The cost of the linear arrangement is given by summing the distance of the edge's endoints in the order over all edges.
    :param g: a graph
    :param order: a list of vertex ids
    :param base_size: non-negative integer that controls on which size connected components the linear arrangement switches to
    a naive method. You should set this base_size smaller than the desired bandwidth. It is recommended to use a value
    larger than 1 to improve performance.
    :return:
    """
    weights = numpy.random.rand(g.ecount())
    spanning_forest = g.spanning_tree(weights=weights, return_tree=True)

    assert spanning_forest.vcount() == g.vcount()

    components = spanning_forest.connected_components(mode='weak')

    assert spanning_forest.ecount() == spanning_forest.vcount() - len(components)

    for cc in components:

        # For very small components, do not bother to optimize as the bandwidth is bounded by the size of the component
        if len(cc) <= base_size:
            order.extend(g.vs[cc]["original_id"])
            continue

        g_cc = spanning_forest.subgraph(cc)

        dfs_v, dfs_p = g_cc.dfs(0, mode='all')

        n = len(cc)
        assert len(dfs_v) == n

        tree_directed = igraph.Graph(n=n, edges=zip(dfs_p[1:], dfs_v[1:]), directed=True)
        tree_directed.vs["original_id"] = g_cc.vs["original_id"]

        assert tree_directed.ecount() == tree_directed.vcount( ) -1
        assert tree_directed.vcount() == n

        linearize_tree(tree_directed, order)


def linearize_tree(g: igraph.Graph, order: list[int]) -> None:
    """
    Pushes the original_id's of the vertices of g onto the order, attempting to minimize the cost of the linear arrangment.
    The cost of the linear arrangement is given by summing the distance of the edge's endoints in the order over all edges.
    :param g: A directed, rooted tree
    :param order: a list of vertex ids
    :return: None
    """

    # This is a little mini DP that computes the size of the subtrees
    # We compute the subtree sizes in reverse topological order
    topo = g.topological_sorting()
    g.vs["subtree_size"] = 1
    for v in reversed(topo):
        for u in g.successors(v):
            g.vs[v]["subtree_size"] = g.vs[v]["subtree_size"] + g.vs[u]["subtree_size"]

    assert g.vs[topo[0]]["subtree_size"] == g.vcount()

    _linearize_tree_stack(g, g.vs[topo[0]], order)


def _linearize_tree_stack(g: igraph.Graph, root, order: list[int]) -> None:

    stack = [root]
    while len(stack) > 0:

        current_vertex = stack.pop()
        order.append(current_vertex["original_id"])

        # We want to visit the larger subtree vertices last, hence we push onto the stack into reverse order
        succ = g.vs[g.successors(current_vertex)]
        children = sorted(succ, key=lambda k: k["subtree_size"], reverse=True)
        stack.extend(children)


def _linearize_tree_recursively(g: igraph.Graph, root, order: list[int]) -> None:

    order.append(root["original_id"])
    succ = g.vs[g.successors(root)]
    next_subtree = sorted(succ, key=lambda k: k["subtree_size"])
    for u in next_subtree:
        _linearize_tree_recursively(g, u, order)


def _arrow_linear_order(g: igraph.Graph, arrow_width=512, deterministic=False) -> list[int]:

    costs = zip(g.vs, g.degree())
    costs_sorted = sorted(costs, key=lambda a: a[1], reverse=True)

    # Prune the highest cost vertices
    high_cost_vertices = [v["original_id"] for v, c in costs_sorted[0: arrow_width]]
    # We will compute the linear arrangement on this nontrivial middle degree part
    middle_vertices = [v for v, c in costs_sorted[arrow_width:] if c > 0]
    # Singletons come last
    singletons = [v["original_id"] for v, c in costs_sorted[arrow_width+len(middle_vertices):] if c == 0]

    assert g.vcount() == len(high_cost_vertices) + len(middle_vertices) + len(singletons)

    order = high_cost_vertices
    G_remaining = g.subgraph(middle_vertices)

    if not deterministic:
        linearize_with_random_forest(G_remaining, order, min(arrow_width-1, 16))
    else:
        linearize_with_ck(G_remaining, order)

    assert len(order) == arrow_width + len(middle_vertices)

    order.extend(singletons)

    assert len(order) == g.vcount()

    return order
