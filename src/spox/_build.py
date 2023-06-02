import itertools
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import numpy
import onnx

from . import _function
from ._exceptions import BuildError
from ._internal_op import Argument, intros
from ._node import Node
from ._scope import Scope
from ._traverse import iterative_dfs
from ._var import Var

if TYPE_CHECKING:
    from ._graph import Graph

T = TypeVar("T")


class Cached(Generic[T]):
    """A generic cached-value type, for which the ``.value`` property raises if it was not previously set."""

    _value: Optional[T]

    def __init__(self, value: Optional[T] = None):
        self._value = value

    @property
    def value(self) -> T:
        if self._value is None:
            raise ValueError("Cannot access missing value")
        return self._value

    @value.setter
    def value(self, to: T):
        self._value = to


@dataclass(frozen=True)
class BuildResult:
    """
    Object containing all the results of the build process.

    Contains information on the graph structure, naming, operators, etc.
    """

    scope: Scope
    nodes: Dict[Node, Tuple[onnx.NodeProto, ...]]
    arguments: Tuple[Var, ...]
    results: Tuple[Var, ...]
    opset_req: Set[Tuple[str, int]]
    functions: Tuple["_function.Function", ...]
    initializers: Dict[Var, numpy.ndarray]


class Builder:
    """
    Class representing an object performing the build process and handling the necessary intermediate state.

    The normal call performing the process is ``Builder(main).build() -> BuildResult``.

    There are three main stages in the build process.

    - **Discovery**
        - We resolve the arguments and results of all graphs. For some graphs (usually only the main) the arguments
          may be unknown and should be resolved as the arguments used anywhere in the Node graph, but not claimed by
          any of the other subgraphs.
        - Results are wrapped in Introduce so that later in the code there is only one "subgraph source" to traverse.
    - **Resolving scopes**
        - Solve the least enclosing scope problem (find the scope tree and where each node is in it).
        - Find topologically sorted lists of nodes to introduce in each of the scopes (subgraphs).
    - **Compilation**
        - Going through all graphs, build required nodes (as returned by the previous step) with Node.to_onnx routines.
        - There is some trickery with subgraphs, as we avoid restarting a fully privileged build process from them.
          Instead, we use known information and inject a custom build result to build the subgraph with, and then place
          it into the node's attribute.
        - This is also the stage where naming is happening, via the Scope class.
    """

    class ScopeTree:
        """
        Structure representing the tree of scopes, which are identified with the respective graphs.

        This structure is the base of the least-enclosing-scope algorithm. Every value (Var), and hence
        the responsible Node - up to its (Python object) identity may appear in multiple scopes, but it should
        best-cased be computed only once in the ONNX graph, same as in the Python source code.

        The goal of the least-enclosing-scope strategy is to find the most concrete/innermost scope for every value,
        such that every scope that has to access it is able to.

        A more naive alternative is for example first-use, where a value is recomputed in the ONNX graph every time
        it is accessed and not available in the outer scope.

        We postulate that the scope structure is a tree, so that every scope has a direct parent. In ONNX
        a scope may access all of its ancestors in the scope tree.

        We have two major constraints for the scoping algorithm:
        - u is an input of v, denoted u->v. Then the scope of u must be an ancestor of the scope of v in the tree.
        - g is a subgraph under u, denoted g=>u. Then the scope owned by g must be a child of the scope of u.

        We represent the scope tree with two dictionaries:
        - ``subgraph_of`` for a given Graph is the Node that it is the subgraph of.
        - ``scope_of`` for a given Node is the Graph to the scope of which it belongs (we identify scopes with Graphs).

        By the second rule, the parent scope of a given scope is ``scope_of[subgraph_of[graph]]``.

        We may find that the first constraint is sometimes not satisfied. Then we set the scope of a node to the
        parent scope of its current scope, as many times as possible. This turns out to be computable with LCA
        (lowest common ancestor), which is a common operation on trees.
        """

        subgraph_owner: Dict["Graph", Node]
        scope_of: Dict[Node, "Graph"]

        def __init__(self):
            self.subgraph_owner = {}
            self.scope_of = {}

        def parent(self, graph: "Graph") -> "Graph":
            """
            Return the parent of a scope in the represented scope tree.

            A scope must be the child of the scope of the node that holds this scope (subgraph)
            If there is no parent (the main graph is not a subgraph of anything), the scope itself is returned.
            """
            return (
                self.scope_of[self.subgraph_owner[graph]]
                if graph in self.subgraph_owner
                else graph
            )

        def lca(self, a: "Graph", b: "Graph") -> "Graph":
            """
            A simple LCA algorithm without preprocessing that only accesses the parents.

            The algorithm is simple - we keep going up one step alternating between the nodes.
            Whenever we hit a node that was already visited, it must be the lowest common ancestor.
            - as it is definitely a common ancestor, and it required the least steps up.

            Time and space complexity in the length of the path between a, b in the tree.
            """
            vis_a, vis_b = {a}, {b}
            while a not in vis_b:
                vis_a.add(a)
                a = self.parent(a)
                a, b = b, a
                vis_a, vis_b = vis_b, vis_a
            return a

    # Graphs needed in the build
    main: "Graph"
    graphs: Set["Graph"]
    graph_topo: List["Graph"]
    # Arguments, results
    arguments_of: Dict["Graph", List[Var]]
    results_of: Dict["Graph", List[Var]]
    source_of: Dict["Graph", Node]
    # Arguments found by traversal
    all_arguments_in: Dict["Graph", Set[Var]]
    claimed_arguments_in: Dict["Graph", Set[Var]]
    # Scopes
    scope_tree: ScopeTree
    scope_own: Dict["Graph", List[Node]]

    def __init__(self, main: "Graph"):
        self.main = main
        self.graphs = set()
        self.graph_topo = list()
        self.arguments_of = {}
        self.results_of = {}
        self.source_of = {}
        self.all_arguments_in = {}
        self.claimed_arguments_in = {}
        self.scope_tree = self.ScopeTree()
        self.scope_own = {}

    def build_main(self) -> BuildResult:
        # Discovery
        self.discover(self.main)
        self.graph_topo.reverse()
        if not self.graphs == set(self.arguments_of) == set(self.results_of):
            raise BuildError("Some graphs have missing build data.")

        # Resolving scopes
        for graph in self.graph_topo:
            self.update_scope_tree(graph)
        self.resolve_scopes()

        # Compilation
        return self.compile_graph(self.main, Scope())

    @staticmethod
    def get_intro_results(
        request_results: Dict[str, Var], set_names: bool
    ) -> List[Var]:
        """
        Helper method for wrapping all requested results into a single Introduce and possibly naming them.

        By default, only the main graph's results are named (and subgraphs get somewhat autogenerated names),
        as usually only ONNX subgraph input/output ordering is significant.
        """
        # Created vars all have the same op
        vars = list(intros(*request_results.values()))
        for key, var in zip(request_results, vars):
            if set_names:
                var._rename(key)
        return vars

    def discover(self, graph: "Graph") -> Tuple[Set[Var], Set[Var]]:
        """
        Run the discovery step of the build process. Resolves arguments and results for the involved graphs.
        Finds the topological ordering between (sub)graphs and sets their owners (nodes of which they are attributes).

        The time complexity of this step is in the order of the sum over the counts of intermediate nodes
        for all subgraphs (scopes). With number of scopes `s` and number of nodes `n`, worst-case is `O(ns) = O(n^2)`.

        The bottleneck is in rediscovering the same argument nodes many times. This could be avoided if used arguments
        (the basis) were stored in the Node itself and computed during construction time.

        Returns
        -------
        all_arguments
            All arguments. Found in this graph by traversal including subgraphs.
        claimed_arguments
            All arguments already claimed by a graph. Found by traversal including subgraphs.
        """
        if graph in self.graphs:
            return self.all_arguments_in[graph], self.claimed_arguments_in[graph]

        self.graphs.add(graph)

        # Create and set the source & results of this graph
        if not graph.requested_results:
            raise BuildError(f"Graph {graph} has no results.")
        self.results_of[graph] = self.get_intro_results(
            graph.requested_results, graph is self.main
        )
        self.source_of[graph] = self.results_of[graph][0]._op

        # Resolving arguments is a bit more complicated and requires a traversal to find all & claimed arguments.
        # To avoid too many dictionary accesses, we create aliases for the relevant sets.
        all_arguments = self.all_arguments_in[graph] = set()
        claimed_arguments = self.claimed_arguments_in[graph] = set()
        used_arguments = set()

        def collect_arguments(nd: Node):
            nonlocal all_arguments, claimed_arguments, used_arguments
            if isinstance(nd, Argument):
                all_arguments.add(nd.outputs.arg)
                used_arguments.add(nd.outputs.arg)
            for subgraph in nd.subgraphs:
                all_arguments_sub, claimed_arguments_sub = self.discover(subgraph)
                all_arguments |= all_arguments_sub
                claimed_arguments |= claimed_arguments_sub
                if subgraph not in self.scope_tree.subgraph_owner:
                    self.scope_tree.subgraph_owner[subgraph] = nd
                if self.scope_tree.subgraph_owner[subgraph] != nd:
                    raise BuildError(
                        "Subgraph has multiple owners (the Graph instance was reused)."
                    )

        # Here, we compute:
        #  - all_arguments to be the set of all Argument instances found anywhere, including subgraphs
        #  - claimed_arguments to be the arguments found anywhere that are already set as subgraph arguments.
        #    This is modified afterwards to include arguments assigned to this graph.
        #  - used_arguments to be arguments that are used directly in this graph, excluding subgraphs.
        iterative_dfs(
            [self.source_of[graph]],
            lambda nd: (a._op for a in nd.dependencies),
            collect_arguments,
        )
        self.graph_topo.append(graph)

        # Now we resolve which arguments we should get.
        if graph.requested_arguments is None:
            # If there's no request, we take all arguments found anywhere in this graph
            self.arguments_of[graph] = list(all_arguments - claimed_arguments)
        else:
            # If there is a request, we may not have found it by traversal if an argument was unused.
            all_arguments |= set(graph.requested_arguments)
            self.arguments_of[graph] = list(graph.requested_arguments)

        if set(self.arguments_of[graph]) & claimed_arguments:
            raise BuildError(
                "Some arguments that this graph claims were already claimed. "
                "Did subgraphs share arguments they requested?"
            )
        # If a claimed argument is used directly in the current graph,
        # a subgraph-local Argument was leaked, which breaks the contract.
        leaked = claimed_arguments & used_arguments
        if leaked:
            raise BuildError(
                "Some subgraph-local arguments were leaked to an outer scope. "
                "Hint: avoid side effects in your subgraph callbacks."
            )
        claimed_arguments |= set(self.arguments_of[graph])

        return all_arguments, claimed_arguments

    def update_scope_tree(self, graph: "Graph") -> None:
        """
        Traverse ``graph`` and update the Builder's scope tree to accommodate the input constraints inside it.

        The algorithm is based on a relaxation of the scopes when a constraint requires it.
        - When a new node is found, by default it is assigned to the scope of the graph it is found in.
        - If a node was already in the graph and had a scope assigned, that scope has to be accessible to the
          graph that we are in (as this is an indirect result to this graph).
          In this case we update the scope to the LCA of this graph's scope and the existing scope.
        - This cannot break the input-scope (1st) constraint (as we only expose more values to scopes),
          and the 2nd constraint is not affected.

        Pessimistically an LCA constraint update may be O(s), and all n nodes may be visited in all s scopes.
        However, a node may be pushed up in the scope tree at most O(s) times, so the complexity is amortised to O(ns).

        It is expected that subgraphs reachable from the source of ``graph`` have already been resolved.
        This method is called for all graphs in topological ordering, which ensures the scope tree
        is completed 'bottom-up'.
        """

        def satisfy_constraints(node):
            # By default, a node is bound to the scope it is found in.
            self.scope_tree.scope_of.setdefault(node, graph)
            # Bring up the scope of its node to its ancestors if it is too low to be accessible in the current graph.
            self.scope_tree.scope_of[node] = self.scope_tree.lca(
                graph, self.scope_tree.scope_of[node]
            )

        iterative_dfs(
            [self.source_of[graph]],
            lambda nd: (a._op for a in nd.dependencies),
            satisfy_constraints,
        )

    def resolve_scopes(self) -> None:
        """
        Using the updated scope tree satisfying all of our constraints, set ``scope_own`` - the topologically sorted
        list of nodes that should be included in a given graph (due to usage or requirement of child scopes).

        This step is O(n log n) - the topological sorting is found in linear time, and afterwards we take the "subsets"
        of the sorting to bind to the actual scopes. The subsets are based on the scope tree relations.

        We use the reverse of the postorder of the implicit node graph (see ``iterative_dfs`` docstring)
        - this is slightly higher quality than a normal topological sorting which attempts to be "parallel",
        while a DFS' postorder is more "localised".
        """
        graph_scope_set: Dict[Any, Set[Node]] = {ctx: set() for ctx in self.graphs}
        for node, owner in self.scope_tree.scope_of.items():
            graph_scope_set[owner].add(node)

        # Here we follow both input and subgraph edges, since they both have an impact on which order values
        # should be defined in.
        topo = iterative_dfs(
            [self.source_of[self.main]],
            lambda nd: itertools.chain(
                (arr._op for arr in nd.dependencies),
                (self.source_of[sub] for sub in nd.subgraphs),
            ),
        )

        set_topo = set(topo)
        if len(topo) != len(set_topo):
            raise BuildError(
                "Improper topological sorting due to returned repeated nodes."
            )

        # We sort the nodes owned by a scope by the index in the topological order
        topo_index = {node: i for i, node in enumerate(topo)}
        for graph in self.graphs:
            if any(nd not in set_topo for nd in graph_scope_set[graph]):
                raise BuildError(
                    "Missing node in topological order that was expected in scope."
                )
            self.scope_own[graph] = sorted(
                graph_scope_set[graph], key=lambda nd: topo_index[nd]
            )

    def get_build_subgraph_callback(
        self, scope: Scope
    ) -> Tuple[Callable, Set[Tuple[str, int]]]:
        """Create a callback for building subgraphs for ``Node.to_onnx``."""

        subgraph_opset_req = set()  # Keeps track of all opset imports in subgraphs

        def build_subgraph(
            subgraph_of: Node, key: str, subgraph: "Graph"
        ) -> onnx.GraphProto:
            nonlocal subgraph_opset_req
            subgraph_name = scope.node[subgraph_of] + f"_{key}"
            subgraph = subgraph.with_name(subgraph_name)._inject_build_result(
                self.compile_graph(subgraph, scope, subgraph_name + "__")
            )
            subgraph_opset_req |= subgraph._get_build_result().opset_req
            return subgraph.to_onnx()

        return build_subgraph, subgraph_opset_req

    def compile_graph(
        self, graph: "Graph", scope: Scope, prefix: str = ""
    ) -> BuildResult:
        """
        Compile a given Graph into a BuildResult. Handles naming of all the Vars/Nodes and only adds Nodes to a
        Graph that should be present in the respective GraphProto. The passed Scope object is aware of values already
        available in the outer scope and may be the source of errors if the build fails.

        Parameters
        ----------
        graph
            Graph to compile a BuildResult for.
        scope
            Scope object for the current graph. May have (visible) parent scopes.
        prefix

        Returns
        -------
        ~
            See the definition for the exact contents of the BuildResult dataclass. Used to build GraphProto/ModelProto
            from a Spox Graph.
        """
        nodes: Dict[Node, Tuple[onnx.NodeProto, ...]] = {}
        # A bunch of model metadata we're collecting
        opset_req: Set[Tuple[str, int]] = set()
        functions: List[_function.Function] = []
        initializers: Dict[Var, numpy.ndarray] = {}

        # Add arguments to our scope
        for arg in self.arguments_of[graph]:
            node = arg._op
            node.update_metadata(opset_req, initializers, functions)
            scope.update(
                node, prefix
            )  # Throws a ScopeError if we attempt to redeclare an argument

        # Build all nodes for this Graph. Also builds subgraph with a recursive call to compile_graph
        build_subgraph, subgraph_opset_req = self.get_build_subgraph_callback(scope)
        for node in self.scope_own[graph]:
            if isinstance(node, Argument):
                continue
            node.update_metadata(opset_req, initializers, functions)
            scope.update(
                node, prefix
            )  # Throws a ScopeError if we attempt to redeclare a node
            # to_onnx throws ScopeErrors if it uses nodes that were not found to be in this scope (or outer)
            nodes[node] = tuple(node.to_onnx(scope, build_subgraph=build_subgraph))
        opset_req |= subgraph_opset_req

        # Return results typed to what we want.
        # We use tuples here to avoid modifying this stuff by mistake down the line.
        return BuildResult(
            scope=scope,
            nodes=nodes,
            arguments=tuple(self.arguments_of[graph]),
            results=tuple(self.results_of[graph]),
            opset_req=opset_req,
            functions=tuple(functions),
            initializers=initializers,
        )
