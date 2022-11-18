from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple, TypeVar

V = TypeVar("V")


def iterative_dfs(
    sources: Iterable[V],
    adj: Callable[[V], Iterable[V]],
    post_callback: Optional[Callable[[V], None]] = None,
    raise_on_cycle: bool = True,
) -> List[V]:
    """
    Performs a depth-first search and returns the postorder of the traversal.
    Throws if the graph contains a cycle. The topological sorting returned is the postorder of the DFS.

    This is a non-recursive implementation. Postorder of a DFS is slightly better than a conventional topological
    sorting for the use-case, as it is more "localised" (rather than "parallel") - values that were "used" at the same
    time end up close to each other in the ordering.

    Parameters
    ----------
    sources
        The sources to start the DFS from.
    adj
        Callback function the returns an iterable adjacency list from a given node (edges leading from the node).
    post_callback
        Callback function called when leaving a node (when it is pushed to the postorder).
    raise_on_cycle
        Whether to raise an error whenever a cycle is found.
    Returns
    -------
    postorder
        The list of nodes found  in postorder.
        Note that the reverse postorder of a DAG traversal is its topological sorting - hence the postorder
        of the transpose of that DAG is also its topological sorting.
    Notes
    -----
    The behaviour of this implementation should be equivalent to this:

        vis = set()
        postorder = []

        def dfs(u: V):
            if u in vis:
                return
            vis.add(u)
            for v in adj(u):
                dfs(v)
            postorder.append(u)
            callback(u)

        for s in sources:
            dfs(s)
    """
    postorder: List[V] = []
    visited: Set[V] = set()
    stack: Set[V] = set()

    # Recursion stack - the state of a DFS is described with a stack of (vertex, nodes left to visit).
    recursion: List[Tuple[V, Iterator[V]]] = []

    def call(w: V):
        """Helper called when we attempt to enter a node ``w``."""
        if w in stack and raise_on_cycle:
            raise RuntimeError(
                "The graph contains a cycle. Was the structure tampered with?"
            )
        if w in visited:
            return
        visited.add(w)
        stack.add(w)
        recursion.append((w, iter(adj(w))))

    for s in sources:
        # Go into the source node and enter the recursion
        call(s)
        while recursion:  # While not done
            # Find the current node in the executed frame and its adjacency list iterator
            u, it = recursion[-1]
            try:
                v = next(it)  # Access the next node to visit
            except StopIteration:  # The next node does not exist
                # This means we are leaving the node now. Wrap up the frame here and run post-update.
                if post_callback is not None:
                    post_callback(u)
                postorder.append(u)
                stack.remove(u)
                recursion.pop()
            else:
                call(v)  # Enter the node since it does exist

    return postorder
