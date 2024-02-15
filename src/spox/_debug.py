import sys
from contextlib import contextmanager

from spox._var import Var

# If `STORE_TRACEBACK` is `True` any node created will store a traceback for its point of creation.
STORE_TRACEBACK = False


@contextmanager
def show_construction_tracebacks(debug_index):
    """
    Context manager constructed with a ``Builder.build_result.debug_index``.

    Useful for debugging ONNX Runtime, which may contain error messages with internal node names.
    This context manager intercepts that error message and prints additional logs to the output with tracebacks
    of where the mentioned nodes where constructed.
    """
    # Avoid circular dependency
    from spox._node import Node

    try:
        yield
    except Exception as e:
        message = str(e)
        by_name = dict(
            sorted(debug_index.items(), key=lambda v: len(v[0]), reverse=True)
        )
        # If starts overlap, the longest name is preferred.
        all_found = [(message.find(name), (name, obj)) for name, obj in by_name.items()]
        all_found = [(i, t) for i, t in all_found if i != -1]
        found = dict(all_found)
        if -1 in found:
            del found[-1]
        for name, obj in reversed(found.values()):
            if isinstance(obj, Var):
                if not obj:
                    continue
                node = obj._op
            else:
                assert isinstance(obj, Node)
                node = obj

            if node._traceback is None:
                raise ValueError(
                    "Set `spox._debug.STORE_TRACEBACK=True` before creating the ONNX model."
                )

            print(f">>> from mentioned {name}\n", file=sys.stderr)
            print("".join(node._traceback[-6:-3]), file=sys.stderr)
            print(f"<<< where {name} was constructed\n", file=sys.stderr)
        raise e
    finally:
        pass
