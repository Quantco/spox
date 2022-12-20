import numpy
import onnx
import onnx.reference.op_run
import onnx.reference.ops._op_list
import onnx.reference.ops.op_cast
from onnx.reference.op_run import OpRun


class PatchedOptionalHasElement(OpRun):
    def _run(self, x):
        return (numpy.array(not ((isinstance(x, list) and x == [None]) or x is None)),)


class PatchedCast(OpRun):
    def _run(self, x, to=None):  # type: ignore
        if to == onnx.TensorProto.STRING:
            return (x.astype(numpy.str_),)
        return (onnx.reference.ops.op_cast.cast_to(x, to),)


def patch_reference_implementations():
    """Patch known broken reference implementations in ONNX.

    As the reference implementation module in ONNX is quite new, it still has bugs which are a nuisance in Spox.
    This function modifies their implementation by catching out the special cases known to be faulty.
    """
    onnx.reference.ops._op_list.OptionalHasElement = PatchedOptionalHasElement
    onnx.reference.ops._op_list.Cast = PatchedCast
