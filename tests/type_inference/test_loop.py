import spox.opset.ai.onnx.v17 as op
from spox import Tensor, argument


def test_loop_inference():
    x, y, zs = op.loop(
        v_initial=[argument(Tensor(float, (None,))), argument(Tensor(int, ("N", 2)))],
        body=lambda i, c, a, b: [op.const(True), a, op.add(i, b), i],
    )
    assert x.type == Tensor(float, (None,))
    assert y.type == Tensor(int, ("N", 2))
    assert zs.type == Tensor(int, (None, 1))
