# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import spox.opset.ai.onnx.v17 as op17
import spox.opset.ai.onnx.v19 as op19
import spox.opset.ai.onnx.v21 as op21
from spox import Tensor, argument


@pytest.mark.parametrize("op", [op17, op19, op21])
def test_loop_inference(op):
    x, y, zs = op.loop(
        v_initial=[argument(Tensor(float, (None,))), argument(Tensor(int, ("N", 2)))],
        body=lambda i, c, a, b: [op.const(True), a, op.add(i, b), i],
    )
    assert x.type == Tensor(float, (None,))
    assert y.type == Tensor(int, ("N", 2))
    assert zs.type == Tensor(int, (None, 1))


@pytest.mark.parametrize("op", [op17, op19, op21])
def test_loop_concat(op):
    num_iters = op.const(1)
    v = op.const([], dtype=np.int64)

    result = op.loop(
        num_iters,
        v_initial=[v],
        body=lambda i, c, x: (op.const(True), op.concat([x, op.const([1])], axis=0)),
    )[0]

    assert result.type == Tensor(int, (None,))
