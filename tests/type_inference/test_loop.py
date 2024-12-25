# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import spox.opset.ai.onnx.v17 as op17
import spox.opset.ai.onnx.v19 as op19
import spox.opset.ai.onnx.v21 as op21
from spox import Optional, Sequence, Tensor, argument


@pytest.mark.parametrize("op", [op17, op19, op21])
def test_loop_inference(op):
    x, y, zs = op.loop(
        v_initial=[
            argument(Tensor(np.float64, (None,))),
            argument(Tensor(np.int64, ("N", 2))),
        ],
        body=lambda i, c, a, b: [op.const(True), a, op.add(i, b), i],
    )
    assert x.type == Tensor(np.float64, (None,))
    assert y.type == Tensor(np.int64, ("N", 2))
    assert zs.type == Tensor(np.int64, (None, 1))


@pytest.mark.parametrize("op", [op17, op19, op21])
def test_loop_concat(op):
    num_iters = op.const(1)
    v = op.const([], dtype=np.int64)

    result = op.loop(
        num_iters,
        v_initial=[v],
        body=lambda i, c, x: (op.const(True), op.concat([x, op.const([1])], axis=0)),
    )[0]

    # type can change, so we cannot infer anything
    assert result.type == Tensor(np.int64, None)


@pytest.mark.parametrize("op", [op17, op19, op21])
def test_loop_sequence(op):
    num_iters = op.const(1)
    v = op.sequence_empty(dtype=np.int64)

    result = op.loop(
        num_iters,
        v_initial=[v],
        body=lambda i, c, x: (op.const(True), op.sequence_insert(x, op.const([1]))),
    )[0]

    assert result.type == Sequence(Tensor(np.int64, None))


@pytest.mark.parametrize("op", [op17, op19, op21])
def test_loop_optional(op):
    num_iters = op.const(1)
    v = op.optional(type=Tensor(np.int64, (1, 2)))

    result = op.loop(
        num_iters,
        v_initial=[v],
        body=lambda i, c, x: (
            op.const(True),
            op.if_(
                op.optional_has_element(x),
                then_branch=lambda: [op.optional(type=Tensor(np.int64, (1, 2)))],
                else_branch=lambda: [op.optional(op.const([[1, 1]]))],
            )[0],
        ),
    )[0]

    assert result.type == Optional(Tensor(np.int64, (1, 2)))


@pytest.mark.parametrize("op", [op17, op19, op21])
def test_loop_optional_no_shape(op):
    num_iters = op.const(1)
    v = op.optional(type=Tensor(np.int64, (1, 2)))

    result = op.loop(
        num_iters,
        v_initial=[v],
        body=lambda i, c, x: (
            op.const(True),
            op.if_(
                op.optional_has_element(x),
                then_branch=lambda: [op.optional(type=Tensor(np.int64, (1, 2)))],
                else_branch=lambda: [op.optional(op.const([[1]]))],
            )[0],
        ),
    )[0]

    # shape can change, we cannot infer type
    assert result.type == Optional(Tensor(np.int64, None))
