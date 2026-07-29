# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import spox.opset.ai.onnx.v17 as op
import spox.opset.ai.onnx.v18 as op18
from spox import argument
from spox._exceptions import InferenceError
from spox._graph import arguments, results
from spox._type_system import Tensor


def test_explicit_unspecified_optional(onnx_helper):
    (x,) = arguments(x=Tensor(np.float32, (None,)))
    r = op.clip(x, min=None, max=op.constant(value_float=0.0))
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "r", x=np.array([-1, 1, 2], np.float32)),
        [-1, 0, 0],
    )


def test_unspecified_optional(onnx_helper):
    (x,) = arguments(x=Tensor(np.float32, (None,)))
    r = op.clip(x, max=op.constant(value_float=1.0))
    r = op.clip(r, min=op.constant(value_float=-1.0))
    graph = results(r=r)
    onnx_helper.assert_close(
        onnx_helper.run(graph, "r", x=np.array([-3, -1, 1, 2], np.float32)),
        [-1, -1, 1, 1],
    )


def test_variadic_no_input_list_mutation(onnx_helper):
    a, b = op.const([1]), op.const([2])
    ins = [a, b]
    concat = op.concat(ins, axis=0)
    ins[1] = b
    assert list(concat._op.inputs.get_var_infos().values()) == [
        a._var_info,
        b._var_info,
    ]


def test_variadic_no_attr_mutation_array(onnx_helper):
    a = np.array([1])
    x = op.constant(value=a)
    a[0] = 0
    assert isinstance(x._op, op._Constant)
    assert x._op.attrs.value is not None
    assert list(x._op.attrs.value.value) == [1]


def test_variadic_no_attr_mutation_list(onnx_helper):
    a = [1]
    x = op.constant(value_ints=a)
    a[0] = 0
    assert isinstance(x._op, op._Constant)
    assert x._op.attrs.value_ints is not None
    assert list(x._op.attrs.value_ints.value) == [1]


def test_deprecated_schemas_removed():
    import spox.opset.ai.onnx.v17 as op17

    assert not hasattr(op17, "scatter")
    assert not hasattr(op17, "upsample")


@pytest.mark.parametrize("kwargs", [{"num_outputs": 2}, {"split": op18.const([2, 3])}])
def test_split18_arguments(kwargs):
    a = argument(Tensor(np.float32, (None,)))
    b, c = op18.split(a, **kwargs)

    assert len(b.unwrap_tensor().shape) == 1  # type: ignore
    assert len(c.unwrap_tensor().shape) == 1  # type: ignore


@pytest.mark.parametrize("bad_shape", [(None,), (), (2, 2)])
def test_split_raises_for_split_input_with_bad_shape(bad_shape):
    a = argument(Tensor(np.float32, (None,)))
    b = argument(Tensor(np.int64, bad_shape))
    with pytest.raises(InferenceError):
        op18.split(a, b)


def test_split_is_usable_with_reshaped_split_input():
    a = argument(Tensor(np.float32, (None,)))
    b = argument(Tensor(np.int64, None))  # 'b' has undefined rank
    op18.split(a, op.reshape(b, op.const([4])))
