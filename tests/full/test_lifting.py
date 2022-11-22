from typing import Callable

import numpy
import pytest

import spox.opset.ai.onnx.v17 as op
from spox._graph import arguments, results
from spox._type_system import Tensor
from spox._var import Var


@pytest.fixture(scope="session")
def optional_lifting_graph(ext):
    def liftA(f: Callable[[Var], Var], x: Var) -> Var:
        return ext.maybe(op.optional_has_element(x), f(op.optional_get_element(x)))

    def liftA2(f: Callable[[Var, Var], Var], x: Var, y: Var) -> Var:
        if x.type != y.type:
            raise TypeError(f"{x.type} != {y.type}")
        return ext.maybe(
            op.and_(op.optional_has_element(x), op.optional_has_element(y)),
            f(op.optional_get_element(x), op.optional_get_element(y)),
        )

    # Model: add the two arguments as float32, only if a is even, otherwise None.
    a, b = arguments(a=Tensor(numpy.int64, ()), b=Tensor(numpy.float32, ()))

    # Value only present if argument a is even
    is_even = op.reshape(
        op.not_(op.cast(op.mod(a, op.const(2)), to=numpy.bool_)), op.const([1])
    )
    a_even = ext.maybe(is_even, a)

    # Cast value to float32 (if it exists)
    a_even_f32 = liftA((lambda x: op.cast(x, to=numpy.float32)), a_even)

    # Add a_even cast to float32 and b (if a_even exists)
    r = liftA2(op.add, a_even_f32, op.optional(b))

    return results(r=r)


@pytest.mark.parametrize("a_value,b_value,r_value", [(8, 1.5, 9.5), (7, 1.5, None)])
def test_optional_lifting(
    onnx_helper, optional_lifting_graph, a_value, b_value, r_value
):
    onnx_helper.assert_close(
        onnx_helper.run(
            optional_lifting_graph,
            "r",
            a=numpy.array(a_value, dtype=numpy.int64),
            b=numpy.array(b_value, dtype=numpy.float32),
        ),
        r_value,
    )
