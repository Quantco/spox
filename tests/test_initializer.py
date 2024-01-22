import itertools
from typing import Any, List

import numpy
import pytest

from spox import Tensor, argument, build, initializer
from spox.opset.ai.onnx import v19 as op

TESTED_INITIALIZER_ROWS: List[List[Any]] = [
    [0, 1, 2],
    [0.0, 1.0, 2.0],
    [numpy.float16(3.14), numpy.float16(5.3)],
    ["abc", "def"],
    [True, False],
]


def assert_expected_initializer(var, value):
    numpy.testing.assert_equal(var._get_value(), numpy.array(value))
    assert var.unwrap_tensor().dtype.type == numpy.array(value).dtype.type
    assert var.unwrap_tensor().shape == numpy.array(value).shape


@pytest.mark.parametrize(
    "value", itertools.chain.from_iterable(TESTED_INITIALIZER_ROWS)
)
def test_initializer_scalar(value):
    assert_expected_initializer(initializer(value), value)


@pytest.mark.parametrize("row", TESTED_INITIALIZER_ROWS)
def test_initializer_iter(row):
    assert_expected_initializer(initializer(row), row)


@pytest.mark.parametrize("row", TESTED_INITIALIZER_ROWS)
def test_initializer_matrix(row):
    assert_expected_initializer(initializer([row, row]), [row, row])


@pytest.mark.parametrize("row", TESTED_INITIALIZER_ROWS)
def test_initializer_naming(row):
    # Get default name for initializer
    init = initializer(row)
    model = build({}, {"init": init})
    init_name = model.graph.initializer[0].name

    # Create some sample model
    arg = argument(Tensor(numpy.array(row).dtype, tuple()))

    # Check if we can create a model with the same name
    # as the initializer
    model1 = build({init_name: arg}, {"ret": op.equal(arg, init)})

    assert len(model1.graph.input) == 1


@pytest.mark.parametrize("row", TESTED_INITIALIZER_ROWS)
def test_initializer_subgraph(row):
    if_ret = op.if_(
        op.const(True),
        then_branch=lambda: [initializer(row)],
        else_branch=lambda: [initializer(row)],
    )[0]

    model = build({}, {"if_ret": if_ret})
    assert len(model.graph.initializer) == 0


@pytest.mark.parametrize("row", TESTED_INITIALIZER_ROWS)
def test_initializer_both_subgraphs(row):
    init = initializer(row)
    if_ret = op.if_(
        op.const(True),
        then_branch=lambda: [init],
        else_branch=lambda: [init],
    )[0]

    model = build({}, {"if_ret": if_ret})

    # The initializer should have been defined in the outer-most scope
    assert len(model.graph.initializer) == 1
