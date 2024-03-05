import functools
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy
import onnx
import onnx.parser
import onnx.shape_inference
import onnxruntime
import pytest

import spox.opset.ai.onnx.v17 as op
from spox._attributes import Attr, AttrFloat32, _Ref
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._function import Function, to_function
from spox._graph import arguments, results
from spox._node import OpType
from spox._type_system import Tensor
from spox._var import Var


@pytest.fixture
def linear():
    class LinearFunction(Function):
        @dataclass
        class Attributes(BaseAttributes):
            slope_outer: AttrFloat32
            shift_outer: AttrFloat32

        @dataclass
        class Inputs(BaseInputs):
            X: Var

        @dataclass
        class Outputs(BaseOutputs):
            Y: Var

        op_type = OpType("LinearFunction", "spox.test", 0)

        attrs: Attributes
        inputs: Inputs
        outputs: Outputs

        def constructor(self, attrs: Dict[str, Attr], inputs: Inputs) -> Outputs:
            # FIXME: At some point, attribute references should be properly type-hinted.
            a = op.constant(
                value_float=_Ref(
                    attrs["slope_outer"], outer_name="slope_outer", name="value_float"
                )  # type: ignore
            )
            b = op.constant(
                value_float=_Ref(
                    attrs["shift_outer"], outer_name="shift_outer", name="value_float"
                )  # type: ignore
            )
            x = inputs.X
            return self.Outputs(op.add(op.mul(a, x), b))

    def linear_inner(
        x: Var, a: Union[float, _Ref[float]], b: Union[float, _Ref[float]]
    ) -> Var:
        return LinearFunction(
            LinearFunction.Attributes(
                slope_outer=AttrFloat32(a, "slope_outer"),
                shift_outer=AttrFloat32(b, "shift_outer"),
            ),
            LinearFunction.Inputs(x),
        ).outputs.Y

    return linear_inner


@pytest.fixture
def linear2(linear):
    class LinearFunction2(Function):
        @dataclass
        class Attributes(BaseAttributes):
            slope1: AttrFloat32
            shift1: AttrFloat32

        @dataclass
        class Inputs(BaseInputs):
            X: Var

        @dataclass
        class Outputs(BaseOutputs):
            Y: Var

        op_type = OpType("LinearFunction2", "spox.test", 0)

        attrs: Attributes
        inputs: Inputs
        outputs: Outputs

        def constructor(self, attrs: Dict[str, Attr], inputs: Inputs) -> Outputs:
            return self.Outputs(
                linear(
                    inputs.X,
                    _Ref(attrs["slope1"], outer_name="slope1", name="slope_outer"),
                    _Ref(attrs["shift1"], outer_name="shift1", name="shift_outer"),
                )
            )

    def linear_inner(
        x: Var, a: Union[float, _Ref[float]], b: Union[float, _Ref[float]]
    ) -> Var:
        return LinearFunction2(
            LinearFunction2.Attributes(
                slope1=AttrFloat32(a, name="slope1"), shift1=AttrFloat32(b, "shift1")
            ),
            LinearFunction2.Inputs(x),
        ).outputs.Y

    return linear_inner


@pytest.fixture
def cubic(linear):
    class CubicFunction(Function):
        @dataclass
        class Attributes(BaseAttributes):
            a3: AttrFloat32
            a2: AttrFloat32
            a1: AttrFloat32
            a0: AttrFloat32

        @dataclass
        class Inputs(BaseInputs):
            X: Var

        @dataclass
        class Outputs(BaseOutputs):
            Y: Var

        op_type = OpType("CubicFunction", "spox.test.extra", 0)

        attrs: Attributes
        inputs: Inputs
        outputs: Outputs

        def constructor(self, attrs: Dict[str, Attr], inputs: Inputs) -> Outputs:
            x = inputs.X
            a = op.mul(
                linear(
                    x,
                    _Ref(attrs["a3"], outer_name="a3", name="slope_outer"),
                    _Ref(attrs["a2"], outer_name="a2", name="shift_outer"),
                ),
                op.mul(x, x),
            )
            b = op.add(
                op.mul(
                    x,
                    op.constant(
                        value_float=_Ref(
                            attrs["a1"], outer_name="a1", name="value_float"
                        )  # type: ignore
                    ),
                ),
                op.constant(
                    value_float=_Ref(attrs["a0"], outer_name="a0", name="value_float"),  # type: ignore
                ),
            )
            y = op.add(a, b)
            return self.Outputs(y)

    def cubic_inner(x: Var, a3: float, a2: float, a1: float, a0: float) -> Var:
        return CubicFunction(
            CubicFunction.Attributes(
                a3=AttrFloat32(a3, name="a3"),
                a2=AttrFloat32(a2, name="a2"),
                a1=AttrFloat32(a1, name="a1"),
                a0=AttrFloat32(a0, name="a0"),
            ),
            CubicFunction.Inputs(X=x),
        ).outputs.Y

    return cubic_inner


@pytest.fixture
def min_fun_graph(linear):
    (start,) = arguments(start=Tensor(numpy.float32, (None,)))
    return results(final=linear(start, 3, 2))


@pytest.fixture
def big_fun_graph(linear):
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32, (None,))
    )
    return results(final=op.div(linear(op.add(first, second), 3, 2), second))


@pytest.fixture
def double_fun_graph(linear):
    (start,) = arguments(start=Tensor(numpy.float32, (None,)))
    return results(final=linear(linear(start, 3, 2), 5, 3))


@pytest.fixture
def wrapped_linear_graph(linear2):
    (start,) = arguments(start=Tensor(numpy.float32, (None,)))
    return results(final=linear2(start, 3, 2))


@pytest.fixture
def cubic_graph(cubic):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    return results(y=cubic(x, 5, 3, 2, 1))


@pytest.fixture
def cubic_rational_graph(cubic):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    return results(y=op.div(cubic(x, 5, 3, 2, 1), cubic(x, 3, 4, 2, 3)))


@pytest.fixture
def cubic_rational_graph_2x3(linear, cubic):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    return results(y=linear(op.div(cubic(x, 5, 3, 2, 1), cubic(x, 3, 4, 2, 3)), 2, 3))


@pytest.fixture
def isnan_graph():
    x, y, z = arguments(
        x=Tensor(numpy.float64, ()),
        y=Tensor(numpy.float64, ()),
        z=Tensor(numpy.float64, ()),
    )

    @to_function("IsNaN", "spox.test")
    def isnan(v: Var) -> List[Var]:
        return [op.not_(op.equal(v, v))]

    return results(
        count=functools.reduce(
            op.add, (op.cast(*isnan(w), to=numpy.int64) for w in (x, y, z))
        )
    )


def test_minimal(onnx_helper, min_fun_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(min_fun_graph, "final", start=a), 3 * a + 2
    )


def test_bigger(onnx_helper, big_fun_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    b = numpy.random.rand(8).astype(numpy.float32) + 1
    onnx_helper.assert_close(
        onnx_helper.run(big_fun_graph, "final", first=a, second=b),
        (3 * (a + b) + 2) / b,
    )


def test_double_application(onnx_helper, double_fun_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(double_fun_graph, "final", start=a), 5 * (3 * a + 2) + 3
    )


def test_onnxruntime_simple_function_support(onnx_helper):
    # See https://github.com/microsoft/onnxruntime/issues/10250 (fixed)
    import onnx.parser

    m = onnx.parser.parse_model(
        """
    <
      ir_version: 8,
      opset_import: [ "" : 14, "local" : 1],
      producer_name: "test",
      producer_version: "1.0",
      model_version: 1,
      doc_string: "Test preprocessing model"
    >
    agraph (uint8[H, W, C] x) => (uint8[H, W, C] x_processed)
    {
        x_processed = local.func(x)
    }

    <
      opset_import: [ "" : 14 ],
      domain: "local",
      doc_string: "function 1"
    >
    f1 (x) => (y) {
        y = Identity(x)
    }

    <
      opset_import: [ "" : 14 ],
      domain: "local",
      doc_string: "function 2"
    >
    f2 (x) => (y) {
        y = Identity(x)
    }

    <
      opset_import: [ "" : 14, "local" : 1 ],
      domain: "local",
      doc_string: "Preprocessing function."
    >
    func (x) => (y) {
        x1 = local.f1(x)
        y = local.f2(x1)
    }

    """
    )

    onnx.checker.check_model(m)
    onnxruntime.InferenceSession(m.SerializeToString())


@pytest.mark.xfail
def test_onnxruntime_nested_function_attr_support():
    m = onnx.parser.parse_model(
        """
    <
      ir_version: 8,
      opset_import: [ "" : 14, "local" : 1],
      model_version: 1
    >
    agraph (int64[H] x) => (int64[H] y)
    {
        y = local.func <outer = 1> (x)
    }

    <
      opset_import: [ "" : 14 ],
      domain: "local",
      doc_string: "function 1"
    >
    f1 <inner> (x) => (y) {
        c = Constant<value_int: int = @inner>()
        y = Add(x, c)
    }

    <
      opset_import: [ "" : 14, "local" : 1 ],
      domain: "local",
      doc_string: "Preprocessing function."
    >
    func <outer> (x) => (y) {
        y = local.f1 <inner: int = @outer> (x)
    }
    """
    )

    onnx.checker.check_model(m)
    session = onnxruntime.InferenceSession(m.SerializeToString())

    x = numpy.array([0, 0, 0], dtype=numpy.int64)
    y = numpy.array([1, 1, 1], dtype=numpy.int64)
    assert (session.run(None, {"x": x})[0] == y).all()


def test_minimal_wrapped(onnx_helper, wrapped_linear_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(wrapped_linear_graph, "final", start=a), 3 * a + 2
    )


def test_simple_nested_calls_session(onnx_helper, cubic_graph):
    model = cubic_graph.to_onnx_model()
    onnxruntime.InferenceSession(model.SerializeToString())


def test_simple_nested_calls(onnx_helper, cubic_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    # increase rtol due to small *non-deterministic* discrepancies
    onnx_helper.assert_close(
        onnx_helper.run(cubic_graph, "y", x=a), (1 + 2 * a) + a * a * (3 + a * 5)
    )


@pytest.mark.skip(
    "ONNX Runtime generates colliding internal identifiers for nested function nodes."
)
def test_nested_calls(onnx_helper, cubic_rational_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(cubic_rational_graph, "y", x=a),
        (1 + a * (2 + a * (3 + a * 5))) / (3 + a * (2 + a * (4 + a * 3))),
    )


@pytest.mark.skip(
    "ONNX Runtime generates colliding internal identifiers for function nodes."
)
def test_complex_nested_calls(onnx_helper, cubic_rational_graph_2x3):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(cubic_rational_graph_2x3, "y", x=a),
        2 * (1 + a * (2 + a * (3 + a * 5))) / (3 + a * (2 + a * (4 + a * 3))) + 3,
    )


def test_to_function_isnan(onnx_helper, isnan_graph):
    nan = numpy.array(numpy.nan, dtype=numpy.float64)
    onnx_helper.assert_close(
        onnx_helper.run(
            isnan_graph,
            "count",
            x=numpy.array(0.0),
            y=numpy.array(1.0),
            z=numpy.array(2.0),
        ),
        0,
    )
    onnx_helper.assert_close(
        onnx_helper.run(isnan_graph, "count", x=nan, y=numpy.array(3.0), z=nan),
        2,
    )
