from typing import Generic, TypeVar

import numpy
import pytest

from steelix._type_inference import InferenceError, InferenceFlag, InferenceWarning
from steelix.arrow import Arrow
from steelix.arrowfields import ArrowFields
from steelix.attrfields import NoAttrs
from steelix.defnode import DefNode
from steelix.graph import arguments
from steelix.node import OpType
from steelix.shape import ShapeError
from steelix.type_system import Tensor

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
Ti32 = TypeVar("Ti32")
Ti64 = TypeVar("Ti64")


def dummy_arrow(value_type):
    (a,) = arguments(_=value_type)
    return a


class SimpleTestNode(DefNode):
    class Inputs(ArrowFields, Generic[T, T1, T2, Ti32]):
        A: Arrow[T]
        B: Arrow[T]
        Y: Arrow[T1]
        Z: Arrow[T2]
        K: Arrow[Ti32]

    class Outputs(ArrowFields, Generic[T, T2, Ti64]):
        C: Arrow[T]
        R: Arrow[T2]
        P: Arrow[Ti64]

    op_type = OpType("SimpleTest", "steelix.test", 0)
    type_members = dict(Ti32={Tensor(numpy.int32)}, Ti64={Tensor(numpy.int64)})
    attrs: NoAttrs
    inputs: Inputs
    outputs: Outputs


class SimpleTStrictTestNode(DefNode):
    class Inputs(ArrowFields, Generic[T, T1, T2, Ti32]):
        A: Arrow[T]
        B: Arrow[T]
        Y: Arrow[T1]
        Z: Arrow[T2]
        K: Arrow[Ti32]

    class Outputs(ArrowFields, Generic[T, T2, Ti64]):
        C: Arrow[T]
        R: Arrow[T2]
        P: Arrow[Ti64]

    op_type = OpType("SimpleTest", "steelix.test", 0)
    type_members = dict(Ti32={Tensor(numpy.int32)}, Ti64={Tensor(numpy.int64)})
    inference_flags = dict(T={InferenceFlag.STRICT})
    attrs: NoAttrs
    inputs: Inputs
    outputs: Outputs


class BroadcastingTestNode(DefNode):
    class Inputs(ArrowFields, Generic[T, T1, T2]):
        A: Arrow[T]
        B: Arrow[T]
        Y: Arrow[T1]
        Z: Arrow[T2]

    class Outputs(ArrowFields, Generic[T, T2]):
        C: Arrow[T]
        R: Arrow[T2]

    op_type = OpType("BroadcastingTest", "steelix.test", 0)
    inference_flags = dict(T={InferenceFlag.BROADCAST})
    attrs: NoAttrs
    inputs: Inputs
    outputs: Outputs


class FailingTestNode(DefNode):
    class Inputs(ArrowFields, Generic[T1]):
        X: Arrow[T1]

    class Outputs(ArrowFields, Generic[T2]):
        Y: Arrow[T2]

    op_type = OpType("FailingTest", "steelix.test", 0)
    attrs: NoAttrs
    inputs: Inputs
    outputs: Outputs


def test_simple_inference():
    node = SimpleTestNode(
        NoAttrs(),
        SimpleTestNode.Inputs(
            A=dummy_arrow(Tensor(numpy.int32, (2, 3, "x"))),
            B=dummy_arrow(Tensor(numpy.int32, (2, 3, "x"))),
            Y=dummy_arrow(Tensor(numpy.int64, (1, 2, 3))),
            Z=dummy_arrow(Tensor(numpy.float32, (4, 5, 6))),
            K=dummy_arrow(Tensor(numpy.int32, (7, 8, 9))),
        ),
    )
    assert node.outputs.C.type == Tensor(numpy.int32, (2, 3, "x"))
    assert node.outputs.R.type == Tensor(numpy.float32, (4, 5, 6))
    assert node.outputs.P.type == Tensor(numpy.int64)


def test_incompatible_generic_assignment_errors():
    with pytest.raises(InferenceError):
        SimpleTestNode(
            NoAttrs(),
            SimpleTestNode.Inputs(
                A=dummy_arrow(Tensor(numpy.int32, (2, 3, "x"))),
                B=dummy_arrow(Tensor(numpy.int64, (2, 4, "x"))),
                Y=dummy_arrow(Tensor(numpy.int64, (1, 2, 3))),
                Z=dummy_arrow(Tensor(numpy.float32, (4, 5, 6))),
                K=dummy_arrow(Tensor(numpy.int32, (7, 8, 9))),
            ),
        )


def test_incompatible_strict_generic_assignment_errors():
    with pytest.raises(InferenceError):
        SimpleTStrictTestNode(
            NoAttrs(),
            SimpleTStrictTestNode.Inputs(
                A=dummy_arrow(Tensor(numpy.int32, (2, 3, "x"))),
                B=dummy_arrow(Tensor(numpy.int32, (2, 4, "x"))),
                Y=dummy_arrow(Tensor(numpy.int64, (1, 2, 3))),
                Z=dummy_arrow(Tensor(numpy.float32, (4, 5, 6))),
                K=dummy_arrow(Tensor(numpy.int32, (7, 8, 9))),
            ),
        )


def test_incompatible_constant_assignment_errors():
    with pytest.raises(InferenceError):
        SimpleTestNode(
            NoAttrs(),
            SimpleTestNode.Inputs(
                A=dummy_arrow(Tensor(numpy.int32, (2, 3, "x"))),
                B=dummy_arrow(Tensor(numpy.int32, (2, 3, "x"))),
                Y=dummy_arrow(Tensor(numpy.int64, (1, 2, 3))),
                Z=dummy_arrow(Tensor(numpy.float32, (4, 5, 6))),
                K=dummy_arrow(Tensor(numpy.float64, (7, 8, 9))),
            ),
        )


def test_broadcasting_inference():
    node = BroadcastingTestNode(
        NoAttrs(),
        BroadcastingTestNode.Inputs(
            A=dummy_arrow(Tensor(numpy.int32, (4, None, 1, "x"))),
            B=dummy_arrow(Tensor(numpy.int32, (1, 1, 4, None, 3, "x"))),
            Y=dummy_arrow(Tensor(numpy.int64)),
            Z=dummy_arrow(Tensor(numpy.float32)),
        ),
    )
    assert node.outputs.C.type == Tensor(numpy.int32, (1, 1, 4, None, 3, "x"))
    assert node.outputs.R.type == Tensor(numpy.float32)


def test_broadcasting_mismatched_element_errors():
    with pytest.raises(ShapeError):
        BroadcastingTestNode(
            NoAttrs(),
            BroadcastingTestNode.Inputs(
                A=dummy_arrow(Tensor(numpy.int32, (2, 3))),
                B=dummy_arrow(Tensor(numpy.int32, (2, 4))),
                Y=dummy_arrow(Tensor(numpy.int64)),
                Z=dummy_arrow(Tensor(numpy.float32)),
            ),
        )


def test_impossible_broadcasting_errors():
    with pytest.raises(InferenceError):
        BroadcastingTestNode(
            NoAttrs(),
            BroadcastingTestNode.Inputs(
                A=dummy_arrow(Tensor(numpy.int32, (4, None, 1, "x"))),
                B=dummy_arrow(Tensor(numpy.int64, (1, 1, 4, None, 3, "x"))),
                Y=dummy_arrow(Tensor(numpy.int64)),
                Z=dummy_arrow(Tensor(numpy.float32)),
            ),
        )


def test_impossible_inference():
    inputs = dummy_arrow(Tensor(numpy.int32, (2, 3)))
    with pytest.warns(InferenceWarning):
        node = FailingTestNode(NoAttrs(), FailingTestNode.Inputs(inputs))
    assert node.outputs.Y.type is None
