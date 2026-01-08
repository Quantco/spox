# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

"""
This test implements a custom operator available in ONNX Runtime, showcasing the components available for extension.
The operator: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftinverse
Of these, only defining the ``Attributes/Inputs/Outputs`` classes as members is necessary, though the typehints
for the respective fields ``attrs/inputs/outputs`` and ``infer_output_types`` will be useful as well.
Of these, ``propagate_values`` is probably least common.
"""

from dataclasses import dataclass

import numpy as np

import spox.opset.ai.onnx.v17 as op
from spox import Var
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._graph import arguments, results
from spox._node import Node, OpType
from spox._type_system import Tensor, Type
from spox._value_prop import PropDict, PropValue
from spox._var import _VarInfo


# Define the Node for this operator - need to know the attributes, inputs and outputs statically
class Inverse(Node):
    op_type = OpType("Inverse", "com.microsoft", 1)

    # Define types for attributes, inputs & outputs
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    # This is optional, but is useful when defining the inference functions below.
    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

    def infer_output_types(self, input_prop_values) -> dict[str, Type]:
        # This is technically optional, but using an operator without type inference may be inconvenient.
        if self.inputs.X.type is None:
            return {}
        t = self.inputs.X.unwrap_tensor()
        if t.shape is not None and t.shape[-2] != t.shape[-1]:
            raise ValueError(
                f"Only square matrices are invertible (got {t.shape[-2]} x {t.shape[-1]})."
            )
        return {"Y": t}

    def propagate_values(self, initializers: PropDict) -> PropDict:
        # This is optional and implements value propagation ('partial data propagation' in ONNX).
        # In essence constant folding carried through for purposes of type inference.
        if initializers.get("X") is None:
            return {}
        arr = np.linalg.inv(initializers["X"].value)
        return {"Y": PropValue(Tensor(arr.dtype, arr.shape), arr)}


# Define the operator constructor which is actually used
def inverse(matrix: Var) -> Var:
    return (
        Inverse(Inverse.Attributes(), Inverse.Inputs(matrix._var_info))
        .get_output_vars(input_prop_values={"X": matrix._value})
        .Y
    )


# Test the correct runtime behaviour with ORT
def test_basic_build(onnx_helper):
    (a,) = arguments(a=Tensor(np.float64, ("N", "N")))
    graph = results(b=inverse(a))
    onnx_helper.assert_close(
        onnx_helper.run(graph, "b", a=np.array([[1.0, 1.0], [0.0, 1.0]])),
        np.array([[1.0, -1.0], [0.0, 1.0]]),
    )


def test_node_overrides():
    f = np.array([[1, 0], [1, 1]], dtype=np.float64)
    a = op.constant(value=f)
    b = inverse(a)
    assert b.type == Tensor(np.float64, (2, 2))
    np.testing.assert_allclose(b._get_value(), np.linalg.inv(f))
