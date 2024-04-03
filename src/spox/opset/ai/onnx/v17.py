# ruff: noqa: E741 -- Allow ambiguous variable name
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)
from typing import cast as typing_cast

import numpy as np
import numpy.typing as npt

from spox._attributes import (
    AttrDtype,
    AttrFloat32,
    AttrFloat32s,
    AttrGraph,
    AttrInt64,
    AttrInt64s,
    AttrString,
    AttrStrings,
    AttrTensor,
    AttrType,
)
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._graph import Graph, subgraph
from spox._node import OpType
from spox._standard import InferenceError, StandardNode
from spox._type_system import Sequence as SpoxSequence
from spox._type_system import Tensor, Type
from spox._value_prop import PropValueType
from spox._var import Var


class _Abs(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Abs", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Acos(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Acos", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Acosh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Acosh", "", 9)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Add(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Add", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _And(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("And", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ArgMax(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        keepdims: AttrInt64
        select_last_index: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ArgMax", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ArgMin(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        keepdims: AttrInt64
        select_last_index: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ArgMin", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Asin(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Asin", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Asinh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Asinh", "", 9)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Atan(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Atan", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Atanh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Atanh", "", 9)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _AveragePool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        ceil_mode: AttrInt64
        count_include_pad: AttrInt64
        kernel_shape: AttrInt64s
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("AveragePool", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _BatchNormalization(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        epsilon: AttrFloat32
        momentum: AttrFloat32
        training_mode: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        scale: Var
        B: Var
        input_mean: Var
        input_var: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var
        running_mean: Optional[Var]
        running_var: Optional[Var]

    op_type = OpType("BatchNormalization", "", 15)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Bernoulli(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: Optional[AttrDtype]
        seed: Optional[AttrFloat32]

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Bernoulli", "", 15)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _BitShift(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        direction: AttrString

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        Y: Var

    @dataclass
    class Outputs(BaseOutputs):
        Z: Var

    op_type = OpType("BitShift", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _BlackmanWindow(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        output_datatype: AttrInt64
        periodic: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        size: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("BlackmanWindow", "", 17)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Cast(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        to: AttrDtype

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Cast", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _CastLike(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        target_type: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("CastLike", "", 15)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Ceil(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Ceil", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Celu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Celu", "", 12)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Clip(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        min: Optional[Var]
        max: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Clip", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Compress(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: Optional[AttrInt64]

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        condition: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    def infer_output_types(self) -> Dict[str, Type]:
        self.infer_output_types_onnx()
        inp, cond = (
            self.inputs.input.unwrap_tensor(),
            self.inputs.condition.unwrap_tensor(),
        )
        if not inp.shape:
            return {"output": Tensor(inp.dtype, None)}
        if cond.dtype != np.dtype(bool):
            raise InferenceError("Compress input 'condition' must be a boolean dtype.")
        if cond.shape and len(cond.shape) != 1:
            raise InferenceError(
                "Compress input 'condition' must be a vector (of rank 1)."
            )
        if self.attrs.axis is not None:
            shape = list(inp.shape)
            axis = self.attrs.axis.value
            if not (-len(shape) <= axis < len(shape)):
                raise InferenceError(
                    f"Compress attribute 'axis' must in range [-rank, rank-1] (rank={len(shape)})."
                )
            shape[axis] = None
        else:
            shape = [None]
        return {"output": Tensor(inp.dtype, tuple(shape))}

    op_type = OpType("Compress", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Concat(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        inputs: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        concat_result: Var

    op_type = OpType("Concat", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ConcatFromSequence(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        new_axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input_sequence: Var

    @dataclass
    class Outputs(BaseOutputs):
        concat_result: Var

    op_type = OpType("ConcatFromSequence", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Constant(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        value: Optional[AttrTensor]
        value_float: Optional[AttrFloat32]
        value_floats: Optional[AttrFloat32s]
        value_int: Optional[AttrInt64]
        value_ints: Optional[AttrInt64s]
        value_string: Optional[AttrString]
        value_strings: Optional[AttrStrings]

    Inputs = BaseInputs

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    def propagate_values(self) -> Dict[str, PropValueType]:
        ((key, raw),) = (
            (k, v.value) for k, v in self.attrs.get_fields().items() if v is not None
        )
        if key == "value":
            value = raw
        elif key == "value_float":
            value = np.array(raw, dtype=np.float32)
        elif key == "value_int":
            value = np.array(raw, dtype=np.int64)
        elif key == "value_string":
            value = np.array(raw, dtype=np.str_)
        elif key == "value_floats":
            value = np.array(list(raw), dtype=np.float32).reshape(-1)
        elif key == "value_ints":
            value = np.array(list(raw), dtype=np.int64).reshape(-1)
        elif key == "value_strings":
            value = np.array(list(raw), dtype=np.str_).reshape(-1)
        elif key == "sparse_value":
            return {}
        else:
            raise RuntimeError(
                f"Could not extract the set Constant value attribute, got: {key}"
            )
        return {"output": value}

    op_type = OpType("Constant", "", 13)

    attrs: Attributes
    inputs: BaseInputs
    outputs: Outputs


class _ConstantOfShape(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        value: Optional[AttrTensor]

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("ConstantOfShape", "", 9)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Conv(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        dilations: Optional[AttrInt64s]
        group: AttrInt64
        kernel_shape: Optional[AttrInt64s]
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        W: Var
        B: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Conv", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ConvInteger(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        dilations: Optional[AttrInt64s]
        group: AttrInt64
        kernel_shape: Optional[AttrInt64s]
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        x: Var
        w: Var
        x_zero_point: Optional[Var]
        w_zero_point: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        y: Var

    op_type = OpType("ConvInteger", "", 10)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ConvTranspose(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        dilations: Optional[AttrInt64s]
        group: AttrInt64
        kernel_shape: Optional[AttrInt64s]
        output_padding: Optional[AttrInt64s]
        output_shape: Optional[AttrInt64s]
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        W: Var
        B: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("ConvTranspose", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Cos(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Cos", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Cosh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Cosh", "", 9)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _CumSum(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        exclusive: AttrInt64
        reverse: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        x: Var
        axis: Var

    @dataclass
    class Outputs(BaseOutputs):
        y: Var

    op_type = OpType("CumSum", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _DFT(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        inverse: AttrInt64
        onesided: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        dft_length: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("DFT", "", 17)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _DepthToSpace(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        blocksize: AttrInt64
        mode: AttrString

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("DepthToSpace", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _DequantizeLinear(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        x: Var
        x_scale: Var
        x_zero_point: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        y: Var

    op_type = OpType("DequantizeLinear", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Det(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Det", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Div(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Div", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Dropout(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        seed: Optional[AttrInt64]

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        ratio: Optional[Var]
        training_mode: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var
        mask: Optional[Var]

    op_type = OpType("Dropout", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _DynamicQuantizeLinear(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        x: Var

    @dataclass
    class Outputs(BaseOutputs):
        y: Var
        y_scale: Var
        y_zero_point: Var

    op_type = OpType("DynamicQuantizeLinear", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Einsum(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        equation: AttrString

    @dataclass
    class Inputs(BaseInputs):
        Inputs: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Output: Var

    op_type = OpType("Einsum", "", 12)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Elu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Elu", "", 6)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Equal(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Equal", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Erf(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Erf", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Exp(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Exp", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Expand(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        shape: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Expand", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _EyeLike(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: Optional[AttrDtype]
        k: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("EyeLike", "", 9)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Flatten(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Flatten", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Floor(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Floor", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GRU(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        activation_alpha: Optional[AttrFloat32s]
        activation_beta: Optional[AttrFloat32s]
        activations: Optional[AttrStrings]
        clip: Optional[AttrFloat32]
        direction: AttrString
        hidden_size: Optional[AttrInt64]
        layout: AttrInt64
        linear_before_reset: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        W: Var
        R: Var
        B: Optional[Var]
        sequence_lens: Optional[Var]
        initial_h: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Optional[Var]
        Y_h: Optional[Var]

    op_type = OpType("GRU", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Gather(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        indices: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Gather", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GatherElements(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        indices: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("GatherElements", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GatherND(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        batch_dims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        indices: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("GatherND", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Gemm(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32
        beta: AttrFloat32
        transA: AttrInt64
        transB: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var
        C: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Gemm", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GlobalAveragePool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("GlobalAveragePool", "", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GlobalLpPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        p: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("GlobalLpPool", "", 2)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GlobalMaxPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("GlobalMaxPool", "", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Greater(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Greater", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GreaterOrEqual(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("GreaterOrEqual", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GridSample(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        align_corners: AttrInt64
        mode: AttrString
        padding_mode: AttrString

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        grid: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("GridSample", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _HammingWindow(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        output_datatype: AttrInt64
        periodic: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        size: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("HammingWindow", "", 17)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _HannWindow(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        output_datatype: AttrInt64
        periodic: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        size: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("HannWindow", "", 17)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _HardSigmoid(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32
        beta: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("HardSigmoid", "", 6)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _HardSwish(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("HardSwish", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Hardmax(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Hardmax", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Identity(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Identity", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _If(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        else_branch: AttrGraph
        then_branch: AttrGraph

    @dataclass
    class Inputs(BaseInputs):
        cond: Var

    @dataclass
    class Outputs(BaseOutputs):
        outputs: Sequence[Var]

    op_type = OpType("If", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _InstanceNormalization(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        epsilon: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        scale: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("InstanceNormalization", "", 6)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _IsInf(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        detect_negative: AttrInt64
        detect_positive: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("IsInf", "", 10)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _IsNaN(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("IsNaN", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LRN(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32
        beta: AttrFloat32
        bias: AttrFloat32
        size: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("LRN", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LSTM(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        activation_alpha: Optional[AttrFloat32s]
        activation_beta: Optional[AttrFloat32s]
        activations: Optional[AttrStrings]
        clip: Optional[AttrFloat32]
        direction: AttrString
        hidden_size: Optional[AttrInt64]
        input_forget: AttrInt64
        layout: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        W: Var
        R: Var
        B: Optional[Var]
        sequence_lens: Optional[Var]
        initial_h: Optional[Var]
        initial_c: Optional[Var]
        P: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Optional[Var]
        Y_h: Optional[Var]
        Y_c: Optional[Var]

    op_type = OpType("LSTM", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LayerNormalization(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        epsilon: AttrFloat32
        stash_type: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        Scale: Var
        B: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var
        Mean: Optional[Var]
        InvStdDev: Optional[Var]

    op_type = OpType("LayerNormalization", "", 17)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LeakyRelu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("LeakyRelu", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Less(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Less", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LessOrEqual(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("LessOrEqual", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Log(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Log", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LogSoftmax(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("LogSoftmax", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Loop(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        body: AttrGraph

    @dataclass
    class Inputs(BaseInputs):
        M: Optional[Var]
        cond: Optional[Var]
        v_initial: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        v_final_and_scan_outputs: Sequence[Var]

    def infer_output_types(self) -> Dict[str, Type]:
        output_types = super().infer_output_types()

        body = self.attrs.body.value
        n = len(body.requested_arguments) - 2

        carried_names = list(self.outputs.get_vars())[:n]
        carried_types = [v.type for v in list(body.requested_results.values())[1:][:n]]

        for name, typ in zip(carried_names, carried_types):
            output_types[name] = typ

        return output_types

    op_type = OpType("Loop", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LpNormalization(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        p: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("LpNormalization", "", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LpPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        kernel_shape: AttrInt64s
        p: AttrInt64
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("LpPool", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _MatMul(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("MatMul", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _MatMulInteger(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var
        a_zero_point: Optional[Var]
        b_zero_point: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("MatMulInteger", "", 10)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Max(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        data_0: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        max: Var

    op_type = OpType("Max", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _MaxPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        ceil_mode: AttrInt64
        dilations: Optional[AttrInt64s]
        kernel_shape: AttrInt64s
        pads: Optional[AttrInt64s]
        storage_order: AttrInt64
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var
        Indices: Optional[Var]

    op_type = OpType("MaxPool", "", 12)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _MaxRoiPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pooled_shape: AttrInt64s
        spatial_scale: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        rois: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("MaxRoiPool", "", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _MaxUnpool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        kernel_shape: AttrInt64s
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        I: Var
        output_shape: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("MaxUnpool", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Mean(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        data_0: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        mean: Var

    op_type = OpType("Mean", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _MeanVarianceNormalization(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: AttrInt64s

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("MeanVarianceNormalization", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _MelWeightMatrix(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        output_datatype: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        num_mel_bins: Var
        dft_length: Var
        sample_rate: Var
        lower_edge_hertz: Var
        upper_edge_hertz: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("MelWeightMatrix", "", 17)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Min(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        data_0: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        min: Var

    op_type = OpType("Min", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Mod(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        fmod: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Mod", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Mul(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Mul", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Multinomial(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype
        sample_size: AttrInt64
        seed: Optional[AttrFloat32]

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Multinomial", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Neg(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Neg", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _NegativeLogLikelihoodLoss(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        ignore_index: Optional[AttrInt64]
        reduction: AttrString

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        target: Var
        weight: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        loss: Var

    op_type = OpType("NegativeLogLikelihoodLoss", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _NonMaxSuppression(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        center_point_box: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        boxes: Var
        scores: Var
        max_output_boxes_per_class: Optional[Var]
        iou_threshold: Optional[Var]
        score_threshold: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        selected_indices: Var

    op_type = OpType("NonMaxSuppression", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _NonZero(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("NonZero", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Not(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Not", "", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _OneHot(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        indices: Var
        depth: Var
        values: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("OneHot", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Optional(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        type: Optional[AttrType]

    @dataclass
    class Inputs(BaseInputs):
        input: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Optional", "", 15)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _OptionalGetElement(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("OptionalGetElement", "", 15)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _OptionalHasElement(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("OptionalHasElement", "", 15)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Or(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Or", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _PRelu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        slope: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("PRelu", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Pad(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        mode: AttrString

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        pads: Var
        constant_value: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Pad", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Pow(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        Y: Var

    @dataclass
    class Outputs(BaseOutputs):
        Z: Var

    op_type = OpType("Pow", "", 15)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _QLinearConv(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        dilations: Optional[AttrInt64s]
        group: AttrInt64
        kernel_shape: Optional[AttrInt64s]
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        x: Var
        x_scale: Var
        x_zero_point: Var
        w: Var
        w_scale: Var
        w_zero_point: Var
        y_scale: Var
        y_zero_point: Var
        B: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        y: Var

    op_type = OpType("QLinearConv", "", 10)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _QLinearMatMul(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        a: Var
        a_scale: Var
        a_zero_point: Var
        b: Var
        b_scale: Var
        b_zero_point: Var
        y_scale: Var
        y_zero_point: Var

    @dataclass
    class Outputs(BaseOutputs):
        y: Var

    op_type = OpType("QLinearMatMul", "", 10)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _QuantizeLinear(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        x: Var
        y_scale: Var
        y_zero_point: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        y: Var

    op_type = OpType("QuantizeLinear", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _RNN(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        activation_alpha: Optional[AttrFloat32s]
        activation_beta: Optional[AttrFloat32s]
        activations: AttrStrings
        clip: Optional[AttrFloat32]
        direction: AttrString
        hidden_size: Optional[AttrInt64]
        layout: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        W: Var
        R: Var
        B: Optional[Var]
        sequence_lens: Optional[Var]
        initial_h: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Optional[Var]
        Y_h: Optional[Var]

    op_type = OpType("RNN", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _RandomNormal(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype
        mean: AttrFloat32
        scale: AttrFloat32
        seed: Optional[AttrFloat32]
        shape: AttrInt64s

    Inputs = BaseInputs

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("RandomNormal", "", 1)

    attrs: Attributes
    inputs: BaseInputs
    outputs: Outputs


class _RandomNormalLike(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: Optional[AttrDtype]
        mean: AttrFloat32
        scale: AttrFloat32
        seed: Optional[AttrFloat32]

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("RandomNormalLike", "", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _RandomUniform(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype
        high: AttrFloat32
        low: AttrFloat32
        seed: Optional[AttrFloat32]
        shape: AttrInt64s

    Inputs = BaseInputs

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("RandomUniform", "", 1)

    attrs: Attributes
    inputs: BaseInputs
    outputs: Outputs


class _RandomUniformLike(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: Optional[AttrDtype]
        high: AttrFloat32
        low: AttrFloat32
        seed: Optional[AttrFloat32]

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("RandomUniformLike", "", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Range(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        start: Var
        limit: Var
        delta: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Range", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Reciprocal(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Reciprocal", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceL1(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceL1", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceL2(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceL2", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceLogSum(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceLogSum", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceLogSumExp(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceLogSumExp", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceMax(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceMax", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceMean(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceMean", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceMin(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceMin", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceProd(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceProd", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceSum(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        keepdims: AttrInt64
        noop_with_empty_axes: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        axes: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceSum", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceSumSquare(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceSumSquare", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Relu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Relu", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Reshape(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        allowzero: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        shape: Var

    @dataclass
    class Outputs(BaseOutputs):
        reshaped: Var

    op_type = OpType("Reshape", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Resize(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        coordinate_transformation_mode: AttrString
        cubic_coeff_a: AttrFloat32
        exclude_outside: AttrInt64
        extrapolation_value: AttrFloat32
        mode: AttrString
        nearest_mode: AttrString

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        roi: Optional[Var]
        scales: Optional[Var]
        sizes: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Resize", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReverseSequence(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        batch_axis: AttrInt64
        time_axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        sequence_lens: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("ReverseSequence", "", 10)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _RoiAlign(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        coordinate_transformation_mode: AttrString
        mode: AttrString
        output_height: AttrInt64
        output_width: AttrInt64
        sampling_ratio: AttrInt64
        spatial_scale: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        rois: Var
        batch_indices: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("RoiAlign", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Round(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Round", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _STFT(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        onesided: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        signal: Var
        frame_step: Var
        window: Optional[Var]
        frame_length: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("STFT", "", 17)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Scan(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        body: AttrGraph
        num_scan_inputs: AttrInt64
        scan_input_axes: Optional[AttrInt64s]
        scan_input_directions: Optional[AttrInt64s]
        scan_output_axes: Optional[AttrInt64s]
        scan_output_directions: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        initial_state_and_scan_inputs: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        final_state_and_scan_outputs: Sequence[Var]

    op_type = OpType("Scan", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ScatterElements(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        reduction: AttrString

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        indices: Var
        updates: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("ScatterElements", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ScatterND(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        reduction: AttrString

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        indices: Var
        updates: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("ScatterND", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Selu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32
        gamma: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Selu", "", 6)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SequenceAt(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input_sequence: Var
        position: Var

    @dataclass
    class Outputs(BaseOutputs):
        tensor: Var

    op_type = OpType("SequenceAt", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SequenceConstruct(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        inputs: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output_sequence: Var

    op_type = OpType("SequenceConstruct", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SequenceEmpty(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: Optional[AttrDtype]

    Inputs = BaseInputs

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("SequenceEmpty", "", 11)

    attrs: Attributes
    inputs: BaseInputs
    outputs: Outputs


class _SequenceErase(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input_sequence: Var
        position: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output_sequence: Var

    op_type = OpType("SequenceErase", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SequenceInsert(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input_sequence: Var
        tensor: Var
        position: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output_sequence: Var

    op_type = OpType("SequenceInsert", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SequenceLength(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input_sequence: Var

    @dataclass
    class Outputs(BaseOutputs):
        length: Var

    op_type = OpType("SequenceLength", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SequenceMap(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        body: AttrGraph

    @dataclass
    class Inputs(BaseInputs):
        input_sequence: Var
        additional_inputs: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        out_sequence: Sequence[Var]

    op_type = OpType("SequenceMap", "", 17)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Shape(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        end: Optional[AttrInt64]
        start: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        shape: Var

    op_type = OpType("Shape", "", 15)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Shrink(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        bias: AttrFloat32
        lambd: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Shrink", "", 9)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Sigmoid(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Sigmoid", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Sign(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Sign", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Sin(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Sin", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Sinh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Sinh", "", 9)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Size(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        size: Var

    op_type = OpType("Size", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Slice(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        starts: Var
        ends: Var
        axes: Optional[Var]
        steps: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Slice", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Softmax(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Softmax", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SoftmaxCrossEntropyLoss(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        ignore_index: Optional[AttrInt64]
        reduction: AttrString

    @dataclass
    class Inputs(BaseInputs):
        scores: Var
        labels: Var
        weights: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var
        log_prob: Optional[Var]

    op_type = OpType("SoftmaxCrossEntropyLoss", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Softplus(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Softplus", "", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Softsign(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Softsign", "", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SpaceToDepth(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        blocksize: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("SpaceToDepth", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Split(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        split: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        outputs: Sequence[Var]

    op_type = OpType("Split", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SplitToSequence(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        keepdims: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        split: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output_sequence: Var

    op_type = OpType("SplitToSequence", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Sqrt(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Sqrt", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Squeeze(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        axes: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        squeezed: Var

    op_type = OpType("Squeeze", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _StringNormalizer(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        case_change_action: AttrString
        is_case_sensitive: AttrInt64
        locale: Optional[AttrString]
        stopwords: Optional[AttrStrings]

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("StringNormalizer", "", 10)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Sub(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Sub", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Sum(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        data_0: Sequence[Var]

    @dataclass
    class Outputs(BaseOutputs):
        sum: Var

    op_type = OpType("Sum", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Tan(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Tan", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Tanh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Tanh", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _TfIdfVectorizer(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        max_gram_length: AttrInt64
        max_skip_count: AttrInt64
        min_gram_length: AttrInt64
        mode: AttrString
        ngram_counts: AttrInt64s
        ngram_indexes: AttrInt64s
        pool_int64s: Optional[AttrInt64s]
        pool_strings: Optional[AttrStrings]
        weights: Optional[AttrFloat32s]

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("TfIdfVectorizer", "", 9)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ThresholdedRelu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("ThresholdedRelu", "", 10)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Tile(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        repeats: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Tile", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _TopK(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        largest: AttrInt64
        sorted: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        K: Var

    @dataclass
    class Outputs(BaseOutputs):
        Values: Var
        Indices: Var

    op_type = OpType("TopK", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Transpose(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        perm: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        data: Var

    @dataclass
    class Outputs(BaseOutputs):
        transposed: Var

    op_type = OpType("Transpose", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Trilu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        upper: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        k: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Trilu", "", 14)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Unique(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: Optional[AttrInt64]
        sorted: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var
        indices: Optional[Var]
        inverse_indices: Optional[Var]
        counts: Optional[Var]

    op_type = OpType("Unique", "", 11)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Unsqueeze(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        data: Var
        axes: Var

    @dataclass
    class Outputs(BaseOutputs):
        expanded: Var

    op_type = OpType("Unsqueeze", "", 13)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Where(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        condition: Var
        X: Var
        Y: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Where", "", 16)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Xor(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        A: Var
        B: Var

    @dataclass
    class Outputs(BaseOutputs):
        C: Var

    op_type = OpType("Xor", "", 7)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def abs(
    X: Var,
) -> Var:
    r"""
    Absolute takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where absolute value, y = abs(x), is applied to the tensor
    elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Abs``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Abs(
        _Abs.Attributes(),
        _Abs.Inputs(
            X=X,
        ),
    ).outputs.Y


def acos(
    input: Var,
) -> Var:
    r"""
    Calculates the arccosine (inverse of cosine) of the given input tensor,
    element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The arccosine of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@7::Acos``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Acos(
        _Acos.Attributes(),
        _Acos.Inputs(
            input=input,
        ),
    ).outputs.output


def acosh(
    input: Var,
) -> Var:
    r"""
    Calculates the hyperbolic arccosine of the given input tensor
    element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The hyperbolic arccosine values of the input tensor computed
        element-wise

    Notes
    =====
    Signature: ``ai.onnx@9::Acosh``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Acosh(
        _Acosh.Attributes(),
        _Acosh.Inputs(
            input=input,
        ),
    ).outputs.output


def add(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Performs element-wise binary addition (with Numpy-style broadcasting
    support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    (Opset 14 change): Extend supported types to include uint8, int8,
    uint16, and int16.

    Parameters
    ==========
    A
        Type T.
        First operand.
    B
        Type T.
        Second operand.

    Returns
    =======
    C : Var
        Type T.
        Result, has same element type as two inputs

    Notes
    =====
    Signature: ``ai.onnx@14::Add``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Add(
        _Add.Attributes(),
        _Add.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def and_(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulted from performing the ``and`` logical
    operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the logical operator.
    B
        Type T.
        Second input operand for the logical operator.

    Returns
    =======
    C : Var
        Type T1.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@7::And``.

    Type constraints:
     - T: `tensor(bool)`
     - T1: `tensor(bool)`
    """
    return _And(
        _And.Attributes(),
        _And.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def arg_max(
    data: Var,
    *,
    axis: int = 0,
    keepdims: int = 1,
    select_last_index: int = 0,
) -> Var:
    r"""
    Computes the indices of the max elements of the input tensor's element
    along the provided axis. The resulting tensor has the same rank as the
    input if keepdims equals 1. If keepdims equals 0, then the resulting
    tensor has the reduced dimension pruned. If select_last_index is True
    (default False), the index of the last occurrence of the max is selected
    if the max appears more than once in the input. Otherwise the index of
    the first occurrence is selected. The type of the output tensor is
    integer.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axis
        Attribute.
        The axis in which to compute the arg indices. Accepted range is [-r,
        r-1] where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.
    select_last_index
        Attribute.
        Whether to select the last index or the first index if the {name}
        appears in multiple indices, default is False (first index).

    Returns
    =======
    reduced : Var
        Type tensor(int64).
        Reduced output tensor with integer data type.

    Notes
    =====
    Signature: ``ai.onnx@13::ArgMax``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ArgMax(
        _ArgMax.Attributes(
            axis=AttrInt64(axis, name="axis"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
            select_last_index=AttrInt64(select_last_index, name="select_last_index"),
        ),
        _ArgMax.Inputs(
            data=data,
        ),
    ).outputs.reduced


def arg_min(
    data: Var,
    *,
    axis: int = 0,
    keepdims: int = 1,
    select_last_index: int = 0,
) -> Var:
    r"""
    Computes the indices of the min elements of the input tensor's element
    along the provided axis. The resulting tensor has the same rank as the
    input if keepdims equals 1. If keepdims equals 0, then the resulting
    tensor has the reduced dimension pruned. If select_last_index is True
    (default False), the index of the last occurrence of the min is selected
    if the min appears more than once in the input. Otherwise the index of
    the first occurrence is selected. The type of the output tensor is
    integer.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axis
        Attribute.
        The axis in which to compute the arg indices. Accepted range is [-r,
        r-1] where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.
    select_last_index
        Attribute.
        Whether to select the last index or the first index if the {name}
        appears in multiple indices, default is False (first index).

    Returns
    =======
    reduced : Var
        Type tensor(int64).
        Reduced output tensor with integer data type.

    Notes
    =====
    Signature: ``ai.onnx@13::ArgMin``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ArgMin(
        _ArgMin.Attributes(
            axis=AttrInt64(axis, name="axis"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
            select_last_index=AttrInt64(select_last_index, name="select_last_index"),
        ),
        _ArgMin.Inputs(
            data=data,
        ),
    ).outputs.reduced


def asin(
    input: Var,
) -> Var:
    r"""
    Calculates the arcsine (inverse of sine) of the given input tensor,
    element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The arcsine of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@7::Asin``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Asin(
        _Asin.Attributes(),
        _Asin.Inputs(
            input=input,
        ),
    ).outputs.output


def asinh(
    input: Var,
) -> Var:
    r"""
    Calculates the hyperbolic arcsine of the given input tensor
    element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The hyperbolic arcsine values of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@9::Asinh``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Asinh(
        _Asinh.Attributes(),
        _Asinh.Inputs(
            input=input,
        ),
    ).outputs.output


def atan(
    input: Var,
) -> Var:
    r"""
    Calculates the arctangent (inverse of tangent) of the given input
    tensor, element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The arctangent of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@7::Atan``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Atan(
        _Atan.Attributes(),
        _Atan.Inputs(
            input=input,
        ),
    ).outputs.output


def atanh(
    input: Var,
) -> Var:
    r"""
    Calculates the hyperbolic arctangent of the given input tensor
    element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The hyperbolic arctangent values of the input tensor computed
        element-wise

    Notes
    =====
    Signature: ``ai.onnx@9::Atanh``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Atanh(
        _Atanh.Attributes(),
        _Atanh.Inputs(
            input=input,
        ),
    ).outputs.output


def average_pool(
    X: Var,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    count_include_pad: int = 0,
    kernel_shape: Iterable[int],
    pads: Optional[Iterable[int]] = None,
    strides: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    AveragePool consumes an input tensor X and applies average pooling
    across the tensor according to kernel sizes, stride sizes, and pad
    lengths. average pooling consisting of computing the average on all
    values of a subset of the input tensor according to the kernel size and
    downsampling the data into the output tensor Y for further processing.
    The output spatial shape will be following:

    ::

       output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

    or

    ::

       output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

    if ceil_mode is enabled

    ::

       * pad_shape[i] is sum of pads along axis i

    ``auto_pad`` is a DEPRECATED attribute. If you are using them currently,
    the output spatial shape will be following when ceil_mode is enabled:

    ::

       VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
       SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])

    or when ceil_mode is disabled:

    ::

       VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
       SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])

    And pad shape will be following if ``SAME_UPPER`` or ``SAME_LOWER``:

    ::

       pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]

    The output of each pooling window is divided by the number of elements
    (exclude pad when attribute count_include_pad is zero).

    Parameters
    ==========
    X
        Type T.
        Input data tensor from the previous operator; dimensions for image case
        are (N x C x H x W), where N is the batch size, C is the number of
        channels, and H and W are the height and the width of the data. For non
        image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
        where N is the batch size. Optionally, if dimension denotation is in
        effect, the operation expects the input data tensor to arrive with the
        dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE,
        DATA_FEATURE ...].
    auto_pad
        Attribute.
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where
        default value is NOTSET, which means explicit padding is used.
        SAME_UPPER or SAME_LOWER mean pad the input so that
        ``output_shape[i] = ceil(input_shape[i] / strides[i])`` for each axis
        ``i``. The padding is split between the two sides equally or almost
        equally (depending on whether it is even or odd). In case the padding is
        an odd number, the extra padding is added at the end for SAME_UPPER and
        at the beginning for SAME_LOWER.
    ceil_mode
        Attribute.
        Whether to use ceil or floor (default) to compute the output shape.
    count_include_pad
        Attribute.
        Whether include pad pixels when calculating values for the edges.
        Default is 0, doesn't count include pad.
    kernel_shape
        Attribute.
        The size of the kernel along each axis.
    pads
        Attribute.
        Padding for the beginning and ending along each spatial axis, it can
        take any value greater than or equal to 0. The value represent the
        number of pixels added to the beginning and end part of the
        corresponding axis. ``pads`` format should be as follow [x1_begin,
        x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
        added at the beginning of axis ``i`` and xi_end, the number of pixels
        added at the end of axis ``i``. This attribute cannot be used
        simultaneously with auto_pad attribute. If not present, the padding
        defaults to 0 along start and end of each spatial axis.
    strides
        Attribute.
        Stride along each spatial axis. If not present, the stride defaults to 1
        along each spatial axis.

    Returns
    =======
    Y : Var
        Type T.
        Output data tensor from average or max pooling across the input tensor.
        Dimensions will vary based on various kernel, stride, and pad sizes.
        Floor value of the dimension is used

    Notes
    =====
    Signature: ``ai.onnx@11::AveragePool``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _AveragePool(
        _AveragePool.Attributes(
            auto_pad=AttrString(auto_pad, name="auto_pad"),
            ceil_mode=AttrInt64(ceil_mode, name="ceil_mode"),
            count_include_pad=AttrInt64(count_include_pad, name="count_include_pad"),
            kernel_shape=AttrInt64s(kernel_shape, name="kernel_shape"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _AveragePool.Inputs(
            X=X,
        ),
    ).outputs.Y


def batch_normalization(
    X: Var,
    scale: Var,
    B: Var,
    input_mean: Var,
    input_var: Var,
    *,
    epsilon: float = 9.999999747378752e-06,
    momentum: float = 0.8999999761581421,
    training_mode: int = 0,
) -> Tuple[Var, Var, Var]:
    r"""
    Carries out batch normalization as described in the paper
    https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
    There are five required inputs 'X', 'scale', 'B', 'input_mean' and
    'input_var'. Note that 'input_mean' and 'input_var' are expected to be
    the estimated statistics in inference mode (training_mode=False,
    default), and the running statistics in training mode
    (training_mode=True). There are multiple cases for the number of
    outputs, which we list below:

    -  Output case #1: Y, running_mean, running_var (training_mode=True)
    -  Output case #2: Y (training_mode=False)

    When training_mode=False, extra outputs are invalid. The outputs are
    updated as follows when training_mode=True:

    ::

       running_mean = input_mean * momentum + current_mean * (1 - momentum)
       running_var = input_var * momentum + current_var * (1 - momentum)

       Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B

    where:

    ::

       current_mean = ReduceMean(X, axis=all_except_channel_index)
       current_var =  ReduceVar(X, axis=all_except_channel_index)

    Notice that ``ReduceVar`` refers to the population variance, and it
    equals to ``sum(sqrd(x_i - x_avg)) / N`` where ``N`` is the population
    size (this formula does not use sample size ``N - 1``).

    The computation of ReduceMean and ReduceVar uses float to avoid overflow
    for float16 inputs.

    When training_mode=False:

    ::

       Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B

    For previous (depreciated) non-spatial cases, implementors are suggested
    to flatten the input shape to (N x C \* D1 \* D2 \* ... \* Dn) before a
    BatchNormalization Op. This operator has **optional** inputs/outputs.
    See `the doc <https://github.com/onnx/onnx/blob/main/docs/IR.md>`__ for
    more details about the representation of optional arguments. An empty
    string may be used in the place of an actual argument's name to indicate
    a missing argument. Trailing optional arguments (those not followed by
    an argument that is present) may also be simply omitted.

    Parameters
    ==========
    X
        Type T.
        Input data tensor from the previous operator; dimensions are in the form
        of (N x C x D1 x D2 ... Dn), where N is the batch size, C is the number
        of channels. Statistics are computed for every channel of C over N and
        D1 to Dn dimensions. For image data, input dimensions become (N x C x H
        x W). The op also accepts single dimension input of size N in which case
        C is assumed to be 1
    scale
        Type T1.
        Scale tensor of shape (C).
    B
        Type T1.
        Bias tensor of shape (C).
    input_mean
        Type T2.
        running (training) or estimated (testing) mean tensor of shape (C).
    input_var
        Type T2.
        running (training) or estimated (testing) variance tensor of shape (C).
    epsilon
        Attribute.
        The epsilon value to use to avoid division by zero.
    momentum
        Attribute.
        Factor used in computing the running mean and variance.e.g.,
        running_mean = running_mean \* momentum + mean \* (1 - momentum).
    training_mode
        Attribute.
        If set to true, it indicates BatchNormalization is being used for
        training, and outputs 1 and 2 are to be computed.

    Returns
    =======
    Y : Var
        Type T.
        The output tensor of the same shape as X
    running_mean : Var
        Type T2.
        The running mean after the BatchNormalization operator.
    running_var : Var
        Type T2.
        The running variance after the BatchNormalization operator. This op uses
        the population size (N) for calculating variance, and not the sample
        size N-1.

    Notes
    =====
    Signature: ``ai.onnx@15::BatchNormalization``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _BatchNormalization(
        _BatchNormalization.Attributes(
            epsilon=AttrFloat32(epsilon, name="epsilon"),
            momentum=AttrFloat32(momentum, name="momentum"),
            training_mode=AttrInt64(training_mode, name="training_mode"),
        ),
        _BatchNormalization.Inputs(
            X=X,
            scale=scale,
            B=B,
            input_mean=input_mean,
            input_var=input_var,
        ),
    ).outputs._unpack_to_any()


def bernoulli(
    input: Var,
    *,
    dtype: Optional[npt.DTypeLike] = None,
    seed: Optional[float] = None,
) -> Var:
    r"""
    Draws binary random numbers (0 or 1) from a Bernoulli distribution. The
    input tensor should be a tensor containing probabilities p (a value in
    the range [0,1]) to be used for drawing the binary random number, where
    an output of 1 is produced with probability p and an output of 0 is
    produced with probability (1-p).

    This operator is non-deterministic and may not produce the same values
    in different implementations (even if a seed is specified).

    Parameters
    ==========
    input
        Type T1.
        All values in input have to be in the range:[0, 1].
    dtype
        Attribute.
        The data type for the elements of the output tensor. if not specified,
        we will use the data type of the input tensor.
    seed
        Attribute.
        (Optional) Seed to the random generator, if not specified we will auto
        generate one.

    Returns
    =======
    output : Var
        Type T2.
        The returned output tensor only has values 0 or 1, same shape as input
        tensor.

    Notes
    =====
    Signature: ``ai.onnx@15::Bernoulli``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Bernoulli(
        _Bernoulli.Attributes(
            dtype=AttrDtype.maybe(dtype, name="dtype"),
            seed=AttrFloat32.maybe(seed, name="seed"),
        ),
        _Bernoulli.Inputs(
            input=input,
        ),
    ).outputs.output


def bit_shift(
    X: Var,
    Y: Var,
    *,
    direction: str,
) -> Var:
    r"""
    Bitwise shift operator performs element-wise operation. For each input
    element, if the attribute "direction" is "RIGHT", this operator moves
    its binary representation toward the right side so that the input value
    is effectively decreased. If the attribute "direction" is "LEFT", bits
    of binary representation moves toward the left side, which results the
    increase of its actual value. The input X is the tensor to be shifted
    and another input Y specifies the amounts of shifting. For example, if
    "direction" is "Right", X is [1, 4], and S is [1, 1], the corresponding
    output Z would be [0, 2]. If "direction" is "LEFT" with X=[1, 2] and
    S=[1, 2], the corresponding output Y would be [2, 8].

    Because this operator supports Numpy-style broadcasting, X's and Y's
    shapes are not necessarily identical. This operator supports
    **multidirectional (i.e., Numpy-style) broadcasting**; for more details
    please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    X
        Type T.
        First operand, input to be shifted.
    Y
        Type T.
        Second operand, amounts of shift.
    direction
        Attribute.
        Direction of moving bits. It can be either "RIGHT" (for right shift) or
        "LEFT" (for left shift).

    Returns
    =======
    Z : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@11::BitShift``.

    Type constraints:
     - T: `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _BitShift(
        _BitShift.Attributes(
            direction=AttrString(direction, name="direction"),
        ),
        _BitShift.Inputs(
            X=X,
            Y=Y,
        ),
    ).outputs.Z


def blackman_window(
    size: Var,
    *,
    output_datatype: int = 1,
    periodic: int = 1,
) -> Var:
    r"""
    Generates a Blackman window as described in the paper
    https://ieeexplore.ieee.org/document/1455106.

    Parameters
    ==========
    size
        Type T1.
        A scalar value indicating the length of the window.
    output_datatype
        Attribute.
        The data type of the output tensor. Strictly must be one of the values
        from DataType enum in TensorProto whose values correspond to T2. The
        default value is 1 = FLOAT.
    periodic
        Attribute.
        If 1, returns a window to be used as periodic function. If 0, return a
        symmetric window. When 'periodic' is specified, hann computes a window
        of length size + 1 and returns the first size points. The default value
        is 1.

    Returns
    =======
    output : Var
        Type T2.
        A Blackman window with length: size. The output has the shape: [size].

    Notes
    =====
    Signature: ``ai.onnx@17::BlackmanWindow``.

    Type constraints:
     - T1: `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _BlackmanWindow(
        _BlackmanWindow.Attributes(
            output_datatype=AttrInt64(output_datatype, name="output_datatype"),
            periodic=AttrInt64(periodic, name="periodic"),
        ),
        _BlackmanWindow.Inputs(
            size=size,
        ),
    ).outputs.output


def cast(
    input: Var,
    *,
    to: npt.DTypeLike,
) -> Var:
    r"""
    The operator casts the elements of a given input tensor to a data type
    specified by the 'to' argument and returns an output tensor of the same
    size in the converted type. The 'to' argument must be one of the data
    types specified in the 'DataType' enum field in the TensorProto message.

    Casting from string tensor in plain (e.g., "3.14" and "1000") and
    scientific numeric representations (e.g., "1e-5" and "1E8") to float
    types is supported. For example, converting string "100.5" to an integer
    may yield result 100. There are some string literals reserved for
    special floating-point values; "+INF" (and "INF"), "-INF", and "NaN" are
    positive infinity, negative infinity, and not-a-number, respectively.
    Any string which can exactly match "+INF" in a case-insensitive way
    would be mapped to positive infinite. Similarly, this case-insensitive
    rule is applied to "INF" and "NaN". When casting from numeric tensors to
    string tensors, plain floating-point representation (such as
    "314.15926") would be used. Converting non-numerical-literal string such
    as "Hello World!" is an undefined behavior. Cases of converting string
    representing floating-point arithmetic value, such as "2.718", to INT is
    an undefined behavior.

    Conversion from a numerical type to any numerical type is always
    allowed. User must be aware of precision loss and value change caused by
    range difference between two types. For example, a 64-bit float
    3.1415926459 may be round to a 32-bit float 3.141592. Similarly,
    converting an integer 36 to Boolean may produce 1 because we truncate
    bits which can't be stored in the targeted type.

    In more detail, the conversion among numerical types should follow these
    rules:

    -  Casting from floating point to:

       -  floating point: +/- infinity if OOR (out of range).
       -  fixed point: undefined if OOR.
       -  bool: +/- 0.0 to False; all else to True.

    -  Casting from fixed point to:

       -  floating point: +/- infinity if OOR. (+ infinity in the case of
          uint)
       -  fixed point: when OOR, discard higher bits and reinterpret (with
          respect to two's complement representation for signed types). For
          example, 200 (int16) -> -56 (int8).
       -  bool: zero to False; nonzero to True.

    -  Casting from bool to:

       -  floating point: ``{1.0, 0.0}``.
       -  fixed point: ``{1, 0}``.
       -  bool: no change.

    Parameters
    ==========
    input
        Type T1.
        Input tensor to be cast.
    to
        Attribute.
        The data type to which the elements of the input tensor are cast.
        Strictly must be one of the types from DataType enum in TensorProto

    Returns
    =======
    output : Var
        Type T2.
        Output tensor with the same shape as input with type specified by the
        'to' argument

    Notes
    =====
    Signature: ``ai.onnx@13::Cast``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Cast(
        _Cast.Attributes(
            to=AttrDtype(to, name="to"),
        ),
        _Cast.Inputs(
            input=input,
        ),
    ).outputs.output


def cast_like(
    input: Var,
    target_type: Var,
) -> Var:
    r"""
    The operator casts the elements of a given input tensor (the first
    input) to the same data type as the elements of the second input tensor.
    See documentation of the Cast operator for further details.

    Parameters
    ==========
    input
        Type T1.
        Input tensor to be cast.
    target_type
        Type T2.
        The (first) input tensor will be cast to produce a tensor of the same
        type as this (second input) tensor.

    Returns
    =======
    output : Var
        Type T2.
        Output tensor produced by casting the first input tensor to have the
        same type as the second input tensor.

    Notes
    =====
    Signature: ``ai.onnx@15::CastLike``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _CastLike(
        _CastLike.Attributes(),
        _CastLike.Inputs(
            input=input,
            target_type=target_type,
        ),
    ).outputs.output


def ceil(
    X: Var,
) -> Var:
    r"""
    Ceil takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the ceil is, y = ceil(x), is applied to the tensor
    elementwise. If x is integral, +0, -0, NaN, or infinite, x itself is
    returned.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Ceil``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Ceil(
        _Ceil.Attributes(),
        _Ceil.Inputs(
            X=X,
        ),
    ).outputs.Y


def celu(
    X: Var,
    *,
    alpha: float = 1.0,
) -> Var:
    r"""
    Continuously Differentiable Exponential Linear Units: Perform the linear
    unit element-wise on the input tensor X using formula:

    ::

       max(0,x) + min(0,alpha*(exp(x/alpha)-1))

    Parameters
    ==========
    X
        Type T.
        Input tensor
    alpha
        Attribute.
        The Alpha value in Celu formula which control the shape of the unit. The
        default value is 1.0.

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@12::Celu``.

    Type constraints:
     - T: `tensor(float)`
    """
    return _Celu(
        _Celu.Attributes(
            alpha=AttrFloat32(alpha, name="alpha"),
        ),
        _Celu.Inputs(
            X=X,
        ),
    ).outputs.Y


def clip(
    input: Var,
    min: Optional[Var] = None,
    max: Optional[Var] = None,
) -> Var:
    r"""
    Clip operator limits the given input within an interval. The interval is
    specified by the inputs 'min' and 'max'. They default to
    numeric_limits::lowest() and numeric_limits::max(), respectively.

    Parameters
    ==========
    input
        Type T.
        Input tensor whose elements to be clipped
    min
        Type T.
        Minimum value, under which element is replaced by min. It must be a
        scalar(tensor of empty shape).
    max
        Type T.
        Maximum value, above which element is replaced by max. It must be a
        scalar(tensor of empty shape).

    Returns
    =======
    output : Var
        Type T.
        Output tensor with clipped input elements

    Notes
    =====
    Signature: ``ai.onnx@13::Clip``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Clip(
        _Clip.Attributes(),
        _Clip.Inputs(
            input=input,
            min=min,
            max=max,
        ),
    ).outputs.output


def compress(
    input: Var,
    condition: Var,
    *,
    axis: Optional[int] = None,
) -> Var:
    r"""
    Selects slices from an input tensor along a given axis where condition
    evaluates to True for each axis index. In case axis is not provided,
    input is flattened before elements are selected. Compress behaves like
    numpy.compress:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html

    Parameters
    ==========
    input
        Type T.
        Tensor of rank r >= 1.
    condition
        Type T1.
        Rank 1 tensor of booleans to indicate which slices or data elements to
        be selected. Its length can be less than the input length along the axis
        or the flattened input size if axis is not specified. In such cases data
        slices or elements exceeding the condition length are discarded.
    axis
        Attribute.
        (Optional) Axis along which to take slices. If not specified, input is
        flattened before elements being selected. Negative value means counting
        dimensions from the back. Accepted range is [-r, r-1] where r =
        rank(input).

    Returns
    =======
    output : Var
        Type T.
        Tensor of rank r if axis is specified. Otherwise output is a Tensor of
        rank 1.

    Notes
    =====
    Signature: ``ai.onnx@11::Compress``.

    Type constraints:
     - T: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(bool)`
    """
    return _Compress(
        _Compress.Attributes(
            axis=AttrInt64.maybe(axis, name="axis"),
        ),
        _Compress.Inputs(
            input=input,
            condition=condition,
        ),
    ).outputs.output


def concat(
    inputs: Sequence[Var],
    *,
    axis: int,
) -> Var:
    r"""
    Concatenate a list of tensors into a single tensor. All input tensors
    must have the same shape, except for the dimension size of the axis to
    concatenate on.

    Parameters
    ==========
    inputs
        Type T.
        List of tensors for concatenation
    axis
        Attribute.
        Which axis to concat on. A negative value means counting dimensions from
        the back. Accepted range is [-r, r-1] where r = rank(inputs)..

    Returns
    =======
    concat_result : Var
        Type T.
        Concatenated tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Concat``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Concat(
        _Concat.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _Concat.Inputs(
            inputs=inputs,
        ),
    ).outputs.concat_result


def concat_from_sequence(
    input_sequence: Var,
    *,
    axis: int,
    new_axis: int = 0,
) -> Var:
    r"""
    Concatenate a sequence of tensors into a single tensor. All input
    tensors must have the same shape, except for the dimension size of the
    axis to concatenate on. By default 'new_axis' is 0, the behavior is
    similar to numpy.concatenate. When 'new_axis' is 1, the behavior is
    similar to numpy.stack.

    Parameters
    ==========
    input_sequence
        Type S.
        Sequence of tensors for concatenation
    axis
        Attribute.
        Which axis to concat on. Accepted range in ``[-r, r - 1]``, where ``r``
        is the rank of input tensors. When ``new_axis`` is 1, accepted range is
        ``[-r - 1, r]``.
    new_axis
        Attribute.
        Insert and concatenate on a new axis or not, default 0 means do not
        insert new axis.

    Returns
    =======
    concat_result : Var
        Type T.
        Concatenated tensor

    Notes
    =====
    Signature: ``ai.onnx@11::ConcatFromSequence``.

    Type constraints:
     - S: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`
     - T: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ConcatFromSequence(
        _ConcatFromSequence.Attributes(
            axis=AttrInt64(axis, name="axis"),
            new_axis=AttrInt64(new_axis, name="new_axis"),
        ),
        _ConcatFromSequence.Inputs(
            input_sequence=input_sequence,
        ),
    ).outputs.concat_result


def constant(
    *,
    value: Optional[np.ndarray] = None,
    value_float: Optional[float] = None,
    value_floats: Optional[Iterable[float]] = None,
    value_int: Optional[int] = None,
    value_ints: Optional[Iterable[int]] = None,
    value_string: Optional[str] = None,
    value_strings: Optional[Iterable[str]] = None,
) -> Var:
    r"""
    This operator produces a constant tensor. Exactly one of the provided
    attributes, either value, sparse_value, or value\_\* must be specified.

    Parameters
    ==========
    sparse_value
        Attribute.
        The value for the elements of the output tensor in sparse format.
    value
        Attribute.
        The value for the elements of the output tensor.
    value_float
        Attribute.
        The value for the sole element for the scalar, float32, output tensor.
    value_floats
        Attribute.
        The values for the elements for the 1D, float32, output tensor.
    value_int
        Attribute.
        The value for the sole element for the scalar, int64, output tensor.
    value_ints
        Attribute.
        The values for the elements for the 1D, int64, output tensor.
    value_string
        Attribute.
        The value for the sole element for the scalar, UTF-8 string, output
        tensor.
    value_strings
        Attribute.
        The values for the elements for the 1D, UTF-8 string, output tensor.

    Returns
    =======
    output : Var
        Type T.
        Output tensor containing the same value of the provided tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Constant``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Constant(
        _Constant.Attributes(
            value=AttrTensor.maybe(value, name="value"),
            value_float=AttrFloat32.maybe(value_float, name="value_float"),
            value_floats=AttrFloat32s.maybe(value_floats, name="value_floats"),
            value_int=AttrInt64.maybe(value_int, name="value_int"),
            value_ints=AttrInt64s.maybe(value_ints, name="value_ints"),
            value_string=AttrString.maybe(value_string, name="value_string"),
            value_strings=AttrStrings.maybe(value_strings, name="value_strings"),
        ),
        _Constant.Inputs(),
    ).outputs.output


def constant_of_shape(
    input: Var,
    *,
    value: Optional[np.ndarray] = None,
) -> Var:
    r"""
    Generate a tensor with given value and shape.

    Parameters
    ==========
    input
        Type T1.
        1D tensor. The shape of the expected output tensor. If empty tensor is
        given, the output would be a scalar. All values must be >= 0.
    value
        Attribute.
        (Optional) The value of the output elements.Should be a one-element
        tensor. If not specified, it defaults to a tensor of value 0 and
        datatype float32

    Returns
    =======
    output : Var
        Type T2.
        Output tensor of shape specified by 'input'.If attribute 'value' is
        specified, the value and datatype of the output tensor is taken from
        'value'.If attribute 'value' is not specified, the value in the output
        defaults to 0, and the datatype defaults to float32.

    Notes
    =====
    Signature: ``ai.onnx@9::ConstantOfShape``.

    Type constraints:
     - T1: `tensor(int64)`
     - T2: `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ConstantOfShape(
        _ConstantOfShape.Attributes(
            value=AttrTensor.maybe(value, name="value"),
        ),
        _ConstantOfShape.Inputs(
            input=input,
        ),
    ).outputs.output


def conv(
    X: Var,
    W: Var,
    B: Optional[Var] = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: Optional[Iterable[int]] = None,
    group: int = 1,
    kernel_shape: Optional[Iterable[int]] = None,
    pads: Optional[Iterable[int]] = None,
    strides: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    The convolution operator consumes an input tensor and a filter, and
    computes the output.

    Parameters
    ==========
    X
        Type T.
        Input data tensor from previous layer; has size (N x C x H x W), where N
        is the batch size, C is the number of channels, and H and W are the
        height and width. Note that this is for the 2D image. Otherwise the size
        is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in
        effect, the operation expects input data tensor to arrive with the
        dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE,
        DATA_FEATURE ...].
    W
        Type T.
        The weight tensor that will be used in the convolutions; has size (M x
        C/group x kH x kW), where C is the number of channels, and kH and kW are
        the height and width of the kernel, and M is the number of feature maps.
        For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x
        k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel.
        Optionally, if dimension denotation is in effect, the operation expects
        the weight tensor to arrive with the dimension denotation of
        [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL
        ...]. Assuming zero based indices for the shape array, X.shape[1] ==
        (W.shape[1] \* group) == C and W.shape[0] mod G == 0. Or in other words
        FILTER_IN_CHANNEL multiplied by the number of groups should be equal to
        DATA_CHANNEL and the number of feature maps M should be a multiple of
        the number of groups G.
    B
        Type T.
        Optional 1D bias to be added to the convolution, has size of M.
    auto_pad
        Attribute.
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where
        default value is NOTSET, which means explicit padding is used.
        SAME_UPPER or SAME_LOWER mean pad the input so that
        ``output_shape[i] = ceil(input_shape[i] / strides[i])`` for each axis
        ``i``. The padding is split between the two sides equally or almost
        equally (depending on whether it is even or odd). In case the padding is
        an odd number, the extra padding is added at the end for SAME_UPPER and
        at the beginning for SAME_LOWER.
    dilations
        Attribute.
        dilation value along each spatial axis of the filter. If not present,
        the dilation defaults is 1 along each spatial axis.
    group
        Attribute.
        number of groups input channels and output channels are divided into.
    kernel_shape
        Attribute.
        The shape of the convolution kernel. If not present, should be inferred
        from input W.
    pads
        Attribute.
        Padding for the beginning and ending along each spatial axis, it can
        take any value greater than or equal to 0. The value represent the
        number of pixels added to the beginning and end part of the
        corresponding axis. ``pads`` format should be as follow [x1_begin,
        x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
        added at the beginning of axis ``i`` and xi_end, the number of pixels
        added at the end of axis ``i``. This attribute cannot be used
        simultaneously with auto_pad attribute. If not present, the padding
        defaults to 0 along start and end of each spatial axis.
    strides
        Attribute.
        Stride along each spatial axis. If not present, the stride defaults is 1
        along each spatial axis.

    Returns
    =======
    Y : Var
        Type T.
        Output data tensor that contains the result of the convolution. The
        output dimensions are functions of the kernel size, stride size, and pad
        lengths.

    Notes
    =====
    Signature: ``ai.onnx@11::Conv``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Conv(
        _Conv.Attributes(
            auto_pad=AttrString(auto_pad, name="auto_pad"),
            dilations=AttrInt64s.maybe(dilations, name="dilations"),
            group=AttrInt64(group, name="group"),
            kernel_shape=AttrInt64s.maybe(kernel_shape, name="kernel_shape"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _Conv.Inputs(
            X=X,
            W=W,
            B=B,
        ),
    ).outputs.Y


def conv_integer(
    x: Var,
    w: Var,
    x_zero_point: Optional[Var] = None,
    w_zero_point: Optional[Var] = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: Optional[Iterable[int]] = None,
    group: int = 1,
    kernel_shape: Optional[Iterable[int]] = None,
    pads: Optional[Iterable[int]] = None,
    strides: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    The integer convolution operator consumes an input tensor, its
    zero-point, a filter, and its zero-point, and computes the output. The
    production MUST never overflow. The accumulation may overflow if and
    only if in 32 bits.

    Parameters
    ==========
    x
        Type T1.
        Input data tensor from previous layer; has size (N x C x H x W), where N
        is the batch size, C is the number of channels, and H and W are the
        height and width. Note that this is for the 2D image. Otherwise the size
        is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in
        effect, the operation expects input data tensor to arrive with the
        dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE,
        DATA_FEATURE ...].
    w
        Type T2.
        The weight tensor that will be used in the convolutions; has size (M x
        C/group x kH x kW), where C is the number of channels, and kH and kW are
        the height and width of the kernel, and M is the number of feature maps.
        For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x
        k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel.
        Optionally, if dimension denotation is in effect, the operation expects
        the weight tensor to arrive with the dimension denotation of
        [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL
        ...]. X.shape[1] == (W.shape[1] \* group) == C (assuming zero based
        indices for the shape array). Or in other words FILTER_IN_CHANNEL should
        be equal to DATA_CHANNEL.
    x_zero_point
        Type T1.
        Zero point tensor for input 'x'. It's optional and default value is 0.
        It's a scalar, which means a per-tensor/layer quantization.
    w_zero_point
        Type T2.
        Zero point tensor for input 'w'. It's optional and default value is 0.
        It could be a scalar or a 1-D tensor, which means a per-tensor/layer or
        per output channel quantization. If it's a 1-D tensor, its number of
        elements should be equal to the number of output channels (M)
    auto_pad
        Attribute.
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where
        default value is NOTSET, which means explicit padding is used.
        SAME_UPPER or SAME_LOWER mean pad the input so that
        ``output_shape[i] = ceil(input_shape[i] / strides[i])`` for each axis
        ``i``. The padding is split between the two sides equally or almost
        equally (depending on whether it is even or odd). In case the padding is
        an odd number, the extra padding is added at the end for SAME_UPPER and
        at the beginning for SAME_LOWER.
    dilations
        Attribute.
        dilation value along each spatial axis of the filter. If not present,
        the dilation defaults to 1 along each axis.
    group
        Attribute.
        number of groups input channels and output channels are divided into.
        default is 1.
    kernel_shape
        Attribute.
        The shape of the convolution kernel. If not present, should be inferred
        from input 'w'.
    pads
        Attribute.
        Padding for the beginning and ending along each spatial axis, it can
        take any value greater than or equal to 0.The value represent the number
        of pixels added to the beginning and end part of the corresponding
        axis.\ ``pads`` format should be as follow [x1_begin, x2_begin...x1_end,
        x2_end,...], where xi_begin the number ofpixels added at the beginning
        of axis ``i`` and xi_end, the number of pixels added at the end of axis
        ``i``.This attribute cannot be used simultaneously with auto_pad
        attribute. If not present, the padding defaultsto 0 along start and end
        of each spatial axis.
    strides
        Attribute.
        Stride along each spatial axis. If not present, the stride defaults to 1
        along each axis.

    Returns
    =======
    y : Var
        Type T3.
        Output data tensor that contains the result of the convolution. The
        output dimensions are functions of the kernel size, stride size, and pad
        lengths.

    Notes
    =====
    Signature: ``ai.onnx@10::ConvInteger``.

    Type constraints:
     - T1: `tensor(int8)`, `tensor(uint8)`
     - T2: `tensor(int8)`, `tensor(uint8)`
     - T3: `tensor(int32)`
    """
    return _ConvInteger(
        _ConvInteger.Attributes(
            auto_pad=AttrString(auto_pad, name="auto_pad"),
            dilations=AttrInt64s.maybe(dilations, name="dilations"),
            group=AttrInt64(group, name="group"),
            kernel_shape=AttrInt64s.maybe(kernel_shape, name="kernel_shape"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _ConvInteger.Inputs(
            x=x,
            w=w,
            x_zero_point=x_zero_point,
            w_zero_point=w_zero_point,
        ),
    ).outputs.y


def conv_transpose(
    X: Var,
    W: Var,
    B: Optional[Var] = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: Optional[Iterable[int]] = None,
    group: int = 1,
    kernel_shape: Optional[Iterable[int]] = None,
    output_padding: Optional[Iterable[int]] = None,
    output_shape: Optional[Iterable[int]] = None,
    pads: Optional[Iterable[int]] = None,
    strides: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    The convolution transpose operator consumes an input tensor and a
    filter, and computes the output.

    If the pads parameter is provided the shape of the output is calculated
    via the following equation:

    output_shape[i] = stride[i] \* (input_size[i] - 1) + output_padding[i] +
    ((kernel_shape[i] - 1) \* dilations[i] + 1) - pads[start_i] -
    pads[end_i]

    output_shape can also be explicitly specified in which case pads values
    are auto generated using these equations:

    total_padding[i] = stride[i] \* (input_size[i] - 1) + output_padding[i]
    + ((kernel_shape[i] - 1) \* dilations[i] + 1) - output_shape[i] If
    (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2;
    pads[end_i] = total_padding[i] - (total_padding[i]/2) Else:
    pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] =
    (total_padding[i]/2).

    Parameters
    ==========
    X
        Type T.
        Input data tensor from previous layer; has size (N x C x H x W), where N
        is the batch size, C is the number of channels, and H and W are the
        height and width. Note that this is for the 2D image. Otherwise the size
        is (N x C x D1 x D2 ... x Dn)
    W
        Type T.
        The weight tensor that will be used in the convolutions; has size (C x
        M/group x kH x kW), where C is the number of channels, and kH and kW are
        the height and width of the kernel, and M is the number of feature maps.
        For more than 2 dimensions, the weight shape will be (C x M/group x k1 x
        k2 x ... x kn), where (k1 x k2 x ... x kn) is the dimension of the
        kernel. The number of channels in the output should be equal to
        W.shape[1] \* group (assuming zero based indices of the shape array)
    B
        Type T.
        Optional 1D bias to be added to the convolution, has size of M.
    auto_pad
        Attribute.
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where
        default value is NOTSET, which means explicit padding is used.
        SAME_UPPER or SAME_LOWER mean pad the input so that
        ``output_shape[i] = input_shape[i] * strides[i]`` for each axis ``i``.
        The padding is split between the two sides equally or almost equally
        (depending on whether it is even or odd). In case the padding is an odd
        number, the extra padding is added at the end for SAME_UPPER and at the
        beginning for SAME_LOWER.
    dilations
        Attribute.
        dilation value along each spatial axis of the filter. If not present,
        the dilation defaults to 1 along each spatial axis.
    group
        Attribute.
        number of groups input channels and output channels are divided into.
    kernel_shape
        Attribute.
        The shape of the convolution kernel. If not present, should be inferred
        from input W.
    output_padding
        Attribute.
        Additional elements added to the side with higher coordinate indices in
        the output. Each padding value in "output_padding" must be less than the
        corresponding stride/dilation dimension. By default, this attribute is a
        zero vector. Note that this attribute doesn't directly affect the
        computed output values. It only controls the selection of the computed
        values, so changing this attribute only adds or removes output elements.
        If "output_shape" is explicitly provided, "output_padding" does not
        contribute additional size to "output_shape" but participates in the
        computation of the needed padding amount. This is also called adjs or
        adjustment in some frameworks.
    output_shape
        Attribute.
        The shape of the output can be explicitly set which will cause pads
        values to be auto generated. If output_shape is specified pads values
        are ignored. See doc for details for equations to generate pads. Note
        that the output_shape attribute value should not include dimensions for
        batch size and channels, which are automatically inferred.
    pads
        Attribute.
        Padding for the beginning and ending along each spatial axis, it can
        take any value greater than or equal to 0. The value represent the
        number of pixels added to the beginning and end part of the
        corresponding axis. ``pads`` format should be as follow [x1_begin,
        x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
        added at the beginning of axis ``i`` and xi_end, the number of pixels
        added at the end of axis ``i``. This attribute cannot be used
        simultaneously with auto_pad attribute. If not present, the padding
        defaults to 0 along start and end of each spatial axis.
    strides
        Attribute.
        Stride along each spatial axis. If not present, the stride defaults to 1
        along each spatial axis.

    Returns
    =======
    Y : Var
        Type T.
        Output data tensor that contains the result of the convolution. The
        output dimensions are functions of the kernel size, stride size, pad
        lengths and group count. The number of channels in the output should be
        equal to W.shape[1] \* group (assuming zero based indices of the shape
        array)

    Notes
    =====
    Signature: ``ai.onnx@11::ConvTranspose``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _ConvTranspose(
        _ConvTranspose.Attributes(
            auto_pad=AttrString(auto_pad, name="auto_pad"),
            dilations=AttrInt64s.maybe(dilations, name="dilations"),
            group=AttrInt64(group, name="group"),
            kernel_shape=AttrInt64s.maybe(kernel_shape, name="kernel_shape"),
            output_padding=AttrInt64s.maybe(output_padding, name="output_padding"),
            output_shape=AttrInt64s.maybe(output_shape, name="output_shape"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _ConvTranspose.Inputs(
            X=X,
            W=W,
            B=B,
        ),
    ).outputs.Y


def cos(
    input: Var,
) -> Var:
    r"""
    Calculates the cosine of the given input tensor, element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The cosine of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@7::Cos``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Cos(
        _Cos.Attributes(),
        _Cos.Inputs(
            input=input,
        ),
    ).outputs.output


def cosh(
    input: Var,
) -> Var:
    r"""
    Calculates the hyperbolic cosine of the given input tensor element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The hyperbolic cosine values of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@9::Cosh``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Cosh(
        _Cosh.Attributes(),
        _Cosh.Inputs(
            input=input,
        ),
    ).outputs.output


def cumsum(
    x: Var,
    axis: Var,
    *,
    exclusive: int = 0,
    reverse: int = 0,
) -> Var:
    r"""
    Performs cumulative sum of the input elements along the given axis. By
    default, it will do the sum inclusively meaning the first element is
    copied as is. Through an ``exclusive`` attribute, this behavior can
    change to exclude the first element. It can also perform summation in
    the opposite direction of the axis. For that, set ``reverse`` attribute
    to 1.

    Example:

    ::

       input_x = [1, 2, 3]
       axis=0
       output = [1, 3, 6]
       exclusive=1
       output = [0, 1, 3]
       exclusive=0
       reverse=1
       output = [6, 5, 3]
       exclusive=1
       reverse=1
       output = [5, 3, 0]

    Parameters
    ==========
    x
        Type T.
        An input tensor that is to be processed.
    axis
        Type T2.
        A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative value
        means counting dimensions from the back.
    exclusive
        Attribute.
        If set to 1 will return exclusive sum in which the top element is not
        included. In other terms, if set to 1, the j-th output element would be
        the sum of the first (j-1) elements. Otherwise, it would be the sum of
        the first j elements.
    reverse
        Attribute.
        If set to 1 will perform the sums in reverse direction.

    Returns
    =======
    y : Var
        Type T.
        Output tensor of the same type as 'x' with cumulative sums of the x's
        elements

    Notes
    =====
    Signature: ``ai.onnx@14::CumSum``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
     - T2: `tensor(int32)`, `tensor(int64)`
    """
    return _CumSum(
        _CumSum.Attributes(
            exclusive=AttrInt64(exclusive, name="exclusive"),
            reverse=AttrInt64(reverse, name="reverse"),
        ),
        _CumSum.Inputs(
            x=x,
            axis=axis,
        ),
    ).outputs.y


def dft(
    input: Var,
    dft_length: Optional[Var] = None,
    *,
    axis: int = 1,
    inverse: int = 0,
    onesided: int = 0,
) -> Var:
    r"""
    Computes the discrete Fourier transform of input.

    Parameters
    ==========
    input
        Type T1.
        For real input, the following shape is expected:
        [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1]. For complex
        input, the following shape is expected:
        [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]. The first
        dimension is the batch dimension. The following N dimensions correspond
        to the signal's dimensions. The final dimension represents the real and
        imaginary parts of the value in that order.
    dft_length
        Type T2.
        The length of the signal as a scalar. If greater than the axis
        dimension, the signal will be zero-padded up to dft_length. If less than
        the axis dimension, only the first dft_length values will be used as the
        signal. It's an optional value.
    axis
        Attribute.
        The axis on which to perform the DFT. By default this value is set to 1,
        which corresponds to the first dimension after the batch index. Negative
        value means counting dimensions from the back. Accepted range is
        :math:`[-r, -2] \cup [0, r-2]` where ``r = rank(input)``. The last
        dimension is for representing complex numbers and thus is an invalid
        axis.
    inverse
        Attribute.
        Whether to perform the inverse discrete fourier transform. By default
        this value is set to 0, which corresponds to false.
    onesided
        Attribute.
        If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) +
        1] are returned because the real-to-complex Fourier transform satisfies
        the conjugate symmetry, i.e., X[m, w] = X[m, n_fft-w]\*. Note if the
        input or window tensors are complex, then onesided output is not
        possible. Enabling onesided with real inputs performs a Real-valued fast
        Fourier transform (RFFT). When invoked with real or complex valued
        input, the default value is 0. Values can be 0 or 1.

    Returns
    =======
    output : Var
        Type T1.
        The Fourier Transform of the input vector. If onesided is 0, the
        following shape is expected:
        [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]. If axis=1 and
        onesided is 1, the following shape is expected:
        [batch_idx][floor(signal_dim1/2)+1][signal_dim2]...[signal_dimN][2]. If
        axis=2 and onesided is 1, the following shape is expected:
        [batch_idx][signal_dim1][floor(signal_dim2/2)+1]...[signal_dimN][2]. If
        axis=N and onesided is 1, the following shape is expected:
        [batch_idx][signal_dim1][signal_dim2]...[floor(signal_dimN/2)+1][2]. The
        signal_dim at the specified axis is equal to the dft_length.

    Notes
    =====
    Signature: ``ai.onnx@17::DFT``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(int32)`, `tensor(int64)`
    """
    return _DFT(
        _DFT.Attributes(
            axis=AttrInt64(axis, name="axis"),
            inverse=AttrInt64(inverse, name="inverse"),
            onesided=AttrInt64(onesided, name="onesided"),
        ),
        _DFT.Inputs(
            input=input,
            dft_length=dft_length,
        ),
    ).outputs.output


def depth_to_space(
    input: Var,
    *,
    blocksize: int,
    mode: str = "DCR",
) -> Var:
    r"""
    DepthToSpace rearranges (permutes) data from depth into blocks of
    spatial data. This is the reverse transformation of SpaceToDepth. More
    specifically, this op outputs a copy of the input tensor where values
    from the depth dimension are moved in spatial blocks to the height and
    width dimensions. By default, ``mode`` = ``DCR``. In the DCR mode,
    elements along the depth dimension from the input tensor are rearranged
    in the following order: depth, column, and then row. The output y is
    computed from the input x as below:

    ::

       b, c, h, w = x.shape
       tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
       tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
       y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])

    In the CRD mode, elements along the depth dimension from the input
    tensor are rearranged in the following order: column, row, and the
    depth. The output y is computed from the input x as below:

    ::

       b, c, h, w = x.shape
       tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
       tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
       y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])

    Parameters
    ==========
    input
        Type T.
        Input tensor of [N,C,H,W], where N is the batch axis, C is the channel
        or depth, H is the height and W is the width.
    blocksize
        Attribute.
        Blocks of [blocksize, blocksize] are moved.
    mode
        Attribute.
        DCR (default) for depth-column-row order re-arrangement. Use CRD for
        column-row-depth order.

    Returns
    =======
    output : Var
        Type T.
        Output tensor of [N, C/(blocksize \* blocksize), H \* blocksize, W \*
        blocksize].

    Notes
    =====
    Signature: ``ai.onnx@13::DepthToSpace``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _DepthToSpace(
        _DepthToSpace.Attributes(
            blocksize=AttrInt64(blocksize, name="blocksize"),
            mode=AttrString(mode, name="mode"),
        ),
        _DepthToSpace.Inputs(
            input=input,
        ),
    ).outputs.output


def dequantize_linear(
    x: Var,
    x_scale: Var,
    x_zero_point: Optional[Var] = None,
    *,
    axis: int = 1,
) -> Var:
    r"""
    The linear dequantization operator. It consumes a quantized tensor, a
    scale, and a zero point to compute the full precision tensor. The
    dequantization formula is ``y = (x - x_zero_point) * x_scale``.
    ``x_scale`` and ``x_zero_point`` must have same shape, and can be either
    a scalar for per-tensor / per layer quantization, or a 1-D tensor for
    per-axis quantization. ``x_zero_point`` and ``x`` must have same type.
    ``x`` and ``y`` must have same shape. In the case of dequantizing int32,
    there's no zero point (zero point is supposed to be 0).

    Parameters
    ==========
    x
        Type T.
        N-D quantized input tensor to be de-quantized.
    x_scale
        Type tensor(float).
        Scale for input 'x'. It can be a scalar, which means a per-tensor/layer
        dequantization, or a 1-D tensor for per-axis dequantization.
    x_zero_point
        Type T.
        Zero point for input 'x'. Shape must match x_scale. It's optional. Zero
        point is 0 when it's not specified.
    axis
        Attribute.
        (Optional) The axis of the dequantizing dimension of the input tensor.
        Ignored for per-tensor quantization. Negative value means counting
        dimensions from the back. Accepted range is [-r, r-1] where r =
        rank(input).

    Returns
    =======
    y : Var
        Type tensor(float).
        N-D full precision output tensor. It has same shape as input 'x'.

    Notes
    =====
    Signature: ``ai.onnx@13::DequantizeLinear``.

    Type constraints:
     - T: `tensor(int32)`, `tensor(int8)`, `tensor(uint8)`
    """
    return _DequantizeLinear(
        _DequantizeLinear.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _DequantizeLinear.Inputs(
            x=x,
            x_scale=x_scale,
            x_zero_point=x_zero_point,
        ),
    ).outputs.y


def det(
    X: Var,
) -> Var:
    r"""
    Det calculates determinant of a square matrix or batches of square
    matrices. Det takes one input tensor of shape ``[*, M, M]``, where ``*``
    is zero or more batch dimensions, and the inner-most 2 dimensions form
    square matrices. The output is a tensor of shape ``[*]``, containing the
    determinants of all input submatrices. e.g., When the input is 2-D, the
    output is a scalar(shape is empty: ``[]``).

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@11::Det``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Det(
        _Det.Attributes(),
        _Det.Inputs(
            X=X,
        ),
    ).outputs.Y


def div(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Performs element-wise binary division (with Numpy-style broadcasting
    support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    (Opset 14 change): Extend supported types to include uint8, int8,
    uint16, and int16.

    Parameters
    ==========
    A
        Type T.
        First operand.
    B
        Type T.
        Second operand.

    Returns
    =======
    C : Var
        Type T.
        Result, has same element type as two inputs

    Notes
    =====
    Signature: ``ai.onnx@14::Div``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Div(
        _Div.Attributes(),
        _Div.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def dropout(
    data: Var,
    ratio: Optional[Var] = None,
    training_mode: Optional[Var] = None,
    *,
    seed: Optional[int] = None,
) -> Tuple[Var, Var]:
    r"""
    Dropout takes an input floating-point tensor, an optional input ratio
    (floating-point scalar) and an optional input training_mode (boolean
    scalar). It produces two tensor outputs, output (floating-point tensor)
    and mask (optional ``Tensor<bool>``). If ``training_mode`` is true then
    the output Y will be a random dropout; Note that this Dropout scales the
    masked input data by the following equation, so to convert the trained
    model into inference mode, the user can simply not pass
    ``training_mode`` input or set it to false.

    ::

       output = scale * data * mask,

    where

    ::

       scale = 1. / (1. - ratio).

    This operator has **optional** inputs/outputs. See `the
    doc <https://github.com/onnx/onnx/blob/main/docs/IR.md>`__ for more
    details about the representation of optional arguments. An empty string
    may be used in the place of an actual argument's name to indicate a
    missing argument. Trailing optional arguments (those not followed by an
    argument that is present) may also be simply omitted.

    Parameters
    ==========
    data
        Type T.
        The input data as Tensor.
    ratio
        Type T1.
        The ratio of random dropout, with value in [0, 1). If this input was not
        set, or if it was set to 0, the output would be a simple copy of the
        input. If it's non-zero, output will be a random dropout of the scaled
        input, which is typically the case during training. It is an optional
        value, if not specified it will default to 0.5.
    training_mode
        Type T2.
        If set to true then it indicates dropout is being used for training. It
        is an optional value hence unless specified explicitly, it is false. If
        it is false, ratio is ignored and the operation mimics inference mode
        where nothing will be dropped from the input data and if mask is
        requested as output it will contain all ones.
    seed
        Attribute.
        (Optional) Seed to the random generator, if not specified we will auto
        generate one.

    Returns
    =======
    output : Var
        Type T.
        The output.
    mask : Var
        Type T2.
        The output mask.

    Notes
    =====
    Signature: ``ai.onnx@13::Dropout``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T1: `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(bool)`
    """
    return _Dropout(
        _Dropout.Attributes(
            seed=AttrInt64.maybe(seed, name="seed"),
        ),
        _Dropout.Inputs(
            data=data,
            ratio=ratio,
            training_mode=training_mode,
        ),
    ).outputs._unpack_to_any()


def dynamic_quantize_linear(
    x: Var,
) -> Tuple[Var, Var, Var]:
    r"""
    A Function to fuse calculation for Scale, Zero Point and FP32->8Bit
    conversion of FP32 Input data. Outputs Scale, ZeroPoint and Quantized
    Input for a given FP32 Input. Scale is calculated as:

    ::

       y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)

    -  where qmax and qmin are max and min values for quantization range
       i.e. [0, 255] in case of uint8
    -  data range is adjusted to include 0.

    Zero point is calculated as:

    ::

       intermediate_zero_point = qmin - min(x)/y_scale
       y_zero_point = cast(round(saturate(itermediate_zero_point)))

    -  where qmax and qmin are max and min values for quantization range
       .i.e [0, 255] in case of uint8
    -  for saturation, it saturates to [0, 255] if it's uint8, or [-127,
       127] if it's int8. Right now only uint8 is supported.
    -  rounding to nearest ties to even.

    Data quantization formula is:

    ::

       y = saturate (round (x / y_scale) + y_zero_point)

    -  for saturation, it saturates to [0, 255] if it's uint8, or [-127,
       127] if it's int8. Right now only uint8 is supported.
    -  rounding to nearest ties to even.

    Parameters
    ==========
    x
        Type T1.
        Input tensor

    Returns
    =======
    y : Var
        Type T2.
        Quantized output tensor
    y_scale : Var
        Type tensor(float).
        Output scale. It's a scalar, which means a per-tensor/layer
        quantization.
    y_zero_point : Var
        Type T2.
        Output zero point. It's a scalar, which means a per-tensor/layer
        quantization.

    Notes
    =====
    Signature: ``ai.onnx@11::DynamicQuantizeLinear``.

    Type constraints:
     - T1: `tensor(float)`
     - T2: `tensor(uint8)`
    """
    return _DynamicQuantizeLinear(
        _DynamicQuantizeLinear.Attributes(),
        _DynamicQuantizeLinear.Inputs(
            x=x,
        ),
    ).outputs._unpack_to_any()


def einsum(
    Inputs: Sequence[Var],
    *,
    equation: str,
) -> Var:
    r"""
    An einsum of the form ``term1, term2 -> output-term`` produces an output
    tensor using the following equation

    ::

       output[output-term] = reduce-sum( input1[term1] * input2[term2] )

    where the reduce-sum performs a summation over all the indices occurring
    in the input terms (term1, term2) that do not occur in the output-term.

    The Einsum operator evaluates algebraic tensor operations on a sequence
    of tensors, using the Einstein summation convention. The equation string
    contains a comma-separated sequence of lower case letters. Each term
    corresponds to an operand tensor, and the characters within the terms
    correspond to operands dimensions.

    This sequence may be followed by "->" to separate the left and right
    hand side of the equation. If the equation contains "->" followed by the
    right-hand side, the explicit (not classical) form of the Einstein
    summation is performed, and the right-hand side indices indicate output
    tensor dimensions. In other cases, output indices are (implicitly) set
    to the alphabetically sorted sequence of indices appearing exactly once
    in the equation.

    When a dimension character is repeated in the left-hand side, it
    represents summation along the dimension.

    The equation may contain ellipsis ("...") to enable broadcasting.
    Ellipsis must indicate a fixed number of dimensions. Specifically, every
    occurrence of ellipsis in the equation must represent the same number of
    dimensions. The right-hand side may contain exactly one ellipsis. In
    implicit mode, the ellipsis dimensions are set to the beginning of the
    output. The equation string may contain space (U+0020) character.

    Parameters
    ==========
    Inputs
        Type T.
        Operands
    equation
        Attribute.
        Einsum expression string.

    Returns
    =======
    Output : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@12::Einsum``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Einsum(
        _Einsum.Attributes(
            equation=AttrString(equation, name="equation"),
        ),
        _Einsum.Inputs(
            Inputs=Inputs,
        ),
    ).outputs.Output


def elu(
    X: Var,
    *,
    alpha: float = 1.0,
) -> Var:
    r"""
    Elu takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the function
    ``f(x) = alpha * (exp(x) - 1.) for x < 0``, ``f(x) = x for x >= 0``., is
    applied to the tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        1D input tensor
    alpha
        Attribute.
        Coefficient of ELU.

    Returns
    =======
    Y : Var
        Type T.
        1D output tensor

    Notes
    =====
    Signature: ``ai.onnx@6::Elu``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Elu(
        _Elu.Attributes(
            alpha=AttrFloat32(alpha, name="alpha"),
        ),
        _Elu.Inputs(
            X=X,
        ),
    ).outputs.Y


def equal(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulted from performing the ``equal`` logical
    operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the logical operator.
    B
        Type T.
        Second input operand for the logical operator.

    Returns
    =======
    C : Var
        Type T1.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Equal``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(bool)`
    """
    return _Equal(
        _Equal.Attributes(),
        _Equal.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def erf(
    input: Var,
) -> Var:
    r"""
    Computes the error function of the given input tensor element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The error function of the input tensor computed element-wise. It has the
        same shape and type of the input.

    Notes
    =====
    Signature: ``ai.onnx@13::Erf``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Erf(
        _Erf.Attributes(),
        _Erf.Inputs(
            input=input,
        ),
    ).outputs.output


def exp(
    input: Var,
) -> Var:
    r"""
    Calculates the exponential of the given input tensor, element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The exponential of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@13::Exp``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Exp(
        _Exp.Attributes(),
        _Exp.Inputs(
            input=input,
        ),
    ).outputs.output


def expand(
    input: Var,
    shape: Var,
) -> Var:
    r"""
    Broadcast the input tensor following the given shape and the broadcast
    rule. The broadcast rule is similar to numpy.array(input) \*
    numpy.ones(shape): Dimensions are right alignment; Two corresponding
    dimensions must have the same value, or one of them is equal to 1. Also,
    this operator is similar to numpy.broadcast_to(input, shape), but the
    major difference is numpy.broadcast_to() does not allow shape to be
    smaller than input.size(). It is possible that the output.shape is not
    equal to shape, when some dimensions in shape is equal to 1, or the
    shape.ndim < input.shape.ndim.

    Parameters
    ==========
    input
        Type T.
        Input tensor
    shape
        Type tensor(int64).
        A 1-D tensor indicates the shape you want to expand to, following the
        broadcast rule

    Returns
    =======
    output : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Expand``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Expand(
        _Expand.Attributes(),
        _Expand.Inputs(
            input=input,
            shape=shape,
        ),
    ).outputs.output


def eye_like(
    input: Var,
    *,
    dtype: Optional[npt.DTypeLike] = None,
    k: int = 0,
) -> Var:
    r"""
    Generate a 2D tensor (matrix) with ones on the diagonal and zeros
    everywhere else. Only 2D tensors are supported, i.e. input T1 must be of
    rank 2. The shape of the output tensor is the same as the input tensor.
    The data type can be specified by the 'dtype' argument. If 'dtype' is
    not specified, then the type of input tensor is used. By default, the
    main diagonal is populated with ones, but attribute 'k' can be used to
    populate upper or lower diagonals. The 'dtype' argument must be one of
    the data types specified in the 'DataType' enum field in the TensorProto
    message and be valid as an output type.

    Parameters
    ==========
    input
        Type T1.
        2D input tensor to copy shape, and optionally, type information from.
    dtype
        Attribute.
        (Optional) The data type for the elements of the output tensor. If not
        specified,the data type of the input tensor T1 is used. If input tensor
        T1 is also notspecified, then type defaults to 'float'.
    k
        Attribute.
        (Optional) Index of the diagonal to be populated with ones. Default is
        0. If T2 is the output, this op sets T2[i, i+k] = 1. k = 0 populates the
        main diagonal, k > 0 populates an upper diagonal, and k < 0 populates a
        lower diagonal.

    Returns
    =======
    output : Var
        Type T2.
        Output tensor, same shape as input tensor T1.

    Notes
    =====
    Signature: ``ai.onnx@9::EyeLike``.

    Type constraints:
     - T1: `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _EyeLike(
        _EyeLike.Attributes(
            dtype=AttrDtype.maybe(dtype, name="dtype"),
            k=AttrInt64(k, name="k"),
        ),
        _EyeLike.Inputs(
            input=input,
        ),
    ).outputs.output


def flatten(
    input: Var,
    *,
    axis: int = 1,
) -> Var:
    r"""
    Flattens the input tensor into a 2D matrix. If input tensor has shape
    (d_0, d_1, ... d_n) then the output will have shape (d_0 X d_1 ...
    d\_(axis-1), d_axis X d\_(axis+1) ... X dn).

    Parameters
    ==========
    input
        Type T.
        A tensor of rank >= axis.
    axis
        Attribute.
        Indicate up to which input dimensions (exclusive) should be flattened to
        the outer dimension of the output. The value for axis must be in the
        range [-r, r], where r is the rank of the input tensor. Negative value
        means counting dimensions from the back. When axis = 0, the shape of the
        output tensor is (1, (d_0 X d_1 ... d_n), where the shape of the input
        tensor is (d_0, d_1, ... d_n).

    Returns
    =======
    output : Var
        Type T.
        A 2D tensor with the contents of the input tensor, with input dimensions
        up to axis flattened to the outer dimension of the output and remaining
        input dimensions flattened into the inner dimension of the output.

    Notes
    =====
    Signature: ``ai.onnx@13::Flatten``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Flatten(
        _Flatten.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _Flatten.Inputs(
            input=input,
        ),
    ).outputs.output


def floor(
    X: Var,
) -> Var:
    r"""
    Floor takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the floor is, y = floor(x), is applied to the tensor
    elementwise. If x is integral, +0, -0, NaN, or infinite, x itself is
    returned.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Floor``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Floor(
        _Floor.Attributes(),
        _Floor.Inputs(
            X=X,
        ),
    ).outputs.Y


def gru(
    X: Var,
    W: Var,
    R: Var,
    B: Optional[Var] = None,
    sequence_lens: Optional[Var] = None,
    initial_h: Optional[Var] = None,
    *,
    activation_alpha: Optional[Iterable[float]] = None,
    activation_beta: Optional[Iterable[float]] = None,
    activations: Optional[Iterable[str]] = None,
    clip: Optional[float] = None,
    direction: str = "forward",
    hidden_size: Optional[int] = None,
    layout: int = 0,
    linear_before_reset: int = 0,
) -> Tuple[Var, Var]:
    r"""
    Computes an one-layer GRU. This operator is usually supported via some
    custom implementation such as CuDNN.

    Notations:

    -  ``X`` - input tensor
    -  ``z`` - update gate
    -  ``r`` - reset gate
    -  ``h`` - hidden gate
    -  ``t`` - time step (t-1 means previous time step)
    -  ``W[zrh]`` - W parameter weight matrix for update, reset, and hidden
       gates
    -  ``R[zrh]`` - R recurrence weight matrix for update, reset, and hidden
       gates
    -  ``Wb[zrh]`` - W bias vectors for update, reset, and hidden gates
    -  ``Rb[zrh]`` - R bias vectors for update, reset, and hidden gates
    -  ``WB[zrh]`` - W parameter weight matrix for backward update, reset,
       and hidden gates
    -  ``RB[zrh]`` - R recurrence weight matrix for backward update, reset,
       and hidden gates
    -  ``WBb[zrh]`` - W bias vectors for backward update, reset, and hidden
       gates
    -  ``RBb[zrh]`` - R bias vectors for backward update, reset, and hidden
       gates
    -  ``H`` - Hidden state
    -  ``num_directions`` - 2 if direction == bidirectional else 1

    Activation functions:

    -  Relu(x) - max(0, x)
    -  Tanh(x) - (1 - e^{-2x})/(1 + e^{-2x})
    -  Sigmoid(x) - 1/(1 + e^{-x})

    NOTE: Below are optional

    -  Affine(x) - alpha \* x + beta
    -  LeakyRelu(x) - x if x >= 0 else alpha \* x
    -  ThresholdedRelu(x) - x if x >= alpha else 0
    -  ScaledTanh(x) - alpha \* Tanh(beta \* x)
    -  HardSigmoid(x) - min(max(alpha \* x + beta, 0), 1)
    -  Elu(x) - x if x >= 0 else alpha \* (e^x - 1)
    -  Softsign(x) - x/(1 + \|x\|)
    -  Softplus(x) - log(1 + e^x)

    Equations (Default: f=Sigmoid, g=Tanh):

    -  zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    -  rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    -  ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when
       linear_before_reset = 0
    -  ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when
       linear_before_reset != 0
    -  Ht = (1 - zt) (.) ht + zt (.) Ht-1 This operator has **optional**
       inputs/outputs. See `the
       doc <https://github.com/onnx/onnx/blob/main/docs/IR.md>`__ for more
       details about the representation of optional arguments. An empty
       string may be used in the place of an actual argument's name to
       indicate a missing argument. Trailing optional arguments (those not
       followed by an argument that is present) may also be simply omitted.

    Parameters
    ==========
    X
        Type T.
        The input sequences packed (and potentially padded) into one 3-D tensor
        with the shape of ``[seq_length, batch_size, input_size]``.
    W
        Type T.
        The weight tensor for the gates. Concatenation of ``W[zrh]`` and
        ``WB[zrh]`` (if bidirectional) along dimension 0. This tensor has shape
        ``[num_directions, 3*hidden_size, input_size]``.
    R
        Type T.
        The recurrence weight tensor. Concatenation of ``R[zrh]`` and
        ``RB[zrh]`` (if bidirectional) along dimension 0. This tensor has shape
        ``[num_directions, 3*hidden_size, hidden_size]``.
    B
        Type T.
        The bias tensor for the gates. Concatenation of ``[Wb[zrh], Rb[zrh]]``
        and ``[WBb[zrh], RBb[zrh]]`` (if bidirectional) along dimension 0. This
        tensor has shape ``[num_directions, 6*hidden_size]``. Optional: If not
        specified - assumed to be 0
    sequence_lens
        Type T1.
        Optional tensor specifying lengths of the sequences in a batch. If not
        specified - assumed all sequences in the batch to have length
        ``seq_length``. It has shape ``[batch_size]``.
    initial_h
        Type T.
        Optional initial value of the hidden. If not specified - assumed to be
        0. It has shape ``[num_directions, batch_size, hidden_size]``.
    activation_alpha
        Attribute.
        Optional scaling values used by some activation functions. The values
        are consumed in the order of activation functions, for example (f, g, h)
        in LSTM. Default values are the same as of corresponding ONNX
        operators.For example with LeakyRelu, the default alpha is 0.01.
    activation_beta
        Attribute.
        Optional scaling values used by some activation functions. The values
        are consumed in the order of activation functions, for example (f, g, h)
        in LSTM. Default values are the same as of corresponding ONNX operators.
    activations
        Attribute.
        A list of 2 (or 4 if bidirectional) activation functions for update,
        reset, and hidden gates. The activation functions must be one of the
        activation functions specified above. Optional: See the equations for
        default if not specified.
    clip
        Attribute.
        Cell clip threshold. Clipping bounds the elements of a tensor in the
        range of [-threshold, +threshold] and is applied to the input of
        activations. No clip if not specified.
    direction
        Attribute.
        Specify if the RNN is forward, reverse, or bidirectional. Must be one of
        forward (default), reverse, or bidirectional.
    hidden_size
        Attribute.
        Number of neurons in the hidden layer
    layout
        Attribute.
        The shape format of inputs X, initial_h and outputs Y, Y_h. If 0, the
        following shapes are expected: X.shape = [seq_length, batch_size,
        input_size], Y.shape = [seq_length, num_directions, batch_size,
        hidden_size], initial_h.shape = Y_h.shape = [num_directions, batch_size,
        hidden_size]. If 1, the following shapes are expected: X.shape =
        [batch_size, seq_length, input_size], Y.shape = [batch_size, seq_length,
        num_directions, hidden_size], initial_h.shape = Y_h.shape = [batch_size,
        num_directions, hidden_size].
    linear_before_reset
        Attribute.
        When computing the output of the hidden gate, apply the linear
        transformation before multiplying by the output of the reset gate.

    Returns
    =======
    Y : Var
        Type T.
        A tensor that concats all the intermediate output values of the hidden.
        It has shape ``[seq_length, num_directions, batch_size, hidden_size]``.
    Y_h : Var
        Type T.
        The last output value of the hidden. It has shape
        ``[num_directions, batch_size, hidden_size]``.

    Notes
    =====
    Signature: ``ai.onnx@14::GRU``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T1: `tensor(int32)`
    """
    return _GRU(
        _GRU.Attributes(
            activation_alpha=AttrFloat32s.maybe(
                activation_alpha, name="activation_alpha"
            ),
            activation_beta=AttrFloat32s.maybe(activation_beta, name="activation_beta"),
            activations=AttrStrings.maybe(activations, name="activations"),
            clip=AttrFloat32.maybe(clip, name="clip"),
            direction=AttrString(direction, name="direction"),
            hidden_size=AttrInt64.maybe(hidden_size, name="hidden_size"),
            layout=AttrInt64(layout, name="layout"),
            linear_before_reset=AttrInt64(
                linear_before_reset, name="linear_before_reset"
            ),
        ),
        _GRU.Inputs(
            X=X,
            W=W,
            R=R,
            B=B,
            sequence_lens=sequence_lens,
            initial_h=initial_h,
        ),
    ).outputs._unpack_to_any()


def gather(
    data: Var,
    indices: Var,
    *,
    axis: int = 0,
) -> Var:
    r"""
    Given ``data`` tensor of rank r >= 1, and ``indices`` tensor of rank q,
    gather entries of the axis dimension of ``data`` (by default outer-most
    one as axis=0) indexed by ``indices``, and concatenates them in an
    output tensor of rank q + (r - 1).

    If ``axis = 0``, let ``k = indices[i_{0}, ..., i_{q-1}]`` then
    ``output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]``:

    ::

       data = [
           [1.0, 1.2],
           [2.3, 3.4],
           [4.5, 5.7],
       ]
       indices = [
           [0, 1],
           [1, 2],
       ]
       output = [
           [
               [1.0, 1.2],
               [2.3, 3.4],
           ],
           [
               [2.3, 3.4],
               [4.5, 5.7],
           ],
       ]

    If ``axis = 1``, let ``k = indices[i_{0}, ..., i_{q-1}]`` then
    ``output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]``:

    ::

       data = [
           [1.0, 1.2, 1.9],
           [2.3, 3.4, 3.9],
           [4.5, 5.7, 5.9],
       ]
       indices = [
           [0, 2],
       ]
       axis = 1,
       output = [
               [[1.0, 1.9]],
               [[2.3, 3.9]],
               [[4.5, 5.9]],
       ]

    Parameters
    ==========
    data
        Type T.
        Tensor of rank r >= 1.
    indices
        Type Tind.
        Tensor of int32/int64 indices, of any rank q. All index values are
        expected to be within bounds [-s, s-1] along axis of size s. It is an
        error if any of the index values are out of bounds.
    axis
        Attribute.
        Which axis to gather on. Negative value means counting dimensions from
        the back. Accepted range is [-r, r-1] where r = rank(data).

    Returns
    =======
    output : Var
        Type T.
        Tensor of rank q + (r - 1).

    Notes
    =====
    Signature: ``ai.onnx@13::Gather``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - Tind: `tensor(int32)`, `tensor(int64)`
    """
    return _Gather(
        _Gather.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _Gather.Inputs(
            data=data,
            indices=indices,
        ),
    ).outputs.output


def gather_elements(
    data: Var,
    indices: Var,
    *,
    axis: int = 0,
) -> Var:
    r"""
    GatherElements takes two inputs ``data`` and ``indices`` of the same
    rank r >= 1 and an optional attribute ``axis`` that identifies an axis
    of ``data`` (by default, the outer-most axis, that is axis 0). It is an
    indexing operation that produces its output by indexing into the input
    data tensor at index positions determined by elements of the ``indices``
    tensor. Its output shape is the same as the shape of ``indices`` and
    consists of one value (gathered from the ``data``) for each element in
    ``indices``.

    For instance, in the 3-D case (r = 3), the output produced is determined
    by the following equations:

    ::

       out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
       out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
       out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,

    This operator is also the inverse of ScatterElements. It is similar to
    Torch's gather operation.

    Example 1:

    ::

       data = [
           [1, 2],
           [3, 4],
       ]
       indices = [
           [0, 0],
           [1, 0],
       ]
       axis = 1
       output = [
           [1, 1],
           [4, 3],
       ]

    Example 2:

    ::

       data = [
           [1, 2, 3],
           [4, 5, 6],
           [7, 8, 9],
       ]
       indices = [
           [1, 2, 0],
           [2, 0, 0],
       ]
       axis = 0
       output = [
           [4, 8, 3],
           [7, 2, 3],
       ]

    Parameters
    ==========
    data
        Type T.
        Tensor of rank r >= 1.
    indices
        Type Tind.
        Tensor of int32/int64 indices, with the same rank r as the input. All
        index values are expected to be within bounds [-s, s-1] along axis of
        size s. It is an error if any of the index values are out of bounds.
    axis
        Attribute.
        Which axis to gather on. Negative value means counting dimensions from
        the back. Accepted range is [-r, r-1] where r = rank(data).

    Returns
    =======
    output : Var
        Type T.
        Tensor of the same shape as indices.

    Notes
    =====
    Signature: ``ai.onnx@13::GatherElements``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - Tind: `tensor(int32)`, `tensor(int64)`
    """
    return _GatherElements(
        _GatherElements.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _GatherElements.Inputs(
            data=data,
            indices=indices,
        ),
    ).outputs.output


def gather_nd(
    data: Var,
    indices: Var,
    *,
    batch_dims: int = 0,
) -> Var:
    r"""
    Given ``data`` tensor of rank ``r`` >= 1, ``indices`` tensor of rank
    ``q`` >= 1, and ``batch_dims`` integer ``b``, this operator gathers
    slices of ``data`` into an output tensor of rank
    ``q + r - indices_shape[-1] - 1 - b``.

    ``indices`` is an q-dimensional integer tensor, best thought of as a
    ``(q-1)``-dimensional tensor of index-tuples into ``data``, where each
    element defines a slice of ``data``

    ``batch_dims`` (denoted as ``b``) is an integer indicating the number of
    batch dimensions, i.e the leading ``b`` number of dimensions of ``data``
    tensor and ``indices`` are representing the batches, and the gather
    starts from the ``b+1`` dimension.

    Some salient points about the inputs' rank and shape:

    1) r >= 1 and q >= 1 are to be honored. There is no dependency condition
       to be met between ranks ``r`` and ``q``

    2) The first ``b`` dimensions of the shape of ``indices`` tensor and
       ``data`` tensor must be equal.

    3) b < min(q, r) is to be honored.

    4) The ``indices_shape[-1]`` should have a value between 1 (inclusive)
       and rank ``r-b`` (inclusive)

    5) All values in ``indices`` are expected to be within bounds [-s, s-1]
       along axis of size ``s`` (i.e.)
       ``-data_shape[i] <= indices[...,i] <= data_shape[i] - 1``. It is an
       error if any of the index values are out of bounds.

    The output is computed as follows:

    The output tensor is obtained by mapping each index-tuple in the
    ``indices`` tensor to the corresponding slice of the input ``data``.

    1) If ``indices_shape[-1] > r-b`` => error condition

    2) If ``indices_shape[-1] == r-b``, since the rank of ``indices`` is
       ``q``, ``indices`` can be thought of as ``N`` ``(q-b-1)``-dimensional
       tensors containing 1-D tensors of dimension ``r-b``, where ``N`` is
       an integer equals to the product of 1 and all the elements in the
       batch dimensions of the indices_shape. Let us think of each such
       ``r-b`` ranked tensor as ``indices_slice``. Each *scalar value*
       corresponding to ``data[0:b-1,indices_slice]`` is filled into the
       corresponding location of the ``(q-b-1)``-dimensional tensor to form
       the ``output`` tensor (Example 1 below)

    3) If ``indices_shape[-1] < r-b``, since the rank of ``indices`` is
       ``q``, ``indices`` can be thought of as ``N`` ``(q-b-1)``-dimensional
       tensor containing 1-D tensors of dimension ``< r-b``. Let us think of
       each such tensors as ``indices_slice``. Each *tensor slice*
       corresponding to ``data[0:b-1, indices_slice , :]`` is filled into
       the corresponding location of the ``(q-b-1)``-dimensional tensor to
       form the ``output`` tensor (Examples 2, 3, 4 and 5 below)

    This operator is the inverse of ``ScatterND``.

    **Example 1**

    ::

       batch_dims = 0
       data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
       indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
       output  = [0,3]           # output_shape  = [2]

    **Example 2**

    ::

       batch_dims = 0
       data    = [[0,1],[2,3]]  # data_shape    = [2, 2]
       indices = [[1],[0]]      # indices_shape = [2, 1]
       output  = [[2,3],[0,1]]  # output_shape  = [2, 2]

    **Example 3**

    ::

       batch_dims = 0
       data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
       indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
       output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]

    **Example 4**

    ::

       batch_dims = 0
       data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
       indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
       output  = [[[2,3]],[[4,5]]]             # output_shape  = [2, 1, 2]

    **Example 5**

    ::

       batch_dims = 1
       data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
       indices = [[1],[0]]                     # indices_shape = [2, 1]
       output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]

    Parameters
    ==========
    data
        Type T.
        Tensor of rank r >= 1.
    indices
        Type tensor(int64).
        Tensor of rank q >= 1. All index values are expected to be within bounds
        [-s, s-1] along axis of size s. It is an error if any of the index
        values are out of bounds.
    batch_dims
        Attribute.
        The number of batch dimensions. The gather of indexing starts from
        dimension of data[batch_dims:]

    Returns
    =======
    output : Var
        Type T.
        Tensor of rank q + r - indices_shape[-1] - 1.

    Notes
    =====
    Signature: ``ai.onnx@13::GatherND``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _GatherND(
        _GatherND.Attributes(
            batch_dims=AttrInt64(batch_dims, name="batch_dims"),
        ),
        _GatherND.Inputs(
            data=data,
            indices=indices,
        ),
    ).outputs.output


def gemm(
    A: Var,
    B: Var,
    C: Optional[Var] = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    transA: int = 0,
    transB: int = 0,
) -> Var:
    r"""
    General Matrix multiplication:
    https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

    -  A' = transpose(A) if transA else A
    -  B' = transpose(B) if transB else B

    Compute Y = alpha \* A' \* B' + beta \* C, where input tensor A has
    shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input
    tensor C is broadcastable to shape (M, N), and output tensor Y has shape
    (M, N). A will be transposed before doing the computation if attribute
    transA is non-zero, same for B and transB. This operator supports
    **unidirectional broadcasting** (tensor C should be unidirectional
    broadcastable to tensor A \* B); for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.
    This operator has **optional** inputs/outputs. See `the
    doc <https://github.com/onnx/onnx/blob/main/docs/IR.md>`__ for more
    details about the representation of optional arguments. An empty string
    may be used in the place of an actual argument's name to indicate a
    missing argument. Trailing optional arguments (those not followed by an
    argument that is present) may also be simply omitted.

    Parameters
    ==========
    A
        Type T.
        Input tensor A. The shape of A should be (M, K) if transA is 0, or (K,
        M) if transA is non-zero.
    B
        Type T.
        Input tensor B. The shape of B should be (K, N) if transB is 0, or (N,
        K) if transB is non-zero.
    C
        Type T.
        Optional input tensor C. If not specified, the computation is done as if
        C is a scalar 0. The shape of C should be unidirectional broadcastable
        to (M, N).
    alpha
        Attribute.
        Scalar multiplier for the product of input tensors A \* B.
    beta
        Attribute.
        Scalar multiplier for input tensor C.
    transA
        Attribute.
        Whether A should be transposed
    transB
        Attribute.
        Whether B should be transposed

    Returns
    =======
    Y : Var
        Type T.
        Output tensor of shape (M, N).

    Notes
    =====
    Signature: ``ai.onnx@13::Gemm``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _Gemm(
        _Gemm.Attributes(
            alpha=AttrFloat32(alpha, name="alpha"),
            beta=AttrFloat32(beta, name="beta"),
            transA=AttrInt64(transA, name="transA"),
            transB=AttrInt64(transB, name="transB"),
        ),
        _Gemm.Inputs(
            A=A,
            B=B,
            C=C,
        ),
    ).outputs.Y


def global_average_pool(
    X: Var,
) -> Var:
    r"""
    GlobalAveragePool consumes an input tensor X and applies average pooling
    across the values in the same channel. This is equivalent to AveragePool
    with kernel size equal to the spatial dimension of input tensor.

    Parameters
    ==========
    X
        Type T.
        Input data tensor from the previous operator; dimensions for image case
        are (N x C x H x W), where N is the batch size, C is the number of
        channels, and H and W are the height and the width of the data. For non
        image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
        where N is the batch size.

    Returns
    =======
    Y : Var
        Type T.
        Output data tensor from pooling across the input tensor. The output
        tensor has the same rank as the input. The first two dimensions of
        output shape are the same as the input (N x C), while the other
        dimensions are all 1.

    Notes
    =====
    Signature: ``ai.onnx@1::GlobalAveragePool``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _GlobalAveragePool(
        _GlobalAveragePool.Attributes(),
        _GlobalAveragePool.Inputs(
            X=X,
        ),
    ).outputs.Y


def global_lp_pool(
    X: Var,
    *,
    p: int = 2,
) -> Var:
    r"""
    GlobalLpPool consumes an input tensor X and applies lp pool pooling
    across the values in the same channel. This is equivalent to LpPool with
    kernel size equal to the spatial dimension of input tensor.

    Parameters
    ==========
    X
        Type T.
        Input data tensor from the previous operator; dimensions for image case
        are (N x C x H x W), where N is the batch size, C is the number of
        channels, and H and W are the height and the width of the data. For non
        image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
        where N is the batch size.
    p
        Attribute.
        p value of the Lp norm used to pool over the input data.

    Returns
    =======
    Y : Var
        Type T.
        Output data tensor from pooling across the input tensor. The output
        tensor has the same rank as the input. The first two dimensions of
        output shape are the same as the input (N x C), while the other
        dimensions are all 1.

    Notes
    =====
    Signature: ``ai.onnx@2::GlobalLpPool``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _GlobalLpPool(
        _GlobalLpPool.Attributes(
            p=AttrInt64(p, name="p"),
        ),
        _GlobalLpPool.Inputs(
            X=X,
        ),
    ).outputs.Y


def global_max_pool(
    X: Var,
) -> Var:
    r"""
    GlobalMaxPool consumes an input tensor X and applies max pooling across
    the values in the same channel. This is equivalent to MaxPool with
    kernel size equal to the spatial dimension of input tensor.

    Parameters
    ==========
    X
        Type T.
        Input data tensor from the previous operator; dimensions for image case
        are (N x C x H x W), where N is the batch size, C is the number of
        channels, and H and W are the height and the width of the data. For non
        image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
        where N is the batch size.

    Returns
    =======
    Y : Var
        Type T.
        Output data tensor from pooling across the input tensor. The output
        tensor has the same rank as the input. The first two dimensions of
        output shape are the same as the input (N x C), while the other
        dimensions are all 1.

    Notes
    =====
    Signature: ``ai.onnx@1::GlobalMaxPool``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _GlobalMaxPool(
        _GlobalMaxPool.Attributes(),
        _GlobalMaxPool.Inputs(
            X=X,
        ),
    ).outputs.Y


def greater(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulted from performing the ``greater`` logical
    operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the logical operator.
    B
        Type T.
        Second input operand for the logical operator.

    Returns
    =======
    C : Var
        Type T1.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Greater``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(bool)`
    """
    return _Greater(
        _Greater.Attributes(),
        _Greater.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def greater_or_equal(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulted from performing the ``greater_equal``
    logical operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the logical operator.
    B
        Type T.
        Second input operand for the logical operator.

    Returns
    =======
    C : Var
        Type T1.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@16::GreaterOrEqual``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(bool)`
    """
    return _GreaterOrEqual(
        _GreaterOrEqual.Attributes(),
        _GreaterOrEqual.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def grid_sample(
    X: Var,
    grid: Var,
    *,
    align_corners: int = 0,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Var:
    r"""
    Given an input ``X`` and a flow-field ``grid``, computes the output
    ``Y`` using ``X`` values and pixel locations from ``grid``. Currently,
    only spatial (4-D) inputs are supported. For input ``X`` with shape (N,
    C, H, W) and ``grid`` with shape (N, H_out, W_out, 2), the output ``Y``
    will have shape (N, C, H_out, W_out).

    The tensor ``X`` contains values at centers of square pixels in a H by W
    2-dimensional image. The tensor ``grid`` describes normalized positions
    where the output ``Y`` is to be computed using a specified interpolation
    method (the mode) and a padding mode (for grid positions falling outside
    the 2-dimensional image).

    Elements in ``grid[N, H_out, W_out]`` are size-2 vectors specifying
    positions in the 2-dimensional space of ``X``. They are used to
    interpolate output values of ``Y[N, C, H_out, W_out]``.

    The GridSample operator is often used in doing grid generator and
    sampler in the `Spatial Transformer
    Networks <https://arxiv.org/abs/1506.02025>`__. See also in
    `torch.nn.functional.grid_sample <https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html#torch-nn-functional-grid-sample>`__.

    Parameters
    ==========
    X
        Type T1.
        4-D tensor of shape (N, C, H, W), where N is the batch size, C is the
        numbers of channels, H and W are the height and width of the input data.
    grid
        Type T2.
        Input offset, 4-D tensor of shape (N, H_out, W_out, 2), where H_out and
        W_out are the height and width of grid and output, Grid specifies the
        sampling pixel locations normalized by the input spatial dimensions.
        Therefore, it should have most values in the range of [-1, 1]. If grid
        has values outside the range of [-1, 1], the corresponding outputs will
        be handled as defined by padding_mode.
    align_corners
        Attribute.
        If align_corners=1, the extrema (-1 and 1) are considered as referring
        to the center points of the input's corner pixels. If align_corners=0,
        they are instead considered as referring to the corner points of the
        input's corner pixels, making the sampling more resolution agnostic.
    mode
        Attribute.
        Three interpolation modes: bilinear (default), nearest and bicubic.
    padding_mode
        Attribute.
        Support padding modes for outside grid values: ``zeros``\ (default),
        ``border``, ``reflection``. zeros: use 0 for out-of-bound grid
        locations, border: use border values for out-of-bound grid locations,
        reflection: use values at locations reflected by the border for
        out-of-bound grid locations. If index 0 represents the margin pixel, the
        reflected value at index -1 will be the same as the value at index 1.
        For location far away from the border, it will keep being reflected
        until becoming in bound. If pixel location x = -3.5 reflects by border
        -1 and becomes x' = 1.5, then reflects by border 1 and becomes x'' =
        0.5.

    Returns
    =======
    Y : Var
        Type T1.
        4-D tensor of shape (N, C, H_out, W_out) of sampled values. For integer
        input types, intermediate values are computed as floating point and cast
        to integer at the end.

    Notes
    =====
    Signature: ``ai.onnx@16::GridSample``.

    Type constraints:
     - T1: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _GridSample(
        _GridSample.Attributes(
            align_corners=AttrInt64(align_corners, name="align_corners"),
            mode=AttrString(mode, name="mode"),
            padding_mode=AttrString(padding_mode, name="padding_mode"),
        ),
        _GridSample.Inputs(
            X=X,
            grid=grid,
        ),
    ).outputs.Y


def hamming_window(
    size: Var,
    *,
    output_datatype: int = 1,
    periodic: int = 1,
) -> Var:
    r"""
    Generates a Hamming window as described in the paper
    https://ieeexplore.ieee.org/document/1455106.

    Parameters
    ==========
    size
        Type T1.
        A scalar value indicating the length of the window.
    output_datatype
        Attribute.
        The data type of the output tensor. Strictly must be one of the values
        from DataType enum in TensorProto whose values correspond to T2. The
        default value is 1 = FLOAT.
    periodic
        Attribute.
        If 1, returns a window to be used as periodic function. If 0, return a
        symmetric window. When 'periodic' is specified, hann computes a window
        of length size + 1 and returns the first size points. The default value
        is 1.

    Returns
    =======
    output : Var
        Type T2.
        A Hamming window with length: size. The output has the shape: [size].

    Notes
    =====
    Signature: ``ai.onnx@17::HammingWindow``.

    Type constraints:
     - T1: `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _HammingWindow(
        _HammingWindow.Attributes(
            output_datatype=AttrInt64(output_datatype, name="output_datatype"),
            periodic=AttrInt64(periodic, name="periodic"),
        ),
        _HammingWindow.Inputs(
            size=size,
        ),
    ).outputs.output


def hann_window(
    size: Var,
    *,
    output_datatype: int = 1,
    periodic: int = 1,
) -> Var:
    r"""
    Generates a Hann window as described in the paper
    https://ieeexplore.ieee.org/document/1455106.

    Parameters
    ==========
    size
        Type T1.
        A scalar value indicating the length of the window.
    output_datatype
        Attribute.
        The data type of the output tensor. Strictly must be one of the values
        from DataType enum in TensorProto whose values correspond to T2. The
        default value is 1 = FLOAT.
    periodic
        Attribute.
        If 1, returns a window to be used as periodic function. If 0, return a
        symmetric window. When 'periodic' is specified, hann computes a window
        of length size + 1 and returns the first size points. The default value
        is 1.

    Returns
    =======
    output : Var
        Type T2.
        A Hann window with length: size. The output has the shape: [size].

    Notes
    =====
    Signature: ``ai.onnx@17::HannWindow``.

    Type constraints:
     - T1: `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _HannWindow(
        _HannWindow.Attributes(
            output_datatype=AttrInt64(output_datatype, name="output_datatype"),
            periodic=AttrInt64(periodic, name="periodic"),
        ),
        _HannWindow.Inputs(
            size=size,
        ),
    ).outputs.output


def hard_sigmoid(
    X: Var,
    *,
    alpha: float = 0.20000000298023224,
    beta: float = 0.5,
) -> Var:
    r"""
    HardSigmoid takes one input data (Tensor<T>) and produces one output
    data (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha
    \* x + beta)), is applied to the tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor
    alpha
        Attribute.
        Value of alpha.
    beta
        Attribute.
        Value of beta.

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@6::HardSigmoid``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _HardSigmoid(
        _HardSigmoid.Attributes(
            alpha=AttrFloat32(alpha, name="alpha"),
            beta=AttrFloat32(beta, name="beta"),
        ),
        _HardSigmoid.Inputs(
            X=X,
        ),
    ).outputs.Y


def hard_swish(
    X: Var,
) -> Var:
    r"""
    HardSwish takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the HardSwish function, y = x \* max(0, min(1, alpha
    \* x + beta)) = x \* HardSigmoid<alpha, beta>(x), where alpha = 1/6 and
    beta = 0.5, is applied to the tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@14::HardSwish``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _HardSwish(
        _HardSwish.Attributes(),
        _HardSwish.Inputs(
            X=X,
        ),
    ).outputs.Y


def hardmax(
    input: Var,
    *,
    axis: int = -1,
) -> Var:
    r"""
    The operator computes the hardmax values for the given input:

    Hardmax(element in input, axis) = 1 if the element is the first maximum
    value along the specified axis, 0 otherwise

    The "axis" attribute indicates the dimension along which Hardmax will be
    performed. The output tensor has the same shape and contains the Hardmax
    values of the corresponding input.

    Parameters
    ==========
    input
        Type T.
        The input tensor of rank >= axis.
    axis
        Attribute.
        Describes the dimension Hardmax will be performed on. Negative value
        means counting dimensions from the back. Accepted range is [-r, r-1]
        where r = rank(input).

    Returns
    =======
    output : Var
        Type T.
        The output values with the same shape as the input tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Hardmax``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Hardmax(
        _Hardmax.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _Hardmax.Inputs(
            input=input,
        ),
    ).outputs.output


def identity(
    input: Var,
) -> Var:
    r"""
    Identity operator

    Parameters
    ==========
    input
        Type V.
        Input tensor

    Returns
    =======
    output : Var
        Type V.
        Tensor to copy input into.

    Notes
    =====
    Signature: ``ai.onnx@16::Identity``.

    Type constraints:
     - V: `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`, `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Identity(
        _Identity.Attributes(),
        _Identity.Inputs(
            input=input,
        ),
    ).outputs.output


def if_(
    cond: Var,
    *,
    else_branch: Callable[[], Iterable[Var]],
    then_branch: Callable[[], Iterable[Var]],
) -> Sequence[Var]:
    r"""
    If conditional

    Parameters
    ==========
    cond
        Type B.
        Condition for the if. The tensor must contain a single element.
    else_branch
        Attribute.
        Graph to run if condition is false. Has N outputs: values you wish to be
        live-out to the enclosing scope. The number of outputs must match the
        number of outputs in the then_branch.
    then_branch
        Attribute.
        Graph to run if condition is true. Has N outputs: values you wish to be
        live-out to the enclosing scope. The number of outputs must match the
        number of outputs in the else_branch.

    Returns
    =======
    outputs : Sequence[Var]
        Type V.
        Values that are live-out to the enclosing scope. The return values in
        the ``then_branch`` and ``else_branch`` must be of the same data type.
        The ``then_branch`` and ``else_branch`` may produce tensors with the
        same element type and different shapes. If corresponding outputs from
        the then-branch and the else-branch have static shapes S1 and S2, then
        the shape of the corresponding output variable of the if-node (if
        present) must be compatible with both S1 and S2 as it represents the
        union of both possible shapes.For example, if in a model file, the first
        output of ``then_branch`` is typed float tensor with shape [2] and the
        first output of ``else_branch`` is another float tensor with shape [3],
        If's first output should have (a) no shape set, or (b) a shape of rank 1
        with neither ``dim_value`` nor ``dim_param`` set, or (c) a shape of rank
        1 with a unique ``dim_param``. In contrast, the first output cannot have
        the shape [2] since [2] and [3] are not compatible.

    Notes
    =====
    Signature: ``ai.onnx@16::If``.

    Type constraints:
     - B: `tensor(bool)`
     - V: `optional(seq(tensor(bfloat16)))`, `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bfloat16))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`, `seq(tensor(bfloat16))`, `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    _else_branch_subgraph: Graph = subgraph((), else_branch)
    _then_branch_subgraph: Graph = subgraph((), then_branch)
    return _If(
        _If.Attributes(
            else_branch=AttrGraph(_else_branch_subgraph, name="else_branch"),
            then_branch=AttrGraph(_then_branch_subgraph, name="then_branch"),
        ),
        _If.Inputs(
            cond=cond,
        ),
        out_variadic=len(_else_branch_subgraph.requested_results),
    ).outputs.outputs


def instance_normalization(
    input: Var,
    scale: Var,
    B: Var,
    *,
    epsilon: float = 9.999999747378752e-06,
) -> Var:
    r"""
    Carries out instance normalization as described in the paper
    https://arxiv.org/abs/1607.08022.

    y = scale \* (x - mean) / sqrt(variance + epsilon) + B, where mean and
    variance are computed per instance per channel.

    Parameters
    ==========
    input
        Type T.
        Input data tensor from the previous operator; dimensions for image case
        are (N x C x H x W), where N is the batch size, C is the number of
        channels, and H and W are the height and the width of the data. For non
        image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
        where N is the batch size.
    scale
        Type T.
        The input 1-dimensional scale tensor of size C.
    B
        Type T.
        The input 1-dimensional bias tensor of size C.
    epsilon
        Attribute.
        The epsilon value to use to avoid division by zero.

    Returns
    =======
    output : Var
        Type T.
        The output tensor of the same shape as input.

    Notes
    =====
    Signature: ``ai.onnx@6::InstanceNormalization``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _InstanceNormalization(
        _InstanceNormalization.Attributes(
            epsilon=AttrFloat32(epsilon, name="epsilon"),
        ),
        _InstanceNormalization.Inputs(
            input=input,
            scale=scale,
            B=B,
        ),
    ).outputs.output


def isinf(
    X: Var,
    *,
    detect_negative: int = 1,
    detect_positive: int = 1,
) -> Var:
    r"""
    Map infinity to true and other values to false.

    Parameters
    ==========
    X
        Type T1.
        input
    detect_negative
        Attribute.
        (Optional) Whether map negative infinity to true. Default to 1 so that
        negative infinity induces true. Set this attribute to 0 if negative
        infinity should be mapped to false.
    detect_positive
        Attribute.
        (Optional) Whether map positive infinity to true. Default to 1 so that
        positive infinity induces true. Set this attribute to 0 if positive
        infinity should be mapped to false.

    Returns
    =======
    Y : Var
        Type T2.
        output

    Notes
    =====
    Signature: ``ai.onnx@10::IsInf``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`
     - T2: `tensor(bool)`
    """
    return _IsInf(
        _IsInf.Attributes(
            detect_negative=AttrInt64(detect_negative, name="detect_negative"),
            detect_positive=AttrInt64(detect_positive, name="detect_positive"),
        ),
        _IsInf.Inputs(
            X=X,
        ),
    ).outputs.Y


def isnan(
    X: Var,
) -> Var:
    r"""
    Returns which elements of the input are NaN.

    Parameters
    ==========
    X
        Type T1.
        input

    Returns
    =======
    Y : Var
        Type T2.
        output

    Notes
    =====
    Signature: ``ai.onnx@13::IsNaN``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(bool)`
    """
    return _IsNaN(
        _IsNaN.Attributes(),
        _IsNaN.Inputs(
            X=X,
        ),
    ).outputs.Y


def lrn(
    X: Var,
    *,
    alpha: float = 9.999999747378752e-05,
    beta: float = 0.75,
    bias: float = 1.0,
    size: int,
) -> Var:
    r"""
    Local Response Normalization proposed in the `AlexNet
    paper <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`__.
    It normalizes over local input regions. The local region is defined
    across the channels. For an element ``X[n, c, d1, ..., dk]`` in a tensor
    of shape ``(N x C x D1 x D2, ..., Dk)``, its region is
    ``{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}``.

    ``square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)``, where
    ``max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))``.

    ``Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta``

    Parameters
    ==========
    X
        Type T.
        Input data tensor from the previous operator; dimensions for image case
        are (N x C x H x W), where N is the batch size, C is the number of
        channels, and H and W are the height and the width of the data. For non
        image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
        where N is the batch size. Optionally, if dimension denotation is in
        effect, the operation expects the input data tensor to arrive with the
        dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE,
        DATA_FEATURE ...].
    alpha
        Attribute.
        Scaling parameter.
    beta
        Attribute.
        The exponent.
    bias
        Attribute.

    size
        Attribute.
        The number of channels to sum over

    Returns
    =======
    Y : Var
        Type T.
        Output tensor, which has the shape and type as input tensor

    Notes
    =====
    Signature: ``ai.onnx@13::LRN``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _LRN(
        _LRN.Attributes(
            alpha=AttrFloat32(alpha, name="alpha"),
            beta=AttrFloat32(beta, name="beta"),
            bias=AttrFloat32(bias, name="bias"),
            size=AttrInt64(size, name="size"),
        ),
        _LRN.Inputs(
            X=X,
        ),
    ).outputs.Y


def lstm(
    X: Var,
    W: Var,
    R: Var,
    B: Optional[Var] = None,
    sequence_lens: Optional[Var] = None,
    initial_h: Optional[Var] = None,
    initial_c: Optional[Var] = None,
    P: Optional[Var] = None,
    *,
    activation_alpha: Optional[Iterable[float]] = None,
    activation_beta: Optional[Iterable[float]] = None,
    activations: Optional[Iterable[str]] = None,
    clip: Optional[float] = None,
    direction: str = "forward",
    hidden_size: Optional[int] = None,
    input_forget: int = 0,
    layout: int = 0,
) -> Tuple[Var, Var, Var]:
    r"""
    Computes an one-layer LSTM. This operator is usually supported via some
    custom implementation such as CuDNN.

    Notations:

    -  ``X`` - input tensor
    -  ``i`` - input gate
    -  ``o`` - output gate
    -  ``f`` - forget gate
    -  ``c`` - cell gate
    -  ``t`` - time step (t-1 means previous time step)
    -  ``W[iofc]`` - W parameter weight matrix for input, output, forget,
       and cell gates
    -  ``R[iofc]`` - R recurrence weight matrix for input, output, forget,
       and cell gates
    -  ``Wb[iofc]`` - W bias vectors for input, output, forget, and cell
       gates
    -  ``Rb[iofc]`` - R bias vectors for input, output, forget, and cell
       gates
    -  ``P[iof]`` - P peephole weight vector for input, output, and forget
       gates
    -  ``WB[iofc]`` - W parameter weight matrix for backward input, output,
       forget, and cell gates
    -  ``RB[iofc]`` - R recurrence weight matrix for backward input, output,
       forget, and cell gates
    -  ``WBb[iofc]`` - W bias vectors for backward input, output, forget,
       and cell gates
    -  ``RBb[iofc]`` - R bias vectors for backward input, output, forget,
       and cell gates
    -  ``PB[iof]`` - P peephole weight vector for backward input, output,
       and forget gates
    -  ``H`` - Hidden state
    -  ``num_directions`` - 2 if direction == bidirectional else 1

    Activation functions:

    -  Relu(x) - max(0, x)
    -  Tanh(x) - (1 - e^{-2x})/(1 + e^{-2x})
    -  Sigmoid(x) - 1/(1 + e^{-x})

    NOTE: Below are optional

    -  Affine(x) - alpha*x + beta
    -  LeakyRelu(x) - x if x >= 0 else alpha \* x
    -  ThresholdedRelu(x) - x if x >= alpha else 0
    -  ScaledTanh(x) - alpha\ *Tanh(beta*\ x)
    -  HardSigmoid(x) - min(max(alpha*x + beta, 0), 1)
    -  Elu(x) - x if x >= 0 else alpha*(e^x - 1)
    -  Softsign(x) - x/(1 + \|x\|)
    -  Softplus(x) - log(1 + e^x)

    Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

    -  it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    -  ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    -  ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    -  Ct = ft (.) Ct-1 + it (.) ct
    -  ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    -  Ht = ot (.) h(Ct) This operator has **optional** inputs/outputs. See
       `the doc <https://github.com/onnx/onnx/blob/main/docs/IR.md>`__ for
       more details about the representation of optional arguments. An empty
       string may be used in the place of an actual argument's name to
       indicate a missing argument. Trailing optional arguments (those not
       followed by an argument that is present) may also be simply omitted.

    Parameters
    ==========
    X
        Type T.
        The input sequences packed (and potentially padded) into one 3-D tensor
        with the shape of ``[seq_length, batch_size, input_size]``.
    W
        Type T.
        The weight tensor for the gates. Concatenation of ``W[iofc]`` and
        ``WB[iofc]`` (if bidirectional) along dimension 0. The tensor has shape
        ``[num_directions, 4*hidden_size, input_size]``.
    R
        Type T.
        The recurrence weight tensor. Concatenation of ``R[iofc]`` and
        ``RB[iofc]`` (if bidirectional) along dimension 0. This tensor has shape
        ``[num_directions, 4*hidden_size, hidden_size]``.
    B
        Type T.
        The bias tensor for input gate. Concatenation of
        ``[Wb[iofc], Rb[iofc]]``, and ``[WBb[iofc], RBb[iofc]]`` (if
        bidirectional) along dimension 0. This tensor has shape
        ``[num_directions, 8*hidden_size]``. Optional: If not specified -
        assumed to be 0.
    sequence_lens
        Type T1.
        Optional tensor specifying lengths of the sequences in a batch. If not
        specified - assumed all sequences in the batch to have length
        ``seq_length``. It has shape ``[batch_size]``.
    initial_h
        Type T.
        Optional initial value of the hidden. If not specified - assumed to be
        0. It has shape ``[num_directions, batch_size, hidden_size]``.
    initial_c
        Type T.
        Optional initial value of the cell. If not specified - assumed to be 0.
        It has shape ``[num_directions, batch_size, hidden_size]``.
    P
        Type T.
        The weight tensor for peepholes. Concatenation of ``P[iof]`` and
        ``PB[iof]`` (if bidirectional) along dimension 0. It has shape
        ``[num_directions, 3*hidde_size]``. Optional: If not specified - assumed
        to be 0.
    activation_alpha
        Attribute.
        Optional scaling values used by some activation functions. The values
        are consumed in the order of activation functions, for example (f, g, h)
        in LSTM. Default values are the same as of corresponding ONNX
        operators.For example with LeakyRelu, the default alpha is 0.01.
    activation_beta
        Attribute.
        Optional scaling values used by some activation functions. The values
        are consumed in the order of activation functions, for example (f, g, h)
        in LSTM. Default values are the same as of corresponding ONNX operators.
    activations
        Attribute.
        A list of 3 (or 6 if bidirectional) activation functions for input,
        output, forget, cell, and hidden. The activation functions must be one
        of the activation functions specified above. Optional: See the equations
        for default if not specified.
    clip
        Attribute.
        Cell clip threshold. Clipping bounds the elements of a tensor in the
        range of [-threshold, +threshold] and is applied to the input of
        activations. No clip if not specified.
    direction
        Attribute.
        Specify if the RNN is forward, reverse, or bidirectional. Must be one of
        forward (default), reverse, or bidirectional.
    hidden_size
        Attribute.
        Number of neurons in the hidden layer
    input_forget
        Attribute.
        Couple the input and forget gates if 1.
    layout
        Attribute.
        The shape format of inputs X, initial_h, initial_c and outputs Y, Y_h,
        Y_c. If 0, the following shapes are expected: X.shape = [seq_length,
        batch_size, input_size], Y.shape = [seq_length, num_directions,
        batch_size, hidden_size], initial_h.shape = Y_h.shape = initial_c.shape
        = Y_c.shape = [num_directions, batch_size, hidden_size]. If 1, the
        following shapes are expected: X.shape = [batch_size, seq_length,
        input_size], Y.shape = [batch_size, seq_length, num_directions,
        hidden_size], initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape
        = [batch_size, num_directions, hidden_size].

    Returns
    =======
    Y : Var
        Type T.
        A tensor that concats all the intermediate output values of the hidden.
        It has shape ``[seq_length, num_directions, batch_size, hidden_size]``.
    Y_h : Var
        Type T.
        The last output value of the hidden. It has shape
        ``[num_directions, batch_size, hidden_size]``.
    Y_c : Var
        Type T.
        The last output value of the cell. It has shape
        ``[num_directions, batch_size, hidden_size]``.

    Notes
    =====
    Signature: ``ai.onnx@14::LSTM``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T1: `tensor(int32)`
    """
    return _LSTM(
        _LSTM.Attributes(
            activation_alpha=AttrFloat32s.maybe(
                activation_alpha, name="activation_alpha"
            ),
            activation_beta=AttrFloat32s.maybe(activation_beta, name="activation_beta"),
            activations=AttrStrings.maybe(activations, name="activations"),
            clip=AttrFloat32.maybe(clip, name="clip"),
            direction=AttrString(direction, name="direction"),
            hidden_size=AttrInt64.maybe(hidden_size, name="hidden_size"),
            input_forget=AttrInt64(input_forget, name="input_forget"),
            layout=AttrInt64(layout, name="layout"),
        ),
        _LSTM.Inputs(
            X=X,
            W=W,
            R=R,
            B=B,
            sequence_lens=sequence_lens,
            initial_h=initial_h,
            initial_c=initial_c,
            P=P,
        ),
    ).outputs._unpack_to_any()


def layer_normalization(
    X: Var,
    Scale: Var,
    B: Optional[Var] = None,
    *,
    axis: int = -1,
    epsilon: float = 9.999999747378752e-06,
    stash_type: int = 1,
) -> Tuple[Var, Var, Var]:
    r"""
    This is layer normalization defined in ONNX as function. The overall
    computation can be split into two stages. The first stage is
    standardization, which makes the normalized elements have zero mean and
    unit variances. The computation required by standardization can be
    described by the following equations.
    ``Mean = ReduceMean<axes=normalized_axes>(X) D = Sub(X, Mean) DD = Mul(D, D) Var = ReduceMean<axes=normalized_axes>(DD) VarEps = Add(Var, epsilon) StdDev = Sqrt(VarEps) InvStdDev = Reciprocal(StdDev) Normalized = Mul(D, InvStdDev)``
    where ``normalized_axes`` is ``[axis, ..., rank of X - 1]``. The
    variables ``Var`` and ``StdDev`` stand for variance and standard
    deviation, respectively. The second output is ``Mean`` and the last one
    is ``InvStdDev``. Depending on ``stash_type`` attribute, the actual
    computation must happen in different floating-point precision. For
    example, if ``stash_type`` is 1, this operator casts all input variables
    to 32-bit float, perform the computation, and finally cast
    ``Normalized`` back to the original type of ``X``. The second stage then
    scales and shifts the outcome of the first stage using
    ``NormalizedScaled = Mul(Normalized, Scale) Y = Add(NormalizedScaled, B)``
    The second stage doesn't depends on ``stash_type``. All equations are in
    `this syntax <https://github.com/onnx/onnx/blob/main/docs/Syntax.md>`__.
    The same variable (i.e., input, output, and attribute) uses the same
    name in the equations above and this operator's definition. Let ``d[i]``
    indicate the i-th dimension of ``X``. If ``X``'s shape is
    ``[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]``, the shape of
    ``Mean`` and ``InvStdDev`` is ``[d[0], ..., d[axis-1], 1, ..., 1]``.
    ``Y`` and ``X`` have the same shape. This operator supports
    unidirectional broadcasting (tensors ``Scale`` and ``B`` should be
    unidirectional broadcastable to tensor ``X``); for more details please
    check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    X
        Type T.
        Tensor to be normalized.
    Scale
        Type T.
        Scale tensor.
    B
        Type T.
        Bias tensor.
    axis
        Attribute.
        The first normalization dimension. If rank(X) is r, axis' allowed range
        is [-r, r). Negative value means counting dimensions from the back.
    epsilon
        Attribute.
        The epsilon value to use to avoid division by zero.
    stash_type
        Attribute.
        Type of Mean and InvStdDev. This also specifies stage one's computation
        precision.

    Returns
    =======
    Y : Var
        Type T.
        Normalized tensor.
    Mean : Var
        Type U.
        Saved mean used during training to speed up gradient computation
    InvStdDev : Var
        Type U.
        Saved inverse standard deviation used during training to speed up
        gradient computation.

    Notes
    =====
    Signature: ``ai.onnx@17::LayerNormalization``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - U: `tensor(bfloat16)`, `tensor(float)`
    """
    return _LayerNormalization(
        _LayerNormalization.Attributes(
            axis=AttrInt64(axis, name="axis"),
            epsilon=AttrFloat32(epsilon, name="epsilon"),
            stash_type=AttrInt64(stash_type, name="stash_type"),
        ),
        _LayerNormalization.Inputs(
            X=X,
            Scale=Scale,
            B=B,
        ),
    ).outputs._unpack_to_any()


def leaky_relu(
    X: Var,
    *,
    alpha: float = 0.009999999776482582,
) -> Var:
    r"""
    LeakyRelu takes input data (Tensor<T>) and an argument alpha, and
    produces one output data (Tensor<T>) where the function
    ``f(x) = alpha * x for x < 0``, ``f(x) = x for x >= 0``, is applied to
    the data tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor
    alpha
        Attribute.
        Coefficient of leakage.

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@16::LeakyRelu``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _LeakyRelu(
        _LeakyRelu.Attributes(
            alpha=AttrFloat32(alpha, name="alpha"),
        ),
        _LeakyRelu.Inputs(
            X=X,
        ),
    ).outputs.Y


def less(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulted from performing the ``less`` logical
    operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the logical operator.
    B
        Type T.
        Second input operand for the logical operator.

    Returns
    =======
    C : Var
        Type T1.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Less``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(bool)`
    """
    return _Less(
        _Less.Attributes(),
        _Less.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def less_or_equal(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulted from performing the ``less_equal`` logical
    operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the logical operator.
    B
        Type T.
        Second input operand for the logical operator.

    Returns
    =======
    C : Var
        Type T1.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@16::LessOrEqual``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(bool)`
    """
    return _LessOrEqual(
        _LessOrEqual.Attributes(),
        _LessOrEqual.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def log(
    input: Var,
) -> Var:
    r"""
    Calculates the natural log of the given input tensor, element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The natural log of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@13::Log``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Log(
        _Log.Attributes(),
        _Log.Inputs(
            input=input,
        ),
    ).outputs.output


def log_softmax(
    input: Var,
    *,
    axis: int = -1,
) -> Var:
    r"""
    The operator computes the log of softmax values for the given input:

    LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

    The "axis" attribute indicates the dimension along which LogSoftmax will
    be performed. The output tensor has the same shape and contains the
    LogSoftmax values of the corresponding input.

    Parameters
    ==========
    input
        Type T.
        The input tensor of rank >= axis.
    axis
        Attribute.
        Describes the dimension LogSoftmax will be performed on. Negative value
        means counting dimensions from the back. Accepted range is [-r, r-1]
        where r = rank(input).

    Returns
    =======
    output : Var
        Type T.
        The output values with the same shape as the input tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::LogSoftmax``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _LogSoftmax(
        _LogSoftmax.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _LogSoftmax.Inputs(
            input=input,
        ),
    ).outputs.output


def loop(
    M: Optional[Var] = None,
    cond: Optional[Var] = None,
    v_initial: Sequence[Var] = (),
    *,
    body: Callable[..., Iterable[Var]],
) -> Sequence[Var]:
    r"""
    Generic Looping construct. This loop has multiple termination
    conditions:

    1) Trip count. Iteration count specified at runtime. Set by specifying
       the input M. Optional. Set to empty string to omit. Note that a
       static trip count (specified at graph construction time) can be
       specified by passing in a constant node for input M.
    2) Loop termination condition. This is an input to the op that
       determines whether to run the first iteration and also a loop-carried
       dependency for the body graph. The body graph must yield a value for
       the condition variable, whether this input is provided or not.

    This table summarizes the operating modes of this operator with
    equivalent C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    -  input ("", ""): for (int i=0; ; ++i) { cond = ... // Note this value
       is ignored, but is required in the body }

    -  input ("", cond) // Note this is analogous to a while loop bool cond
       = ...; for (int i=0; cond; ++i) { cond = ...; }

    -  input ("", 1) // Note this is analogous to a do-while loop bool cond
       = true for (int i=0; cond; ++i) { cond = ...; }

    -  input (trip_count, "") // Note this is analogous to a for loop int
       trip_count = ... for (int i=0; i < trip_count; ++i) { cond = ...; //
       ignored }

    -  input (trip_count, cond) int trip_count = ...; bool cond = ...; for
       (int i=0; i < trip_count && cond; ++i) { cond = ...; }

    *Sample usage - cond as well as trip count*

    ::

       graph predict-net {
         %a = Constant[value = <Scalar Tensor [3]>]()
         %b = Constant[value = <Scalar Tensor [6]>]()
         %keepgoing = Constant[value = <Scalar Tensor [1]>]()
         %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
         %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
         return
       }

       graph body-net (
         %i[INT32, scalar]           // iteration number
         %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
         %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
       ) {
         %my_local = Add(%a, %b_in)
         %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
         %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
         %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
         return %keepgoing_out, %b_out, %user_defined_val
       }

    *Sample equivalent C code*

    ::

       {
         /* User-defined code (enclosing scope) */
         int a = 3, b = 6;
         bool keepgoing = true; // Analogous to input cond
         /* End user-defined code */

         /* Implicitly-defined code */
         const int max_trip_count = 10; // Analogous to input M
         int user_defined_vals[]; // Imagine this is resizable
         /* End implicitly-defined code */
         /* initialize loop-carried variables and scan-output variables */
         bool keepgoing_out = keepgoing
         int b_out = b

         for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
           /* Implicitly-defined code: bind actual parameter values
              to formal parameter variables of loop-body */
           bool keepgoing_in = keepgoing_out;
           bool b_in = b_out;

           /* User-defined code (loop body) */
           int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
           b_out = a - b_in;
           keepgoing_out = my_local > b_out;
           user_defined_val = b_in + b_in; // b_in and b_out are different variables
           /* End user-defined code */

           /* Implicitly defined-code */
           user_defined_vals[i] = user_defined_val // accumulate scan-output values
         }
         // int t = my_local; // Can't do this. my_local is not accessible here.

         // The values below are bound to the output variables of the loop and therefore accessible
         // b_out; user_defined_vals; keepgoing_out;
       }

    There are several things of note in this code snippet:

    1) Values from the enclosing scope (i.e. variable "a" here) are in scope
       and can be referenced in the inputs of the loop.
    2) Any values computed in the loop body that needs to be used in a
       subsequent iteration or after the loop are modelled using a pair of
       variables in the loop-body, consisting of an input variable (eg.,
       b_in) and an output variable (eg., b_out). These are referred to as
       loop-carried dependences. The loop operation node supplies the input
       value of the input variable for the first iteration, and returns the
       output value of the output variable produced by the final iteration.
    3) Scan_output variables are used to implicitly concatenate values
       computed across all the iterations. In the above example, the value
       of user_defined_val computed over all iterations are concatenated and
       returned as the value of user_defined_vals after the loop.
    4) Values created in the body cannot be accessed in the enclosing scope,
       except using the mechanism described above.

    Note that the semantics of this op support "diagonal" or "wavefront"
    execution. (See Step 3 here for an example:
    https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
    Frontends should emit multi-layer RNNs as a series of While operators
    (with time being the inner looping dimension), with each successive
    layer consuming the scan_outputs from the previous layer, possibly going
    through several point-wise operators (e.g. dropout, residual
    connections, linear layer).

    The input/output of subgraph (produced by loop node) matching is based
    on order instead of name. The implementation will figure out the names
    based on this order.

    Parameters
    ==========
    M
        Type I.
        A maximum trip-count for the loop specified at runtime. Optional. Pass
        empty string to skip.
    cond
        Type B.
        A boolean termination condition. Optional. Pass empty string to skip.
    v_initial
        Type V.
        The initial values of any loop-carried dependencies (values that change
        across loop iterations)
    body
        Attribute.
        The graph run each iteration. It has 2+N inputs: (iteration_num,
        condition, loop carried dependencies...). It has 1+N+K outputs:
        (condition, loop carried dependencies..., scan_outputs...). Each
        scan_output is created by concatenating the value of the specified
        output value at the end of each iteration of the loop. It is an error if
        the dimensions or data type of these scan_outputs change across loop
        iterations.

    Returns
    =======
    v_final_and_scan_outputs : Sequence[Var]
        Type V.
        Final N loop carried dependency values then K scan_outputs. Scan outputs
        must be Tensors.

    Notes
    =====
    Signature: ``ai.onnx@16::Loop``.

    Type constraints:
     - I: `tensor(int64)`
     - B: `tensor(bool)`
     - V: `optional(seq(tensor(bfloat16)))`, `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bfloat16))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`, `seq(tensor(bfloat16))`, `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    _body_subgraph: Graph = subgraph(
        typing_cast(List[Type], [Tensor(np.int64, (1,)), Tensor(np.bool_, (1,))])
        + [var.unwrap_type() for var in v_initial],
        body,
    )
    return _Loop(
        _Loop.Attributes(
            body=AttrGraph(_body_subgraph, name="body"),
        ),
        _Loop.Inputs(
            M=M,
            cond=cond,
            v_initial=v_initial,
        ),
        out_variadic=len(_body_subgraph.requested_results) - 1,
    ).outputs.v_final_and_scan_outputs


def lp_normalization(
    input: Var,
    *,
    axis: int = -1,
    p: int = 2,
) -> Var:
    r"""
    Given a matrix, apply Lp-normalization along the provided axis.

    Parameters
    ==========
    input
        Type T.
        Input matrix
    axis
        Attribute.
        The axis on which to apply normalization, -1 mean last axis.
    p
        Attribute.
        The order of the normalization, only 1 or 2 are supported.

    Returns
    =======
    output : Var
        Type T.
        Matrix after normalization

    Notes
    =====
    Signature: ``ai.onnx@1::LpNormalization``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _LpNormalization(
        _LpNormalization.Attributes(
            axis=AttrInt64(axis, name="axis"),
            p=AttrInt64(p, name="p"),
        ),
        _LpNormalization.Inputs(
            input=input,
        ),
    ).outputs.output


def lp_pool(
    X: Var,
    *,
    auto_pad: str = "NOTSET",
    kernel_shape: Iterable[int],
    p: int = 2,
    pads: Optional[Iterable[int]] = None,
    strides: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    LpPool consumes an input tensor X and applies Lp pooling across the
    tensor according to kernel sizes, stride sizes, and pad lengths. Lp
    pooling consisting of computing the Lp norm on all values of a subset of
    the input tensor according to the kernel size and downsampling the data
    into the output tensor Y for further processing.

    Parameters
    ==========
    X
        Type T.
        Input data tensor from the previous operator; dimensions for image case
        are (N x C x H x W), where N is the batch size, C is the number of
        channels, and H and W are the height and the width of the data. For non
        image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
        where N is the batch size.
    auto_pad
        Attribute.
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where
        default value is NOTSET, which means explicit padding is used.
        SAME_UPPER or SAME_LOWER mean pad the input so that
        ``output_shape[i] = ceil(input_shape[i] / strides[i])`` for each axis
        ``i``. The padding is split between the two sides equally or almost
        equally (depending on whether it is even or odd). In case the padding is
        an odd number, the extra padding is added at the end for SAME_UPPER and
        at the beginning for SAME_LOWER.
    kernel_shape
        Attribute.
        The size of the kernel along each axis.
    p
        Attribute.
        p value of the Lp norm used to pool over the input data.
    pads
        Attribute.
        Padding for the beginning and ending along each spatial axis, it can
        take any value greater than or equal to 0. The value represent the
        number of pixels added to the beginning and end part of the
        corresponding axis. ``pads`` format should be as follow [x1_begin,
        x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
        added at the beginning of axis ``i`` and xi_end, the number of pixels
        added at the end of axis ``i``. This attribute cannot be used
        simultaneously with auto_pad attribute. If not present, the padding
        defaults to 0 along start and end of each spatial axis.
    strides
        Attribute.
        Stride along each spatial axis. If not present, the stride defaults to 1
        along each spatial axis.

    Returns
    =======
    Y : Var
        Type T.
        Output data tensor from Lp pooling across the input tensor. Dimensions
        will vary based on various kernel, stride, and pad sizes.

    Notes
    =====
    Signature: ``ai.onnx@11::LpPool``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _LpPool(
        _LpPool.Attributes(
            auto_pad=AttrString(auto_pad, name="auto_pad"),
            kernel_shape=AttrInt64s(kernel_shape, name="kernel_shape"),
            p=AttrInt64(p, name="p"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _LpPool.Inputs(
            X=X,
        ),
    ).outputs.Y


def matmul(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Matrix product that behaves like numpy.matmul:
    https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html

    Parameters
    ==========
    A
        Type T.
        N-dimensional matrix A
    B
        Type T.
        N-dimensional matrix B

    Returns
    =======
    Y : Var
        Type T.
        Matrix multiply results from A \* B

    Notes
    =====
    Signature: ``ai.onnx@13::MatMul``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _MatMul(
        _MatMul.Attributes(),
        _MatMul.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.Y


def matmul_integer(
    A: Var,
    B: Var,
    a_zero_point: Optional[Var] = None,
    b_zero_point: Optional[Var] = None,
) -> Var:
    r"""
    Matrix product that behaves like numpy.matmul:
    https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
    The production MUST never overflow. The accumulation may overflow if and
    only if in 32 bits.

    Parameters
    ==========
    A
        Type T1.
        N-dimensional matrix A
    B
        Type T2.
        N-dimensional matrix B
    a_zero_point
        Type T1.
        Zero point tensor for input 'A'. It's optional and default value is 0.
        It could be a scalar or N-D tensor. Scalar refers to per tensor
        quantization whereas N-D refers to per row quantization. If the input is
        2D of shape [M, K] then zero point tensor may be an M element vector
        [zp_1, zp_2, ..., zp_M]. If the input is N-D tensor with shape [D1, D2,
        M, K] then zero point tensor may have shape [D1, D2, M, 1].
    b_zero_point
        Type T2.
        Zero point tensor for input 'B'. It's optional and default value is 0.
        It could be a scalar or a N-D tensor, Scalar refers to per tensor
        quantization whereas N-D refers to per col quantization. If the input is
        2D of shape [K, N] then zero point tensor may be an N element vector
        [zp_1, zp_2, ..., zp_N]. If the input is N-D tensor with shape [D1, D2,
        K, N] then zero point tensor may have shape [D1, D2, 1, N].

    Returns
    =======
    Y : Var
        Type T3.
        Matrix multiply results from A \* B

    Notes
    =====
    Signature: ``ai.onnx@10::MatMulInteger``.

    Type constraints:
     - T1: `tensor(int8)`, `tensor(uint8)`
     - T2: `tensor(int8)`, `tensor(uint8)`
     - T3: `tensor(int32)`
    """
    return _MatMulInteger(
        _MatMulInteger.Attributes(),
        _MatMulInteger.Inputs(
            A=A,
            B=B,
            a_zero_point=a_zero_point,
            b_zero_point=b_zero_point,
        ),
    ).outputs.Y


def max(
    data_0: Sequence[Var],
) -> Var:
    r"""
    Element-wise max of each of the input tensors (with Numpy-style
    broadcasting support). All inputs and outputs must have the same data
    type. This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    data_0
        Type T.
        List of tensors for max.

    Returns
    =======
    max : Var
        Type T.
        Output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Max``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Max(
        _Max.Attributes(),
        _Max.Inputs(
            data_0=data_0,
        ),
    ).outputs.max


def max_pool(
    X: Var,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    dilations: Optional[Iterable[int]] = None,
    kernel_shape: Iterable[int],
    pads: Optional[Iterable[int]] = None,
    storage_order: int = 0,
    strides: Optional[Iterable[int]] = None,
) -> Tuple[Var, Var]:
    r"""
    MaxPool consumes an input tensor X and applies max pooling across the
    tensor according to kernel sizes, stride sizes, and pad lengths. max
    pooling consisting of computing the max on all values of a subset of the
    input tensor according to the kernel size and downsampling the data into
    the output tensor Y for further processing. The output spatial shape is
    calculated differently depending on whether explicit padding is used,
    where pads is employed, or auto padding is used, where auto_pad is
    utilized. With explicit padding
    (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):

    ::

       output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)

    or

    ::

       output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)

    if ceil_mode is enabled. ``pad_shape[i]`` is the sum of pads along axis
    ``i``. Sliding windows that would start in the right padded region are
    ignored.

    ``auto_pad`` is a DEPRECATED attribute. If you are using them currently,
    the output spatial shape will be following when ceil_mode is enabled:

    ::

       VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
       SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])

    or when ceil_mode is disabled
    (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):

    ::

       VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
       SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1

    And pad shape will be following if ``SAME_UPPER`` or ``SAME_LOWER``:

    ::

       pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]

    The output of each pooling window is maximum number of elements exclude
    pad.

    Parameters
    ==========
    X
        Type T.
        Input data tensor from the previous operator; dimensions for image case
        are (N x C x H x W), where N is the batch size, C is the number of
        channels, and H and W are the height and the width of the data. For non
        image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
        where N is the batch size. Optionally, if dimension denotation is in
        effect, the operation expects the input data tensor to arrive with the
        dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE,
        DATA_FEATURE ...].
    auto_pad
        Attribute.
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where
        default value is NOTSET, which means explicit padding is used.
        SAME_UPPER or SAME_LOWER mean pad the input so that
        ``output_shape[i] = ceil(input_shape[i] / strides[i])`` for each axis
        ``i``. The padding is split between the two sides equally or almost
        equally (depending on whether it is even or odd). In case the padding is
        an odd number, the extra padding is added at the end for SAME_UPPER and
        at the beginning for SAME_LOWER.
    ceil_mode
        Attribute.
        Whether to use ceil or floor (default) to compute the output shape.
    dilations
        Attribute.
        Dilation value along each spatial axis of filter. If not present, the
        dilation defaults to 1 along each spatial axis.
    kernel_shape
        Attribute.
        The size of the kernel along each axis.
    pads
        Attribute.
        Padding for the beginning and ending along each spatial axis, it can
        take any value greater than or equal to 0. The value represent the
        number of pixels added to the beginning and end part of the
        corresponding axis. ``pads`` format should be as follow [x1_begin,
        x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
        added at the beginning of axis ``i`` and xi_end, the number of pixels
        added at the end of axis ``i``. This attribute cannot be used
        simultaneously with auto_pad attribute. If not present, the padding
        defaults to 0 along start and end of each spatial axis.
    storage_order
        Attribute.
        The storage order of the tensor. 0 is row major, and 1 is column major.
        This attribute is used only to convert an n-tuple index value into a
        single integer value for producing the second output.
    strides
        Attribute.
        Stride along each spatial axis. If not present, the stride defaults to 1
        along each spatial axis.

    Returns
    =======
    Y : Var
        Type T.
        Output data tensor from average or max pooling across the input tensor.
        Dimensions will vary based on various kernel, stride, and pad sizes.
        Floor value of the dimension is used
    Indices : Var
        Type I.
        Indices tensor from max pooling across the input tensor. The dimensions
        of indices are the same as output tensor. The values in indices of are
        the indices of the selected values during pooling. The indices are
        computed as flatten 1-D tensor, and the indices do not consider padding.
        So the values in indices are in [0, N x C x D1 x ... x Dn).

    Notes
    =====
    Signature: ``ai.onnx@12::MaxPool``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int8)`, `tensor(uint8)`
     - I: `tensor(int64)`
    """
    return _MaxPool(
        _MaxPool.Attributes(
            auto_pad=AttrString(auto_pad, name="auto_pad"),
            ceil_mode=AttrInt64(ceil_mode, name="ceil_mode"),
            dilations=AttrInt64s.maybe(dilations, name="dilations"),
            kernel_shape=AttrInt64s(kernel_shape, name="kernel_shape"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            storage_order=AttrInt64(storage_order, name="storage_order"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _MaxPool.Inputs(
            X=X,
        ),
    ).outputs._unpack_to_any()


def max_roi_pool(
    X: Var,
    rois: Var,
    *,
    pooled_shape: Iterable[int],
    spatial_scale: float = 1.0,
) -> Var:
    r"""
    ROI max pool consumes an input tensor X and region of interests (RoIs)
    to apply max pooling across each RoI, to produce output 4-D tensor of
    shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).

    Parameters
    ==========
    X
        Type T.
        Input data tensor from the previous operator; dimensions for image case
        are (N x C x H x W), where N is the batch size, C is the number of
        channels, and H and W are the height and the width of the data.
    rois
        Type T.
        RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of shape
        (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].
    pooled_shape
        Attribute.
        ROI pool output shape (height, width).
    spatial_scale
        Attribute.
        Multiplicative spatial scale factor to translate ROI coordinates from
        their input scale to the scale used when pooling.

    Returns
    =======
    Y : Var
        Type T.
        RoI pooled output 4-D tensor of shape (num_rois, channels,
        pooled_shape[0], pooled_shape[1]).

    Notes
    =====
    Signature: ``ai.onnx@1::MaxRoiPool``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _MaxRoiPool(
        _MaxRoiPool.Attributes(
            pooled_shape=AttrInt64s(pooled_shape, name="pooled_shape"),
            spatial_scale=AttrFloat32(spatial_scale, name="spatial_scale"),
        ),
        _MaxRoiPool.Inputs(
            X=X,
            rois=rois,
        ),
    ).outputs.Y


def max_unpool(
    X: Var,
    I: Var,
    output_shape: Optional[Var] = None,
    *,
    kernel_shape: Iterable[int],
    pads: Optional[Iterable[int]] = None,
    strides: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    MaxUnpool essentially computes the partial inverse of the MaxPool op.
    The input information to this op is typically the output information
    from a MaxPool op. The first input tensor X is the tensor that needs to
    be unpooled, which is typically the pooled tensor (first output) from
    MaxPool. The second input tensor, I, contains the indices to the
    (locally maximal) elements corresponding to the elements in the first
    input tensor X. Input tensor I is typically the second output of the
    MaxPool op. The third (optional) input is a tensor that specifies the
    output size of the unpooling operation.

    MaxUnpool is intended to do 'partial' inverse of the MaxPool op.
    'Partial' because all the non-maximal values from the original input to
    MaxPool are set to zero in the output of the MaxUnpool op. Pooling the
    result of an unpooling operation should give back the original input to
    the unpooling op.

    MaxUnpool can produce the same output size for several input sizes,
    which makes unpooling op ambiguous. The third input argument,
    output_size, is meant to disambiguate the op and produce output tensor
    of known/predictable size.

    In addition to the inputs, MaxUnpool takes three attributes, namely
    kernel_shape, strides, and pads, which define the exact unpooling op.
    The attributes typically have the same values as the corresponding
    pooling op that the unpooling op is trying to invert.

    Parameters
    ==========
    X
        Type T1.
        Input data tensor that has to be unpooled. This tensor is typically the
        first output of the MaxPool op.Dimensions for image case are (N x C x H
        x W), where N is the batch size, C is the number of channels, and H and
        W are the height and the width of the data. For non-image case, the
        dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the
        batch size. Optionally, if dimension denotation is in effect, the
        operation expects the input data tensor to arrive with the dimension
        denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE
        ...].
    I
        Type T2.
        Input data tensor containing the indices corresponding to elements in
        the first input tensor X.This tensor is typically the second output of
        the MaxPool op.Dimensions must be the same as input tensor X. The
        indices are linear, i.e. computed considering the tensor as flattened
        1-D tensor, assuming row-major storage. Also, the linear indices should
        not consider padding. So the values in indices are in the range [0, N x
        C x D1 x ... x Dn).
    output_shape
        Type T2.
        The shape of the output can be explicitly set which will cause pads
        values to be auto generated. If 'output_shape' is specified, 'pads'
        values are ignored.
    kernel_shape
        Attribute.
        The size of the kernel along each axis.
    pads
        Attribute.
        Padding for the beginning and ending along each spatial axis, it can
        take any value greater than or equal to 0. The value represent the
        number of pixels added to the beginning and end part of the
        corresponding axis. ``pads`` format should be as follow [x1_begin,
        x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
        added at the beginning of axis ``i`` and xi_end, the number of pixels
        added at the end of axis ``i``. This attribute cannot be used
        simultaneously with auto_pad attribute. If not present, the padding
        defaults to 0 along start and end of each spatial axis.
    strides
        Attribute.
        Stride along each spatial axis. If not present, the stride defaults to 1
        along each spatial axis.

    Returns
    =======
    output : Var
        Type T1.
        Output data tensor that contains the result of the unpooling.

    Notes
    =====
    Signature: ``ai.onnx@11::MaxUnpool``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(int64)`
    """
    return _MaxUnpool(
        _MaxUnpool.Attributes(
            kernel_shape=AttrInt64s(kernel_shape, name="kernel_shape"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _MaxUnpool.Inputs(
            X=X,
            I=I,
            output_shape=output_shape,
        ),
    ).outputs.output


def mean(
    data_0: Sequence[Var],
) -> Var:
    r"""
    Element-wise mean of each of the input tensors (with Numpy-style
    broadcasting support). All inputs and outputs must have the same data
    type. This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    data_0
        Type T.
        List of tensors for mean.

    Returns
    =======
    mean : Var
        Type T.
        Output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Mean``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Mean(
        _Mean.Attributes(),
        _Mean.Inputs(
            data_0=data_0,
        ),
    ).outputs.mean


def mean_variance_normalization(
    X: Var,
    *,
    axes: Iterable[int] = (0, 2, 3),
) -> Var:
    r"""
    A MeanVarianceNormalization Function: Perform mean variance
    normalization on the input tensor X using formula:
    ``(X-EX)/sqrt(E(X-EX)^2)``

    Parameters
    ==========
    X
        Type T.
        Input tensor
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to calculate
        along axes [0,2,3] for calculating mean and variance along each channel.
        Two variables with the same C-coordinate are associated with the same
        mean and variance.

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@13::MeanVarianceNormalization``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _MeanVarianceNormalization(
        _MeanVarianceNormalization.Attributes(
            axes=AttrInt64s(axes, name="axes"),
        ),
        _MeanVarianceNormalization.Inputs(
            X=X,
        ),
    ).outputs.Y


def mel_weight_matrix(
    num_mel_bins: Var,
    dft_length: Var,
    sample_rate: Var,
    lower_edge_hertz: Var,
    upper_edge_hertz: Var,
    *,
    output_datatype: int = 1,
) -> Var:
    r"""
    Generate a MelWeightMatrix that can be used to re-weight a Tensor
    containing a linearly sampled frequency spectra (from DFT or STFT) into
    num_mel_bins frequency information based on the [lower_edge_hertz,
    upper_edge_hertz] range on the mel scale. This function defines the mel
    scale in terms of a frequency in hertz according to the following
    formula:

    ::

       mel(f) = 2595 * log10(1 + f/700)

    In the returned matrix, all the triangles (filterbanks) have a peak
    value of 1.0.

    The returned MelWeightMatrix can be used to right-multiply a spectrogram
    S of shape [frames, num_spectrogram_bins] of linear scale spectrum
    values (e.g. STFT magnitudes) to generate a "mel spectrogram" M of shape
    [frames, num_mel_bins].

    Parameters
    ==========
    num_mel_bins
        Type T1.
        The number of bands in the mel spectrum.
    dft_length
        Type T1.
        The size of the original DFT. The size of the original DFT is used to
        infer the size of the onesided DFT, which is understood to be
        floor(dft_length/2) + 1, i.e. the spectrogram only contains the
        nonredundant DFT bins.
    sample_rate
        Type T1.
        Samples per second of the input signal used to create the spectrogram.
        Used to figure out the frequencies corresponding to each spectrogram
        bin, which dictates how they are mapped into the mel scale.
    lower_edge_hertz
        Type T2.
        Lower bound on the frequencies to be included in the mel spectrum. This
        corresponds to the lower edge of the lowest triangular band.
    upper_edge_hertz
        Type T2.
        The desired top edge of the highest frequency band.
    output_datatype
        Attribute.
        The data type of the output tensor. Strictly must be one of the values
        from DataType enum in TensorProto whose values correspond to T3. The
        default value is 1 = FLOAT.

    Returns
    =======
    output : Var
        Type T3.
        The Mel Weight Matrix. The output has the shape: [floor(dft_length/2) +
        1][num_mel_bins].

    Notes
    =====
    Signature: ``ai.onnx@17::MelWeightMatrix``.

    Type constraints:
     - T1: `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T3: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _MelWeightMatrix(
        _MelWeightMatrix.Attributes(
            output_datatype=AttrInt64(output_datatype, name="output_datatype"),
        ),
        _MelWeightMatrix.Inputs(
            num_mel_bins=num_mel_bins,
            dft_length=dft_length,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
        ),
    ).outputs.output


def min(
    data_0: Sequence[Var],
) -> Var:
    r"""
    Element-wise min of each of the input tensors (with Numpy-style
    broadcasting support). All inputs and outputs must have the same data
    type. This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    data_0
        Type T.
        List of tensors for min.

    Returns
    =======
    min : Var
        Type T.
        Output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Min``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Min(
        _Min.Attributes(),
        _Min.Inputs(
            data_0=data_0,
        ),
    ).outputs.min


def mod(
    A: Var,
    B: Var,
    *,
    fmod: int = 0,
) -> Var:
    r"""
    Performs element-wise binary modulus (with Numpy-style broadcasting
    support). The sign of the remainder is the same as that of the Divisor.

    Mod operator can also behave like C fmod() or numpy.fmod. In this case,
    the sign of the remainder however, will be the same as the Dividend (in
    contrast to integer mod). To force a behavior like numpy.fmod() an
    'fmod' Attribute is provided. This attribute is set to 0 by default
    causing the behavior to be like integer mod. Setting this attribute to 1
    causes the remainder to be calculated similar to that of numpy.fmod().

    If the input type is floating point, then ``fmod`` attribute must be set
    to 1.

    In case of dividend being zero, the results will be platform dependent.

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        Dividend tensor
    B
        Type T.
        Divisor tensor
    fmod
        Attribute.
        Whether the operator should behave like fmod (default=0 meaning it will
        do integer mods); Set this to 1 to force fmod treatment

    Returns
    =======
    C : Var
        Type T.
        Remainder tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Mod``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Mod(
        _Mod.Attributes(
            fmod=AttrInt64(fmod, name="fmod"),
        ),
        _Mod.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def mul(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Performs element-wise binary multiplication (with Numpy-style
    broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    (Opset 14 change): Extend supported types to include uint8, int8,
    uint16, and int16.

    Parameters
    ==========
    A
        Type T.
        First operand.
    B
        Type T.
        Second operand.

    Returns
    =======
    C : Var
        Type T.
        Result, has same element type as two inputs

    Notes
    =====
    Signature: ``ai.onnx@14::Mul``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Mul(
        _Mul.Attributes(),
        _Mul.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def multinomial(
    input: Var,
    *,
    dtype: npt.DTypeLike = np.int32,
    sample_size: int = 1,
    seed: Optional[float] = None,
) -> Var:
    r"""
    Generate a tensor of samples from a multinomial distribution according
    to the probabilities of each of the possible outcomes.

    Parameters
    ==========
    input
        Type T1.
        Input tensor with shape [batch_size, class_size], where class_size is
        the number of all possible outcomes. Each value along the axis zero
        represents the unnormalized log-probability of each corresponding
        outcome in a batch.
    dtype
        Attribute.
        (Optional) The data type for the elements of the output tensor, if not
        specified, we will use int32.
    sample_size
        Attribute.
        Number of times to sample.
    seed
        Attribute.
        (Optional) Seed to the random generator, if not specified we will auto
        generate one.

    Returns
    =======
    output : Var
        Type T2.
        Output tensor with shape [batch_size, sample_size], where sample_size is
        the number of times to sample. Each value along the axis zero represents
        the outcome of the corresponding sample in a batch.

    Notes
    =====
    Signature: ``ai.onnx@7::Multinomial``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(int32)`, `tensor(int64)`
    """
    return _Multinomial(
        _Multinomial.Attributes(
            dtype=AttrDtype(dtype, name="dtype"),
            sample_size=AttrInt64(sample_size, name="sample_size"),
            seed=AttrFloat32.maybe(seed, name="seed"),
        ),
        _Multinomial.Inputs(
            input=input,
        ),
    ).outputs.output


def neg(
    X: Var,
) -> Var:
    r"""
    Neg takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where each element flipped sign, y = -x, is applied to the
    tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Neg``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`
    """
    return _Neg(
        _Neg.Attributes(),
        _Neg.Inputs(
            X=X,
        ),
    ).outputs.Y


def negative_log_likelihood_loss(
    input: Var,
    target: Var,
    weight: Optional[Var] = None,
    *,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> Var:
    r"""
    A NegativeLogLikelihoodLoss operator computes (weighted) negative log
    likelihood loss. Its "input" tensor has the shape of (N, C, d1, d2, ...,
    dk) where k >= 0. The "input" tensor contains log-probabilities for
    input[n, :, d_1, d_2,..., d_k] being in a class of [0, C). The
    operator's "target" input tensor has the shape of (N, d1, d2, ..., dk).
    It encodes class labels (one of C classes) or it may contain a special
    value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x
    dk samples. The loss value for input[n, :, d_1, d_2,...d_k] being
    classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

    ::

       loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].

    When an optional "weight" is provided, the sample loss is calculated as:

    ::

       loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].

    loss is zero for the case when target-value equals ignore_index.

    ::

       loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index

    If "reduction" attribute is set to "none", the operator's output will be
    the above loss with shape (N, d1, d2, ..., dk). If "reduction" attribute
    is set to "mean" (the default attribute value), the output loss is
    (weight) averaged:

    ::

       mean(loss), if "weight" is not provided,

    or if weight is provided,

    ::

       sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.

    If "reduction" attribute is set to "sum", the output is a scalar:
    ``sum(loss)``.

    See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

    Example 1:

    ::

       // negative log likelihood loss, "none" reduction
       N, C, d1 = 2, 3, 2
       input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
                 [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
       target = [[2, 1], [0, 2]]

       loss = np.zeros((N, d1))
       for n in range(N):
           for d_1 in range(d1):
               c = target[n][d_1]
               loss[n][d_1] = -input[n][c][d_1]

       // print(loss)
       // [[-3. -2.]
       //  [-0. -2.]]

    Example 2:

    ::

       // weighted negative log likelihood loss, sum reduction
       N, C, d1 = 2, 3, 2
       input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
               [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
       target = [[2, 1], [0, 2]]
       weight = [0.2, 0.3, 0.1]
       loss = np.zeros((N, d1))
       for n in range(N):
           for d_1 in range(d1):
               c = target[n][d_1]
               loss[n][d_1] = -input[n][c][d_1] * weight[c]

       loss = np.sum(loss)
       // print(loss)
       // -1.1

    Example 3:

    ::

       // weighted negative log likelihood loss, mean reduction
       N, C, d1 = 2, 3, 2
       input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
               [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
       target = [[2, 1], [0, 2]]
       weight = [0.2, 0.3, 0.1]
       loss = np.zeros((N, d1))
       weight_total = 0
       for n in range(N):
           for d_1 in range(d1):
               c = target[n][d_1]
               loss[n][d_1] = -input[n][c][d_1] * weight[c]
               weight_total = weight_total + weight[c]

       loss = np.sum(loss) / weight_total
       // print(loss)
       // -1.57

    Parameters
    ==========
    input
        Type T.
        Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).
    target
        Type Tind.
        Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element value
        shall be in range of [0, C). If ignore_index is specified, it may have a
        value outside [0, C) and the target values should either be in the range
        [0, C) or have the value ignore_index.
    weight
        Type T.
        Optional rescaling weight tensor. If given, it has to be a tensor of
        size C. Otherwise, it is treated as if having all ones.
    ignore_index
        Attribute.
        Specifies a target value that is ignored and does not contribute to the
        input gradient. It's an optional value.
    reduction
        Attribute.
        Type of reduction to apply to loss: none, sum, mean (default). 'none':
        the output is the loss for each sample. 'sum': the output will be
        summed. 'mean': the sum of the output will be divided by the sum of
        applied weights.

    Returns
    =======
    loss : Var
        Type T.
        The negative log likelihood loss

    Notes
    =====
    Signature: ``ai.onnx@13::NegativeLogLikelihoodLoss``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - Tind: `tensor(int32)`, `tensor(int64)`
    """
    return _NegativeLogLikelihoodLoss(
        _NegativeLogLikelihoodLoss.Attributes(
            ignore_index=AttrInt64.maybe(ignore_index, name="ignore_index"),
            reduction=AttrString(reduction, name="reduction"),
        ),
        _NegativeLogLikelihoodLoss.Inputs(
            input=input,
            target=target,
            weight=weight,
        ),
    ).outputs.loss


def non_max_suppression(
    boxes: Var,
    scores: Var,
    max_output_boxes_per_class: Optional[Var] = None,
    iou_threshold: Optional[Var] = None,
    score_threshold: Optional[Var] = None,
    *,
    center_point_box: int = 0,
) -> Var:
    r"""
    Filter out boxes that have high intersection-over-union (IOU) overlap
    with previously selected boxes. Bounding boxes with score less than
    score_threshold are removed. Bounding box format is indicated by
    attribute center_point_box. Note that this algorithm is agnostic to
    where the origin is in the coordinate system and more generally is
    invariant to orthogonal transformations and translations of the
    coordinate system; thus translating or reflections of the coordinate
    system result in the same boxes being selected by the algorithm. The
    selected_indices output is a set of integers indexing into the input
    collection of bounding boxes representing the selected boxes. The
    bounding box coordinates corresponding to the selected indices can then
    be obtained using the Gather or GatherND operation.

    Parameters
    ==========
    boxes
        Type tensor(float).
        An input tensor with shape [num_batches, spatial_dimension, 4]. The
        single box data format is indicated by center_point_box.
    scores
        Type tensor(float).
        An input tensor with shape [num_batches, num_classes, spatial_dimension]
    max_output_boxes_per_class
        Type tensor(int64).
        Integer representing the maximum number of boxes to be selected per
        batch per class. It is a scalar. Default to 0, which means no output.
    iou_threshold
        Type tensor(float).
        Float representing the threshold for deciding whether boxes overlap too
        much with respect to IOU. It is scalar. Value range [0, 1]. Default to
        0.
    score_threshold
        Type tensor(float).
        Float representing the threshold for deciding when to remove boxes based
        on score. It is a scalar.
    center_point_box
        Attribute.
        Integer indicate the format of the box data. The default is 0. 0 - the
        box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are
        the coordinates of any diagonal pair of box corners and the coordinates
        can be provided as normalized (i.e., lying in the interval [0, 1]) or
        absolute. Mostly used for TF models. 1 - the box data is supplied as
        [x_center, y_center, width, height]. Mostly used for Pytorch models.

    Returns
    =======
    selected_indices : Var
        Type tensor(int64).
        selected indices from the boxes tensor. [num_selected_indices, 3], the
        selected index format is [batch_index, class_index, box_index].

    Notes
    =====
    Signature: ``ai.onnx@11::NonMaxSuppression``.

    """
    return _NonMaxSuppression(
        _NonMaxSuppression.Attributes(
            center_point_box=AttrInt64(center_point_box, name="center_point_box"),
        ),
        _NonMaxSuppression.Inputs(
            boxes=boxes,
            scores=scores,
            max_output_boxes_per_class=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        ),
    ).outputs.selected_indices


def non_zero(
    X: Var,
) -> Var:
    r"""
    Returns the indices of the elements that are non-zero (in row-major
    order - by dimension). NonZero behaves similar to numpy.nonzero:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
    but for scalar input, NonZero produces output shape (0, N) instead of
    (1, N), which is different from Numpy's behavior.

    Parameters
    ==========
    X
        Type T.
        input

    Returns
    =======
    Y : Var
        Type tensor(int64).
        output

    Notes
    =====
    Signature: ``ai.onnx@13::NonZero``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _NonZero(
        _NonZero.Attributes(),
        _NonZero.Inputs(
            X=X,
        ),
    ).outputs.Y


def not_(
    X: Var,
) -> Var:
    r"""
    Returns the negation of the input tensor element-wise.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@1::Not``.

    Type constraints:
     - T: `tensor(bool)`
    """
    return _Not(
        _Not.Attributes(),
        _Not.Inputs(
            X=X,
        ),
    ).outputs.Y


def one_hot(
    indices: Var,
    depth: Var,
    values: Var,
    *,
    axis: int = -1,
) -> Var:
    r"""
    Produces a one-hot tensor based on inputs. The locations represented by
    the index values in the 'indices' input tensor will have 'on_value' and
    the other locations will have 'off_value' in the output tensor, where
    'on_value' and 'off_value' are specified as part of required input
    argument 'values', which is a two-element tensor of format [off_value,
    on_value]. The rank of the output tensor will be one greater than the
    rank of the input tensor. The additional dimension is for one-hot
    representation. The additional dimension will be inserted at the
    position specified by 'axis'. If 'axis' is not specified then then
    additional dimension will be inserted as the innermost dimension, i.e.
    axis=-1. The size of the additional dimension is specified by required
    scalar input 'depth'. The type of the output tensor is the same as the
    type of the 'values' input. Any entries in the 'indices' input tensor
    with values outside the range [-depth, depth-1] will result in one-hot
    representation with all 'off_value' values in the output tensor.

    ::

       when axis = 0:
       output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

       when axis = -1:
       output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.

    Parameters
    ==========
    indices
        Type T1.
        Input tensor containing indices. Any entries in the 'indices' input
        tensor with values outside the range [-depth, depth-1] will result in
        one-hot representation with all 'off_value' values in the output
        tensor.In case 'indices' is of non-integer type, the values will be
        casted to int64 before use.
    depth
        Type T2.
        Scalar or Rank 1 tensor containing exactly one element, specifying the
        number of classes in one-hot tensor. This is also the size of the
        one-hot dimension (specified by 'axis' attribute) added on in the output
        tensor. The values in the 'indices' input tensor are expected to be in
        the range [-depth, depth-1]. In case 'depth' is of non-integer type, it
        will be casted to int64 before use.
    values
        Type T3.
        Rank 1 tensor containing exactly two elements, in the format [off_value,
        on_value], where 'on_value' is the value used for filling locations
        specified in 'indices' input tensor, and 'off_value' is the value used
        for filling locations other than those specified in 'indices' input
        tensor.
    axis
        Attribute.
        (Optional) Axis along which one-hot representation in added. Default:
        axis=-1. axis=-1 means that the additional dimension will be inserted as
        the innermost/last dimension in the output tensor. Negative value means
        counting dimensions from the back. Accepted range is [-r-1, r] where r =
        rank(indices).

    Returns
    =======
    output : Var
        Type T3.
        Tensor of rank one greater than input tensor 'indices', i.e.
        rank(output) = rank(indices) + 1. The data type for the elements of the
        output tensor is the same as the type of input 'values' is used.

    Notes
    =====
    Signature: ``ai.onnx@11::OneHot``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T3: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _OneHot(
        _OneHot.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _OneHot.Inputs(
            indices=indices,
            depth=depth,
            values=values,
        ),
    ).outputs.output


def optional(
    input: Optional[Var] = None,
    *,
    type: Optional[Type] = None,
) -> Var:
    r"""
    Constructs an optional-type value containing either an empty optional of
    a certain type specified by the attribute, or a non-empty value
    containing the input element.

    Parameters
    ==========
    input
        Type V.
        The input element.
    type
        Attribute.
        Type of the element in the optional output

    Returns
    =======
    output : Var
        Type O.
        The optional output enclosing the input element.

    Notes
    =====
    Signature: ``ai.onnx@15::Optional``.

    Type constraints:
     - V: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - O: `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`
    """
    return _Optional(
        _Optional.Attributes(
            type=AttrType.maybe(type, name="type"),
        ),
        _Optional.Inputs(
            input=input,
        ),
    ).outputs.output


def optional_get_element(
    input: Var,
) -> Var:
    r"""
    Outputs the element in the optional-type input. It is an error if the
    input value does not have an element and the behavior is undefined in
    this case.

    Parameters
    ==========
    input
        Type O.
        The optional input.

    Returns
    =======
    output : Var
        Type V.
        Output element in the optional input.

    Notes
    =====
    Signature: ``ai.onnx@15::OptionalGetElement``.

    Type constraints:
     - O: `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`
     - V: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _OptionalGetElement(
        _OptionalGetElement.Attributes(),
        _OptionalGetElement.Inputs(
            input=input,
        ),
    ).outputs.output


def optional_has_element(
    input: Var,
) -> Var:
    r"""
    Returns true if the optional-type input contains an element. If it is an
    empty optional-type, this op returns false.

    Parameters
    ==========
    input
        Type O.
        The optional input.

    Returns
    =======
    output : Var
        Type B.
        A scalar boolean tensor. If true, it indicates that optional-type input
        contains an element. Otherwise, it is empty.

    Notes
    =====
    Signature: ``ai.onnx@15::OptionalHasElement``.

    Type constraints:
     - O: `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`
     - B: `tensor(bool)`
    """
    return _OptionalHasElement(
        _OptionalHasElement.Attributes(),
        _OptionalHasElement.Inputs(
            input=input,
        ),
    ).outputs.output


def or_(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulted from performing the ``or`` logical operation
    elementwise on the input tensors ``A`` and ``B`` (with Numpy-style
    broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the logical operator.
    B
        Type T.
        Second input operand for the logical operator.

    Returns
    =======
    C : Var
        Type T1.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@7::Or``.

    Type constraints:
     - T: `tensor(bool)`
     - T1: `tensor(bool)`
    """
    return _Or(
        _Or.Attributes(),
        _Or.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def prelu(
    X: Var,
    slope: Var,
) -> Var:
    r"""
    PRelu takes input data (Tensor<T>) and slope tensor as input, and
    produces one output data (Tensor<T>) where the function
    ``f(x) = slope * x for x < 0``, ``f(x) = x for x >= 0``., is applied to
    the data tensor elementwise. This operator supports **unidirectional
    broadcasting** (tensor slope should be unidirectional broadcastable to
    input tensor X); for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    X
        Type T.
        Input tensor
    slope
        Type T.
        Slope tensor. The shape of slope can be smaller than first input X; if
        so, its shape must be unidirectional broadcastable to X

    Returns
    =======
    Y : Var
        Type T.
        Output tensor (same size as X)

    Notes
    =====
    Signature: ``ai.onnx@16::PRelu``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _PRelu(
        _PRelu.Attributes(),
        _PRelu.Inputs(
            X=X,
            slope=slope,
        ),
    ).outputs.Y


def pad(
    data: Var,
    pads: Var,
    constant_value: Optional[Var] = None,
    *,
    mode: str = "constant",
) -> Var:
    r"""
    Given a tensor containing the data to be padded (``data``), a tensor
    containing the number of start and end pad values for axis (``pads``),
    (optionally) a ``mode``, and (optionally) ``constant_value``, a padded
    tensor (``output``) is generated.

    The three supported ``modes`` are (similar to corresponding modes
    supported by ``numpy.pad``):

    1) ``constant``\ (default) - pads with a given constant value as
       specified by ``constant_value`` (which defaults to 0, empty string,
       or False)

    2) ``reflect`` - pads with the reflection of the vector mirrored on the
       first and last values of the vector along each axis

    3) ``edge`` - pads with the edge values of array

    Example 1 (``constant`` mode): Insert 0 pads to the beginning of the
    second dimension.

    data = [ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7], ]

    pads = [0, 2, 0, 0]

    mode = 'constant'

    constant_value = 0.0

    output = [ [0.0, 0.0, 1.0, 1.2], [0.0, 0.0, 2.3, 3.4], [0.0, 0.0, 4.5,
    5.7], ]

    Example 2 (``reflect`` mode): data = [ [1.0, 1.2], [2.3, 3.4], [4.5,
    5.7], ]

    pads = [0, 2, 0, 0]

    mode = 'reflect'

    output = [ [1.0, 1.2, 1.0, 1.2], [2.3, 3.4, 2.3, 3.4], [4.5, 5.7, 4.5,
    5.7], ]

    Example 3 (``edge`` mode): data = [ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7],
    ]

    pads = [0, 2, 0, 0]

    mode = 'edge'

    output = [ [1.0, 1.0, 1.0, 1.2], [2.3, 2.3, 2.3, 3.4], [4.5, 4.5, 4.5,
    5.7], ]

    Parameters
    ==========
    data
        Type T.
        Input tensor.
    pads
        Type tensor(int64).
        Tensor of integers indicating the number of padding elements to add or
        remove (if negative) at the beginning and end of each axis. For 2D input
        tensor, it is the number of pixels. ``pads`` should be a 1D tensor of
        shape [2 \* input_rank]. ``pads`` format should be: [x1_begin,
        x2_begin,...,x1_end, x2_end,...], where xi_begin is the number of pad
        values added at the beginning of axis ``i`` and xi_end, the number of
        pad values added at the end of axis ``i``.
    constant_value
        Type T.
        (Optional) A scalar value to be used if the mode chosen is ``constant``
        (by default it is 0, empty string or False).
    mode
        Attribute.
        Supported modes: ``constant``\ (default), ``reflect``, ``edge``

    Returns
    =======
    output : Var
        Type T.
        Tensor after padding.

    Notes
    =====
    Signature: ``ai.onnx@13::Pad``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Pad(
        _Pad.Attributes(
            mode=AttrString(mode, name="mode"),
        ),
        _Pad.Inputs(
            data=data,
            pads=pads,
            constant_value=constant_value,
        ),
    ).outputs.output


def pow(
    X: Var,
    Y: Var,
) -> Var:
    r"""
    Pow takes input data (Tensor<T>) and exponent Tensor, and produces one
    output data (Tensor<T>) where the function ``f(x) = x^exponent``, is
    applied to the data tensor elementwise. This operator supports
    **multidirectional (i.e., Numpy-style) broadcasting**; for more details
    please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    X
        Type T.
        First operand, base of the exponent.
    Y
        Type T1.
        Second operand, power of the exponent.

    Returns
    =======
    Z : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@15::Pow``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Pow(
        _Pow.Attributes(),
        _Pow.Inputs(
            X=X,
            Y=Y,
        ),
    ).outputs.Z


def qlinear_conv(
    x: Var,
    x_scale: Var,
    x_zero_point: Var,
    w: Var,
    w_scale: Var,
    w_zero_point: Var,
    y_scale: Var,
    y_zero_point: Var,
    B: Optional[Var] = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: Optional[Iterable[int]] = None,
    group: int = 1,
    kernel_shape: Optional[Iterable[int]] = None,
    pads: Optional[Iterable[int]] = None,
    strides: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    The convolution operator consumes a quantized input tensor, its scale
    and zero point, a quantized filter, its scale and zero point, and
    output's scale and zero point, and computes the quantized output. Each
    scale and zero-point pair must have same shape. It means they must be
    either scalars (per tensor) or 1-D tensors (per output channel). Each
    input or output and its related zero point must have same type. When
    bias is present it must be quantized using scale = input scale \* weight
    scale and zero point as 0.

    Parameters
    ==========
    x
        Type T1.
        Input data tensor from previous layer; has size (N x C x H x W), where N
        is the batch size, C is the number of channels, and H and W are the
        height and width. Note that this is for the 2D image. Otherwise the size
        is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in
        effect, the operation expects input data tensor to arrive with the
        dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE,
        DATA_FEATURE ...].
    x_scale
        Type tensor(float).
        Scale tensor for input 'x'. It's a scalar, which means a
        per-tensor/layer quantization.
    x_zero_point
        Type T1.
        Zero point tensor for input 'x'. It's a scalar, which means a
        per-tensor/layer quantization.
    w
        Type T2.
        The weight tensor that will be used in the convolutions; has size (M x
        C/group x kH x kW), where C is the number of channels, and kH and kW are
        the height and width of the kernel, and M is the number of feature maps.
        For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x
        k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel.
        Optionally, if dimension denotation is in effect, the operation expects
        the weight tensor to arrive with the dimension denotation of
        [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL
        ...]. X.shape[1] == (W.shape[1] \* group) == C (assuming zero based
        indices for the shape array). Or in other words FILTER_IN_CHANNEL should
        be equal to DATA_CHANNEL.
    w_scale
        Type tensor(float).
        Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which
        means a per-tensor/layer or per output channel quantization. If it's a
        1-D tensor, its number of elements should be equal to the number of
        output channels (M).
    w_zero_point
        Type T2.
        Zero point tensor for input 'w'. It could be a scalar or a 1-D tensor,
        which means a per-tensor/layer or per output channel quantization. If
        it's a 1-D tensor, its number of elements should be equal to the number
        of output channels (M).
    y_scale
        Type tensor(float).
        Scale tensor for output 'y'. It's a scalar, which means a
        per-tensor/layer quantization.
    y_zero_point
        Type T3.
        Zero point tensor for output 'y'. It's a scalar, which means a
        per-tensor/layer quantization.
    B
        Type T4.
        Optional 1D bias to be added to the convolution, has size of M. Bias
        must be quantized using scale = x_scale \* w_scale and zero_point = 0
    auto_pad
        Attribute.
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where
        default value is NOTSET, which means explicit padding is used.
        SAME_UPPER or SAME_LOWER mean pad the input so that
        ``output_shape[i] = ceil(input_shape[i] / strides[i])`` for each axis
        ``i``. The padding is split between the two sides equally or almost
        equally (depending on whether it is even or odd). In case the padding is
        an odd number, the extra padding is added at the end for SAME_UPPER and
        at the beginning for SAME_LOWER.
    dilations
        Attribute.
        dilation value along each spatial axis of the filter. If not present,
        the dilation defaults to 1 along each spatial axis.
    group
        Attribute.
        number of groups input channels and output channels are divided into.
        default is 1.
    kernel_shape
        Attribute.
        The shape of the convolution kernel. If not present, should be inferred
        from input 'w'.
    pads
        Attribute.
        Padding for the beginning and ending along each spatial axis, it can
        take any value greater than or equal to 0.The value represent the number
        of pixels added to the beginning and end part of the corresponding
        axis.\ ``pads`` format should be as follow [x1_begin, x2_begin...x1_end,
        x2_end,...], where xi_begin the number ofpixels added at the beginning
        of axis ``i`` and xi_end, the number of pixels added at the end of axis
        ``i``.This attribute cannot be used simultaneously with auto_pad
        attribute. If not present, the padding defaultsto 0 along start and end
        of each spatial axis.
    strides
        Attribute.
        Stride along each spatial axis. If not present, the stride defaults to 1
        along each spatial axis.

    Returns
    =======
    y : Var
        Type T3.
        Output data tensor that contains the result of the convolution. The
        output dimensions are functions of the kernel size, stride size, and pad
        lengths.

    Notes
    =====
    Signature: ``ai.onnx@10::QLinearConv``.

    Type constraints:
     - T1: `tensor(int8)`, `tensor(uint8)`
     - T2: `tensor(int8)`, `tensor(uint8)`
     - T3: `tensor(int8)`, `tensor(uint8)`
     - T4: `tensor(int32)`
    """
    return _QLinearConv(
        _QLinearConv.Attributes(
            auto_pad=AttrString(auto_pad, name="auto_pad"),
            dilations=AttrInt64s.maybe(dilations, name="dilations"),
            group=AttrInt64(group, name="group"),
            kernel_shape=AttrInt64s.maybe(kernel_shape, name="kernel_shape"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _QLinearConv.Inputs(
            x=x,
            x_scale=x_scale,
            x_zero_point=x_zero_point,
            w=w,
            w_scale=w_scale,
            w_zero_point=w_zero_point,
            y_scale=y_scale,
            y_zero_point=y_zero_point,
            B=B,
        ),
    ).outputs.y


def qlinear_matmul(
    a: Var,
    a_scale: Var,
    a_zero_point: Var,
    b: Var,
    b_scale: Var,
    b_zero_point: Var,
    y_scale: Var,
    y_zero_point: Var,
) -> Var:
    r"""
    Matrix product that behaves like numpy.matmul:
    https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
    It consumes two quantized input tensors, their scales and zero points,
    scale and zero point of output, and computes the quantized output. The
    quantization formula is y = saturate((x / y_scale) + y_zero_point). For
    (x / y_scale), it is rounding to nearest ties to even. Refer to
    https://en.wikipedia.org/wiki/Rounding for details. Scale and zero point
    must have same shape. They must be either scalar (per tensor) or N-D
    tensor (per row for 'a' and per column for 'b'). Scalar refers to per
    tensor quantization whereas N-D refers to per row or per column
    quantization. If the input is 2D of shape [M, K] then zero point and
    scale tensor may be an M element vector [v_1, v_2, ..., v_M] for per row
    quantization and K element vector of shape [v_1, v_2, ..., v_K] for per
    column quantization. If the input is N-D tensor with shape [D1, D2, M,
    K] then zero point and scale tensor may have shape [D1, D2, M, 1] for
    per row quantization and shape [D1, D2, 1, K] for per column
    quantization. Production must never overflow, and accumulation may
    overflow if and only if in 32 bits.

    Parameters
    ==========
    a
        Type T1.
        N-dimensional quantized matrix a
    a_scale
        Type tensor(float).
        scale of quantized input a
    a_zero_point
        Type T1.
        zero point of quantized input a
    b
        Type T2.
        N-dimensional quantized matrix b
    b_scale
        Type tensor(float).
        scale of quantized input b
    b_zero_point
        Type T2.
        zero point of quantized input b
    y_scale
        Type tensor(float).
        scale of quantized output y
    y_zero_point
        Type T3.
        zero point of quantized output y

    Returns
    =======
    y : Var
        Type T3.
        Quantized matrix multiply results from a \* b

    Notes
    =====
    Signature: ``ai.onnx@10::QLinearMatMul``.

    Type constraints:
     - T1: `tensor(int8)`, `tensor(uint8)`
     - T2: `tensor(int8)`, `tensor(uint8)`
     - T3: `tensor(int8)`, `tensor(uint8)`
    """
    return _QLinearMatMul(
        _QLinearMatMul.Attributes(),
        _QLinearMatMul.Inputs(
            a=a,
            a_scale=a_scale,
            a_zero_point=a_zero_point,
            b=b,
            b_scale=b_scale,
            b_zero_point=b_zero_point,
            y_scale=y_scale,
            y_zero_point=y_zero_point,
        ),
    ).outputs.y


def quantize_linear(
    x: Var,
    y_scale: Var,
    y_zero_point: Optional[Var] = None,
    *,
    axis: int = 1,
) -> Var:
    r"""
    The linear quantization operator. It consumes a high precision tensor, a
    scale, and a zero point to compute the low precision / quantized tensor.
    The scale factor and zero point must have same shape, and can be either
    a scalar for per-tensor / per layer quantization, or a 1-D tensor for
    per-axis quantization. The quantization formula is y = saturate ((x /
    y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if
    it's uint8, or [-128, 127] if it's int8. For (x / y_scale), it's
    rounding to the nearest even. Refer to
    https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and
    'y' must have same type.

    Parameters
    ==========
    x
        Type T1.
        N-D full precision Input tensor to be quantized.
    y_scale
        Type tensor(float).
        Scale for doing quantization to get 'y'. It can be a scalar, which means
        per-tensor/layer quantization, or a 1-D Tensor for per-axis
        quantization.
    y_zero_point
        Type T2.
        Zero point for doing quantization to get 'y'. Shape must match y_scale.
        Default is uint8 with zero point of 0 if it's not specified.
    axis
        Attribute.
        (Optional) The axis of the quantization dimension of the input tensor.
        Ignored for per-tensor quantization. Negative value means counting
        dimensions from the back. Accepted range is [-r, r-1] where r =
        rank(input).

    Returns
    =======
    y : Var
        Type T2.
        N-D quantized output tensor. It has same shape as input 'x'.

    Notes
    =====
    Signature: ``ai.onnx@13::QuantizeLinear``.

    Type constraints:
     - T1: `tensor(float)`, `tensor(int32)`
     - T2: `tensor(int8)`, `tensor(uint8)`
    """
    return _QuantizeLinear(
        _QuantizeLinear.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _QuantizeLinear.Inputs(
            x=x,
            y_scale=y_scale,
            y_zero_point=y_zero_point,
        ),
    ).outputs.y


def rnn(
    X: Var,
    W: Var,
    R: Var,
    B: Optional[Var] = None,
    sequence_lens: Optional[Var] = None,
    initial_h: Optional[Var] = None,
    *,
    activation_alpha: Optional[Iterable[float]] = None,
    activation_beta: Optional[Iterable[float]] = None,
    activations: Iterable[str] = ("Tanh", "Tanh"),
    clip: Optional[float] = None,
    direction: str = "forward",
    hidden_size: Optional[int] = None,
    layout: int = 0,
) -> Tuple[Var, Var]:
    r"""
    Computes an one-layer simple RNN. This operator is usually supported via
    some custom implementation such as CuDNN.

    Notations:

    -  ``X`` - input tensor
    -  ``i`` - input gate
    -  ``t`` - time step (t-1 means previous time step)
    -  ``Wi`` - W parameter weight matrix for input gate
    -  ``Ri`` - R recurrence weight matrix for input gate
    -  ``Wbi`` - W parameter bias vector for input gate
    -  ``Rbi`` - R parameter bias vector for input gate
    -  ``WBi`` - W parameter weight matrix for backward input gate
    -  ``RBi`` - R recurrence weight matrix for backward input gate
    -  ``WBbi`` - WR bias vectors for backward input gate
    -  ``RBbi`` - RR bias vectors for backward input gate
    -  ``H`` - Hidden state
    -  ``num_directions`` - 2 if direction == bidirectional else 1

    Activation functions:

    -  Relu(x) - max(0, x)
    -  Tanh(x) - (1 - e^{-2x})/(1 + e^{-2x})
    -  Sigmoid(x) - 1/(1 + e^{-x})

    NOTE: Below are optional

    -  Affine(x) - alpha*x + beta
    -  LeakyRelu(x) - x if x >= 0 else alpha \* x
    -  ThresholdedRelu(x) - x if x >= alpha else 0
    -  ScaledTanh(x) - alpha\ *Tanh(beta*\ x)
    -  HardSigmoid(x) - min(max(alpha*x + beta, 0), 1)
    -  Elu(x) - x if x >= 0 else alpha*(e^x - 1)
    -  Softsign(x) - x/(1 + \|x\|)
    -  Softplus(x) - log(1 + e^x)

    Equations (Default: f=Tanh):

    -  Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi) This operator has
       **optional** inputs/outputs. See `the
       doc <https://github.com/onnx/onnx/blob/main/docs/IR.md>`__ for more
       details about the representation of optional arguments. An empty
       string may be used in the place of an actual argument's name to
       indicate a missing argument. Trailing optional arguments (those not
       followed by an argument that is present) may also be simply omitted.

    Parameters
    ==========
    X
        Type T.
        The input sequences packed (and potentially padded) into one 3-D tensor
        with the shape of ``[seq_length, batch_size, input_size]``.
    W
        Type T.
        The weight tensor for input gate. Concatenation of ``Wi`` and ``WBi``
        (if bidirectional). The tensor has shape
        ``[num_directions, hidden_size, input_size]``.
    R
        Type T.
        The recurrence weight tensor. Concatenation of ``Ri`` and ``RBi`` (if
        bidirectional). The tensor has shape
        ``[num_directions, hidden_size, hidden_size]``.
    B
        Type T.
        The bias tensor for input gate. Concatenation of ``[Wbi, Rbi]`` and
        ``[WBbi, RBbi]`` (if bidirectional). The tensor has shape
        ``[num_directions, 2*hidden_size]``. Optional: If not specified -
        assumed to be 0.
    sequence_lens
        Type T1.
        Optional tensor specifying lengths of the sequences in a batch. If not
        specified - assumed all sequences in the batch to have length
        ``seq_length``. It has shape ``[batch_size]``.
    initial_h
        Type T.
        Optional initial value of the hidden. If not specified - assumed to be
        0. It has shape ``[num_directions, batch_size, hidden_size]``.
    activation_alpha
        Attribute.
        Optional scaling values used by some activation functions. The values
        are consumed in the order of activation functions, for example (f, g, h)
        in LSTM. Default values are the same as of corresponding ONNX
        operators.For example with LeakyRelu, the default alpha is 0.01.
    activation_beta
        Attribute.
        Optional scaling values used by some activation functions. The values
        are consumed in the order of activation functions, for example (f, g, h)
        in LSTM. Default values are the same as of corresponding ONNX operators.
    activations
        Attribute.
        One (or two if bidirectional) activation function for input gate. The
        activation function must be one of the activation functions specified
        above. Optional: Default ``Tanh`` if not specified.
    clip
        Attribute.
        Cell clip threshold. Clipping bounds the elements of a tensor in the
        range of [-threshold, +threshold] and is applied to the input of
        activations. No clip if not specified.
    direction
        Attribute.
        Specify if the RNN is forward, reverse, or bidirectional. Must be one of
        forward (default), reverse, or bidirectional.
    hidden_size
        Attribute.
        Number of neurons in the hidden layer
    layout
        Attribute.
        The shape format of inputs X, initial_h and outputs Y, Y_h. If 0, the
        following shapes are expected: X.shape = [seq_length, batch_size,
        input_size], Y.shape = [seq_length, num_directions, batch_size,
        hidden_size], initial_h.shape = Y_h.shape = [num_directions, batch_size,
        hidden_size]. If 1, the following shapes are expected: X.shape =
        [batch_size, seq_length, input_size], Y.shape = [batch_size, seq_length,
        num_directions, hidden_size], initial_h.shape = Y_h.shape = [batch_size,
        num_directions, hidden_size].

    Returns
    =======
    Y : Var
        Type T.
        A tensor that concats all the intermediate output values of the hidden.
        It has shape ``[seq_length, num_directions, batch_size, hidden_size]``.
    Y_h : Var
        Type T.
        The last output value of the hidden. It has shape
        ``[num_directions, batch_size, hidden_size]``.

    Notes
    =====
    Signature: ``ai.onnx@14::RNN``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T1: `tensor(int32)`
    """
    return _RNN(
        _RNN.Attributes(
            activation_alpha=AttrFloat32s.maybe(
                activation_alpha, name="activation_alpha"
            ),
            activation_beta=AttrFloat32s.maybe(activation_beta, name="activation_beta"),
            activations=AttrStrings(activations, name="activations"),
            clip=AttrFloat32.maybe(clip, name="clip"),
            direction=AttrString(direction, name="direction"),
            hidden_size=AttrInt64.maybe(hidden_size, name="hidden_size"),
            layout=AttrInt64(layout, name="layout"),
        ),
        _RNN.Inputs(
            X=X,
            W=W,
            R=R,
            B=B,
            sequence_lens=sequence_lens,
            initial_h=initial_h,
        ),
    ).outputs._unpack_to_any()


def random_normal(
    *,
    dtype: npt.DTypeLike = np.float32,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: Optional[float] = None,
    shape: Iterable[int],
) -> Var:
    r"""
    Generate a tensor with random values drawn from a normal distribution.
    The shape of the tensor is specified by the ``shape`` argument and the
    parameter of the normal distribution specified by ``mean`` and
    ``scale``.

    The data type is specified by the 'dtype' argument. The 'dtype' argument
    must be one of the data types specified in the 'DataType' enum field in
    the TensorProto message.

    Parameters
    ==========
    dtype
        Attribute.
        The data type for the elements of the output tensor. Default is
        TensorProto::FLOAT.
    mean
        Attribute.
        The mean of the normal distribution.
    scale
        Attribute.
        The standard deviation of the normal distribution.
    seed
        Attribute.
        (Optional) Seed to the random generator, if not specified we will auto
        generate one.
    shape
        Attribute.
        The shape of the output tensor.

    Returns
    =======
    output : Var
        Type T.
        Output tensor of random values drawn from normal distribution

    Notes
    =====
    Signature: ``ai.onnx@1::RandomNormal``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _RandomNormal(
        _RandomNormal.Attributes(
            dtype=AttrDtype(dtype, name="dtype"),
            mean=AttrFloat32(mean, name="mean"),
            scale=AttrFloat32(scale, name="scale"),
            seed=AttrFloat32.maybe(seed, name="seed"),
            shape=AttrInt64s(shape, name="shape"),
        ),
        _RandomNormal.Inputs(),
    ).outputs.output


def random_normal_like(
    input: Var,
    *,
    dtype: Optional[npt.DTypeLike] = None,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: Optional[float] = None,
) -> Var:
    r"""
    Generate a tensor with random values drawn from a normal distribution.
    The shape of the output tensor is copied from the shape of the input
    tensor, and the parameters of the normal distribution are specified by
    ``mean`` and ``scale``.

    The data type is specified by the 'dtype' argument, or copied from the
    input tensor if not provided. The 'dtype' argument must be one of the
    data types specified in the 'DataType' enum field in the TensorProto
    message, and be valid as an output type.

    Parameters
    ==========
    input
        Type T1.
        Input tensor to copy shape and optionally type information from.
    dtype
        Attribute.
        (Optional) The data type for the elements of the output tensor, if not
        specified, we will use the data type of the input tensor.
    mean
        Attribute.
        The mean of the normal distribution.
    scale
        Attribute.
        The standard deviation of the normal distribution.
    seed
        Attribute.
        (Optional) Seed to the random generator, if not specified we will auto
        generate one.

    Returns
    =======
    output : Var
        Type T2.
        Output tensor of random values drawn from normal distribution

    Notes
    =====
    Signature: ``ai.onnx@1::RandomNormalLike``.

    Type constraints:
     - T1: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _RandomNormalLike(
        _RandomNormalLike.Attributes(
            dtype=AttrDtype.maybe(dtype, name="dtype"),
            mean=AttrFloat32(mean, name="mean"),
            scale=AttrFloat32(scale, name="scale"),
            seed=AttrFloat32.maybe(seed, name="seed"),
        ),
        _RandomNormalLike.Inputs(
            input=input,
        ),
    ).outputs.output


def random_uniform(
    *,
    dtype: npt.DTypeLike = np.float32,
    high: float = 1.0,
    low: float = 0.0,
    seed: Optional[float] = None,
    shape: Iterable[int],
) -> Var:
    r"""
    Generate a tensor with random values drawn from a uniform distribution.
    The shape of the tensor is specified by the ``shape`` argument and the
    range by ``low`` and ``high``.

    The data type is specified by the 'dtype' argument. The 'dtype' argument
    must be one of the data types specified in the 'DataType' enum field in
    the TensorProto message.

    Parameters
    ==========
    dtype
        Attribute.
        The data type for the elements of the output tensor. If not specified,
        default is TensorProto::FLOAT.
    high
        Attribute.
        Upper boundary of the output values.
    low
        Attribute.
        Lower boundary of the output values.
    seed
        Attribute.
        (Optional) Seed to the random generator, if not specified we will auto
        generate one.
    shape
        Attribute.
        The shape of the output tensor.

    Returns
    =======
    output : Var
        Type T.
        Output tensor of random values drawn from uniform distribution

    Notes
    =====
    Signature: ``ai.onnx@1::RandomUniform``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _RandomUniform(
        _RandomUniform.Attributes(
            dtype=AttrDtype(dtype, name="dtype"),
            high=AttrFloat32(high, name="high"),
            low=AttrFloat32(low, name="low"),
            seed=AttrFloat32.maybe(seed, name="seed"),
            shape=AttrInt64s(shape, name="shape"),
        ),
        _RandomUniform.Inputs(),
    ).outputs.output


def random_uniform_like(
    input: Var,
    *,
    dtype: Optional[npt.DTypeLike] = None,
    high: float = 1.0,
    low: float = 0.0,
    seed: Optional[float] = None,
) -> Var:
    r"""
    Generate a tensor with random values drawn from a uniform distribution.
    The shape of the output tensor is copied from the shape of the input
    tensor, and the parameters of the uniform distribution are specified by
    ``low`` and ``high``.

    The data type is specified by the 'dtype' argument, or copied from the
    input tensor if not provided. The 'dtype' argument must be one of the
    data types specified in the 'DataType' enum field in the TensorProto
    message and be valid as an output type.

    Parameters
    ==========
    input
        Type T1.
        Input tensor to copy shape and optionally type information from.
    dtype
        Attribute.
        (Optional) The data type for the elements of the output tensor, if not
        specified, we will use the data type of the input tensor.
    high
        Attribute.
        Upper boundary of the output values.
    low
        Attribute.
        Lower boundary of the output values.
    seed
        Attribute.
        (Optional) Seed to the random generator, if not specified we will auto
        generate one.

    Returns
    =======
    output : Var
        Type T2.
        Output tensor of random values drawn from uniform distribution

    Notes
    =====
    Signature: ``ai.onnx@1::RandomUniformLike``.

    Type constraints:
     - T1: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _RandomUniformLike(
        _RandomUniformLike.Attributes(
            dtype=AttrDtype.maybe(dtype, name="dtype"),
            high=AttrFloat32(high, name="high"),
            low=AttrFloat32(low, name="low"),
            seed=AttrFloat32.maybe(seed, name="seed"),
        ),
        _RandomUniformLike.Inputs(
            input=input,
        ),
    ).outputs.output


def range(
    start: Var,
    limit: Var,
    delta: Var,
) -> Var:
    r"""
    Generate a tensor containing a sequence of numbers that begin at
    ``start`` and extends by increments of ``delta`` up to ``limit``
    (exclusive).

    The number of elements in the output of range is computed as below:

    ::

       number_of_elements = max( ceil( (limit - start) / delta ) , 0 )

    The pseudocode determining the contents of the output is shown below:

    ::

       for(int i=0; i<number_of_elements; ++i) {
         output[i] =  start + (i * delta);
       }

    Example 1

    ::

       Inputs: start = 3, limit = 9, delta = 3
       Output: [3, 6]

    Example 2

    ::

       Inputs: start = 10, limit = 4, delta = -2
       Output: [10, 8, 6]

    Parameters
    ==========
    start
        Type T.
        Scalar. First entry for the range of output values.
    limit
        Type T.
        Scalar. Exclusive upper limit for the range of output values.
    delta
        Type T.
        Scalar. Value to step by.

    Returns
    =======
    output : Var
        Type T.
        A 1-D tensor with same type as the inputs containing generated range of
        values.

    Notes
    =====
    Signature: ``ai.onnx@11::Range``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`
    """
    return _Range(
        _Range.Attributes(),
        _Range.Inputs(
            start=start,
            limit=limit,
            delta=delta,
        ),
    ).outputs.output


def reciprocal(
    X: Var,
) -> Var:
    r"""
    Reciprocal takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the reciprocal is, y = 1/x, is applied to the tensor
    elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Reciprocal``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Reciprocal(
        _Reciprocal.Attributes(),
        _Reciprocal.Inputs(
            X=X,
        ),
    ).outputs.Y


def reduce_l1(
    data: Var,
    *,
    axes: Optional[Iterable[int]] = None,
    keepdims: int = 1,
) -> Var:
    r"""
    Computes the L1 norm of the input tensor's elements along the provided
    axes. The resulting tensor has the same rank as the input if
    ``keepdims`` equals 1. If ``keepdims`` equals 0, then the resulting
    tensor has the reduced dimension pruned. Input tensors of rank zero are
    valid. Reduction over an empty set of values yields 0.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to reduce over
        all the dimensions of the input tensor. Accepted range is [-r, r-1]
        where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceL1``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceL1(
        _ReduceL1.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _ReduceL1.Inputs(
            data=data,
        ),
    ).outputs.reduced


def reduce_l2(
    data: Var,
    *,
    axes: Optional[Iterable[int]] = None,
    keepdims: int = 1,
) -> Var:
    r"""
    Computes the L2 norm of the input tensor's elements along the provided
    axes. The resulting tensor has the same rank as the input if
    ``keepdims`` equals 1. If ``keepdims`` equals 0, then the resulting
    tensor has the reduced dimension pruned. Input tensors of rank zero are
    valid. Reduction over an empty set of values yields 0.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to reduce over
        all the dimensions of the input tensor. Accepted range is [-r, r-1]
        where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceL2``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceL2(
        _ReduceL2.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _ReduceL2.Inputs(
            data=data,
        ),
    ).outputs.reduced


def reduce_log_sum(
    data: Var,
    *,
    axes: Optional[Iterable[int]] = None,
    keepdims: int = 1,
) -> Var:
    r"""
    Computes the log sum of the input tensor's elements along the provided
    axes. The resulting tensor has the same rank as the input if
    ``keepdims`` equals 1. If ``keepdims`` equals 0, then the resulting
    tensor has the reduced dimension pruned. Input tensors of rank zero are
    valid. Reduction over an empty set of values yields minus infinity (if
    supported by the datatype) or undefined otherwise.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to reduce over
        all the dimensions of the input tensor. Accepted range is [-r, r-1]
        where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceLogSum``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceLogSum(
        _ReduceLogSum.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _ReduceLogSum.Inputs(
            data=data,
        ),
    ).outputs.reduced


def reduce_log_sum_exp(
    data: Var,
    *,
    axes: Optional[Iterable[int]] = None,
    keepdims: int = 1,
) -> Var:
    r"""
    Computes the log sum exponent of the input tensor's elements along the
    provided axes. The resulting tensor has the same rank as the input if
    ``keepdims`` equals 1. If ``keepdims`` equals 0, then the resulting
    tensor has the reduced dimension pruned. Input tensors of rank zero are
    valid. Reduction over an empty set of values yields minus infinity (if
    supported by the datatype) or undefined otherwise.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to reduce over
        all the dimensions of the input tensor. Accepted range is [-r, r-1]
        where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceLogSumExp``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceLogSumExp(
        _ReduceLogSumExp.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _ReduceLogSumExp.Inputs(
            data=data,
        ),
    ).outputs.reduced


def reduce_max(
    data: Var,
    *,
    axes: Optional[Iterable[int]] = None,
    keepdims: int = 1,
) -> Var:
    r"""
    Computes the max of the input tensor's elements along the provided axes.
    The resulting tensor has the same rank as the input if ``keepdims``
    equals 1. If ``keepdims`` equals 0, then the resulting tensor has the
    reduced dimension pruned. Input tensors of rank zero are valid.
    Reduction over an empty set of values yields minus infinity (if
    supported by the datatype) or the minimum value of the data type
    otherwise.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to reduce over
        all the dimensions of the input tensor. Accepted range is [-r, r-1]
        where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceMax``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ReduceMax(
        _ReduceMax.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _ReduceMax.Inputs(
            data=data,
        ),
    ).outputs.reduced


def reduce_mean(
    data: Var,
    *,
    axes: Optional[Iterable[int]] = None,
    keepdims: int = 1,
) -> Var:
    r"""
    Computes the mean of the input tensor's elements along the provided
    axes. The resulting tensor has the same rank as the input if
    ``keepdims`` equals 1. If ``keepdims`` equals 0, then the resulting
    tensor has the reduced dimension pruned. Input tensors of rank zero are
    valid. Reduction over an empty set of values yields undefined.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to reduce over
        all the dimensions of the input tensor. Accepted range is [-r, r-1]
        where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceMean``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceMean(
        _ReduceMean.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _ReduceMean.Inputs(
            data=data,
        ),
    ).outputs.reduced


def reduce_min(
    data: Var,
    *,
    axes: Optional[Iterable[int]] = None,
    keepdims: int = 1,
) -> Var:
    r"""
    Computes the min of the input tensor's elements along the provided axes.
    The resulting tensor has the same rank as the input if ``keepdims``
    equals 1. If ``keepdims`` equals 0, then the resulting tensor has the
    reduced dimension pruned. Input tensors of rank zero are valid.
    Reduction over an empty set of values yields plus infinity (if supported
    by the datatype) or the maximum value of the data type otherwise.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to reduce over
        all the dimensions of the input tensor. Accepted range is [-r, r-1]
        where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceMin``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ReduceMin(
        _ReduceMin.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _ReduceMin.Inputs(
            data=data,
        ),
    ).outputs.reduced


def reduce_prod(
    data: Var,
    *,
    axes: Optional[Iterable[int]] = None,
    keepdims: int = 1,
) -> Var:
    r"""
    Computes the product of the input tensor's elements along the provided
    axes. The resulting tensor has the same rank as the input if
    ``keepdims`` equals 1. If ``keepdims`` equals 0, then the resulting
    tensor has the reduced dimension pruned. Input tensors of rank zero are
    valid. Reduction over an empty set of values yields 1.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to reduce over
        all the dimensions of the input tensor. Accepted range is [-r, r-1]
        where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceProd``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceProd(
        _ReduceProd.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _ReduceProd.Inputs(
            data=data,
        ),
    ).outputs.reduced


def reduce_sum(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> Var:
    r"""
    Computes the sum of the input tensor's elements along the provided axes.
    The resulting tensor has the same rank as the input if ``keepdims``
    equals 1. If ``keepdims`` equals 0, then the resulting tensor has the
    reduced dimension pruned. Input tensors of rank zero are valid.
    Reduction over an empty set of values yields 0.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Type tensor(int64).
        Optional input list of integers, along which to reduce. The default is
        to reduce over all the dimensions of the input tensor if
        'noop_with_empty_axes' is false, else act as an Identity op when
        'noop_with_empty_axes' is true. Accepted range is [-r, r-1] where r =
        rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.
    noop_with_empty_axes
        Attribute.
        Defines behavior if 'axes' is empty. Default behavior with 'false' is to
        reduce all axes. When axes is empty and this attribute is set to true,
        input tensor will not be reduced,and the output tensor would be
        equivalent to input tensor.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceSum``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceSum(
        _ReduceSum.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceSum.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def reduce_sum_square(
    data: Var,
    *,
    axes: Optional[Iterable[int]] = None,
    keepdims: int = 1,
) -> Var:
    r"""
    Computes the sum square of the input tensor's elements along the
    provided axes. The resulting tensor has the same rank as the input if
    ``keepdims`` equals 1. If ``keepdims`` equals 0, then the resulting
    tensor has the reduced dimension pruned. Input tensors of rank zero are
    valid. Reduction over an empty set of values yields 0.

    The above behavior is similar to numpy, with the exception that numpy
    defaults ``keepdims`` to ``False`` instead of ``True``.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    axes
        Attribute.
        A list of integers, along which to reduce. The default is to reduce over
        all the dimensions of the input tensor. Accepted range is [-r, r-1]
        where r = rank(data).
    keepdims
        Attribute.
        Keep the reduced dimension or not, default 1 means keep reduced
        dimension.

    Returns
    =======
    reduced : Var
        Type T.
        Reduced output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::ReduceSumSquare``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceSumSquare(
        _ReduceSumSquare.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _ReduceSumSquare.Inputs(
            data=data,
        ),
    ).outputs.reduced


def relu(
    X: Var,
) -> Var:
    r"""
    Relu takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the rectified linear function, y = max(0, x), is
    applied to the tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@14::Relu``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`
    """
    return _Relu(
        _Relu.Attributes(),
        _Relu.Inputs(
            X=X,
        ),
    ).outputs.Y


def reshape(
    data: Var,
    shape: Var,
    *,
    allowzero: int = 0,
) -> Var:
    r"""
    Reshape the input tensor similar to numpy.reshape. First input is the
    data tensor, second input is a shape tensor which specifies the output
    shape. It outputs the reshaped tensor. At most one dimension of the new
    shape can be -1. In this case, the value is inferred from the size of
    the tensor and the remaining dimensions. A dimension could also be 0, in
    which case the actual dimension value is unchanged (i.e. taken from the
    input tensor). If 'allowzero' is set, and the new shape includes 0, the
    dimension will be set explicitly to zero (i.e. not taken from input
    tensor). Shape (second input) could be an empty shape, which means
    converting to a scalar. The input tensor's shape and the output tensor's
    shape are required to have the same number of elements.

    If the attribute 'allowzero' is set, it is invalid for the specified
    shape to contain both a zero value and -1, as the value of the dimension
    corresponding to -1 cannot be determined uniquely.

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    shape
        Type tensor(int64).
        Specified shape for output.
    allowzero
        Attribute.
        (Optional) By default, when any value in the 'shape' input is equal to
        zero the corresponding dimension value is copied from the input tensor
        dynamically. allowzero=1 indicates that if any value in the 'shape'
        input is set to zero, the zero value is honored, similar to NumPy.

    Returns
    =======
    reshaped : Var
        Type T.
        Reshaped data.

    Notes
    =====
    Signature: ``ai.onnx@14::Reshape``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Reshape(
        _Reshape.Attributes(
            allowzero=AttrInt64(allowzero, name="allowzero"),
        ),
        _Reshape.Inputs(
            data=data,
            shape=shape,
        ),
    ).outputs.reshaped


def resize(
    X: Var,
    roi: Optional[Var] = None,
    scales: Optional[Var] = None,
    sizes: Optional[Var] = None,
    *,
    coordinate_transformation_mode: str = "half_pixel",
    cubic_coeff_a: float = -0.75,
    exclude_outside: int = 0,
    extrapolation_value: float = 0.0,
    mode: str = "nearest",
    nearest_mode: str = "round_prefer_floor",
) -> Var:
    r"""
    Resize the input tensor. In general, it calculates every value in the
    output tensor as a weighted average of neighborhood (a.k.a. sampling
    locations) in the input tensor. Each dimension value of the output
    tensor is: output_dimension = floor(input_dimension \* (roi_end -
    roi_start) \* scale) if input "sizes" is not specified.

    Parameters
    ==========
    X
        Type T1.
        N-D tensor
    roi
        Type T2.
        1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is
        the rank of X. The RoIs' coordinates are normalized in the coordinate
        system of the input image. It only takes effect when
        coordinate_transformation_mode is "tf_crop_and_resize"
    scales
        Type tensor(float).
        The scale array along each dimension. It takes value greater than 0. If
        it's less than 1, it's sampling down, otherwise, it's upsampling. The
        number of elements of 'scales' should be the same as the rank of input
        'X'. One of 'scales' and 'sizes' MUST be specified and it is an error if
        both are specified. If 'sizes' is needed, the user can use an empty
        string as the name of 'scales' in this operator's input list.
    sizes
        Type tensor(int64).
        The size of the output tensor. The number of elements of 'sizes' should
        be the same as the rank of input 'X'. Only one of 'scales' and 'sizes'
        can be specified.
    coordinate_transformation_mode
        Attribute.
        This attribute describes how to transform the coordinate in the resized
        tensor to the coordinate in the original tensor.

        The coordinate of each dimension is transformed individually. Let's
        describe a case using axis x as an example. Denote x_resized as the
        coordinate of axis x in the resized tensor, x_original as the coordinate
        of axis x in the original tensor, length_original as the length of the
        original tensor in axis x, length_resized as the length of the resized
        tensor in axis x, roi_x = (start_x, end_x) of the axis x in input "roi",
        scale = length_resized / length_original,

        if coordinate_transformation_mode is "half_pixel", x_original =
        (x_resized + 0.5) / scale - 0.5,

        if coordinate_transformation_mode is "pytorch_half_pixel", x_original =
        length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0,

        if coordinate_transformation_mode is "align_corners", x_original =
        x_resized \* (length_original - 1) / (length_resized - 1),

        if coordinate_transformation_mode is "asymmetric", x_original =
        x_resized / scale,

        if coordinate_transformation_mode is "tf_crop_and_resize", x_original =
        length_resized > 1 ? start_x \* (length_original - 1) + x_resized \*
        (end_x - start_x) \* (length_original - 1) / (length_resized - 1) : 0.5
        \* (start_x + end_x) \* (length_original - 1).
    cubic_coeff_a
        Attribute.
        The coefficient 'a' used in cubic interpolation. Two common choice are
        -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check out
        Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the
        details. This attribute is valid only if "mode" is "cubic".
    exclude_outside
        Attribute.
        If set to 1, the weight of sampling locations outside the tensor will be
        set to 0 and the weight will be renormalized so that their sum is 1.0.
        The default value is 0.
    extrapolation_value
        Attribute.
        When coordinate_transformation_mode is "tf_crop_and_resize" and
        x_original is outside the range [0, length_original - 1], this value is
        used as the corresponding output value. Default is 0.0f.
    mode
        Attribute.
        Three interpolation modes: nearest (default), linear and cubic. The
        "linear" mode includes linear interpolation for 1D tensor and N-linear
        interpolation for N-D tensor (for example, bilinear interpolation for 2D
        tensor). The "cubic" mode includes cubic interpolation for 1D tensor and
        N-cubic interpolation for N-D tensor (for example, bicubic interpolation
        for 2D tensor).
    nearest_mode
        Attribute.
        Four modes: round_prefer_floor (default, as known as round half down),
        round_prefer_ceil (as known as round half up), floor, ceil. Only used by
        nearest interpolation. It indicates how to get "nearest" pixel in input
        tensor from x_original, so this attribute is valid only if "mode" is
        "nearest".

    Returns
    =======
    Y : Var
        Type T1.
        N-D tensor after resizing

    Notes
    =====
    Signature: ``ai.onnx@13::Resize``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Resize(
        _Resize.Attributes(
            coordinate_transformation_mode=AttrString(
                coordinate_transformation_mode, name="coordinate_transformation_mode"
            ),
            cubic_coeff_a=AttrFloat32(cubic_coeff_a, name="cubic_coeff_a"),
            exclude_outside=AttrInt64(exclude_outside, name="exclude_outside"),
            extrapolation_value=AttrFloat32(
                extrapolation_value, name="extrapolation_value"
            ),
            mode=AttrString(mode, name="mode"),
            nearest_mode=AttrString(nearest_mode, name="nearest_mode"),
        ),
        _Resize.Inputs(
            X=X,
            roi=roi,
            scales=scales,
            sizes=sizes,
        ),
    ).outputs.Y


def reverse_sequence(
    input: Var,
    sequence_lens: Var,
    *,
    batch_axis: int = 1,
    time_axis: int = 0,
) -> Var:
    r"""
    Reverse batch of sequences having different lengths specified by
    ``sequence_lens``.

    For each slice i iterating on batch axis, the operator reverses the
    first sequence_lens[i] elements on time axis, and copies elements whose
    index's beyond sequence_lens[i] to the output. So the output slice i
    contains reversed sequences on the first sequence_lens[i] elements, then
    have original values copied for the other elements.

    Example 1: input = [[0.0, 4.0, 8.0, 12.0], [1.0, 5.0, 9.0, 13.0], [2.0,
    6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0]] sequence_lens = [4, 3, 2, 1]
    time_axis = 0 batch_axis = 1

    output = [[3.0, 6.0, 9.0, 12.0], [2.0, 5.0, 8.0, 13.0], [1.0, 4.0, 10.0,
    14.0], [0.0, 7.0, 11.0, 15.0]]

    Example 2: input = [[0.0, 1.0, 2.0, 3.0 ], [4.0, 5.0, 6.0, 7.0 ], [8.0,
    9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]] sequence_lens = [1, 2, 3, 4]
    time_axis = 1 batch_axis = 0

    output = [[0.0, 1.0, 2.0, 3.0 ], [5.0, 4.0, 6.0, 7.0 ], [10.0, 9.0, 8.0,
    11.0], [15.0, 14.0, 13.0, 12.0]]

    Parameters
    ==========
    input
        Type T.
        Tensor of rank r >= 2.
    sequence_lens
        Type tensor(int64).
        Tensor specifying lengths of the sequences in a batch. It has shape
        ``[batch_size]``.
    batch_axis
        Attribute.
        (Optional) Specify which axis is batch axis. Must be one of 1 (default),
        or 0.
    time_axis
        Attribute.
        (Optional) Specify which axis is time axis. Must be one of 0 (default),
        or 1.

    Returns
    =======
    Y : Var
        Type T.
        Tensor with same shape of input.

    Notes
    =====
    Signature: ``ai.onnx@10::ReverseSequence``.

    Type constraints:
     - T: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ReverseSequence(
        _ReverseSequence.Attributes(
            batch_axis=AttrInt64(batch_axis, name="batch_axis"),
            time_axis=AttrInt64(time_axis, name="time_axis"),
        ),
        _ReverseSequence.Inputs(
            input=input,
            sequence_lens=sequence_lens,
        ),
    ).outputs.Y


def roi_align(
    X: Var,
    rois: Var,
    batch_indices: Var,
    *,
    coordinate_transformation_mode: str = "half_pixel",
    mode: str = "avg",
    output_height: int = 1,
    output_width: int = 1,
    sampling_ratio: int = 0,
    spatial_scale: float = 1.0,
) -> Var:
    r"""
    Region of Interest (RoI) align operation described in the `Mask R-CNN
    paper <https://arxiv.org/abs/1703.06870>`__. RoiAlign consumes an input
    tensor X and region of interests (rois) to apply pooling across each
    RoI; it produces a 4-D tensor of shape (num_rois, C, output_height,
    output_width).

    RoiAlign is proposed to avoid the misalignment by removing quantizations
    while converting from original image into feature map and from feature
    map into RoI feature; in each ROI bin, the value of the sampled
    locations are computed directly through bilinear interpolation.

    Parameters
    ==========
    X
        Type T1.
        Input data tensor from the previous operator; 4-D feature map of shape
        (N, C, H, W), where N is the batch size, C is the number of channels,
        and H and W are the height and the width of the data.
    rois
        Type T1.
        RoIs (Regions of Interest) to pool over; rois is 2-D input of shape
        (num_rois, 4) given as [[x1, y1, x2, y2], ...]. The RoIs' coordinates
        are in the coordinate system of the input image. Each coordinate set has
        a 1:1 correspondence with the 'batch_indices' input.
    batch_indices
        Type T2.
        1-D tensor of shape (num_rois,) with each element denoting the index of
        the corresponding image in the batch.
    coordinate_transformation_mode
        Attribute.
        Allowed values are 'half_pixel' and 'output_half_pixel'. Use the value
        'half_pixel' to pixel shift the input coordinates by -0.5 (the
        recommended behavior). Use the value 'output_half_pixel' to omit the
        pixel shift for the input (use this for a backward-compatible behavior).
    mode
        Attribute.
        The pooling method. Two modes are supported: 'avg' and 'max'. Default is
        'avg'.
    output_height
        Attribute.
        default 1; Pooled output Y's height.
    output_width
        Attribute.
        default 1; Pooled output Y's width.
    sampling_ratio
        Attribute.
        Number of sampling points in the interpolation grid used to compute the
        output value of each pooled output bin. If > 0, then exactly
        sampling_ratio x sampling_ratio grid points are used. If == 0, then an
        adaptive number of grid points are used (computed as ceil(roi_width /
        output_width), and likewise for height). Default is 0.
    spatial_scale
        Attribute.
        Multiplicative spatial scale factor to translate ROI coordinates from
        their input spatial scale to the scale used when pooling, i.e., spatial
        scale of the input feature map X relative to the input image. E.g.;
        default is 1.0f.

    Returns
    =======
    Y : Var
        Type T1.
        RoI pooled output, 4-D tensor of shape (num_rois, C, output_height,
        output_width). The r-th batch element Y[r-1] is a pooled feature map
        corresponding to the r-th RoI X[r-1].

    Notes
    =====
    Signature: ``ai.onnx@16::RoiAlign``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(int64)`
    """
    return _RoiAlign(
        _RoiAlign.Attributes(
            coordinate_transformation_mode=AttrString(
                coordinate_transformation_mode, name="coordinate_transformation_mode"
            ),
            mode=AttrString(mode, name="mode"),
            output_height=AttrInt64(output_height, name="output_height"),
            output_width=AttrInt64(output_width, name="output_width"),
            sampling_ratio=AttrInt64(sampling_ratio, name="sampling_ratio"),
            spatial_scale=AttrFloat32(spatial_scale, name="spatial_scale"),
        ),
        _RoiAlign.Inputs(
            X=X,
            rois=rois,
            batch_indices=batch_indices,
        ),
    ).outputs.Y


def round(
    X: Var,
) -> Var:
    r"""
    Round takes one input Tensor and rounds the values, element-wise,
    meaning it finds the nearest integer for each value. In case of halves,
    the rule is to round them to the nearest even integer. If input x is
    integral, +0, -0, NaN, or infinite, x itself is returned. The output
    tensor has the same shape and type as the input.

    Examples:

    ::

       round([0.9]) = [1.0]
       round([2.5]) = [2.0]
       round([2.3]) = [2.0]
       round([1.5]) = [2.0]
       round([-4.5]) = [-4.0]

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@11::Round``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Round(
        _Round.Attributes(),
        _Round.Inputs(
            X=X,
        ),
    ).outputs.Y


def stft(
    signal: Var,
    frame_step: Var,
    window: Optional[Var] = None,
    frame_length: Optional[Var] = None,
    *,
    onesided: int = 1,
) -> Var:
    r"""
    Computes the Short-time Fourier Transform of the signal.

    Parameters
    ==========
    signal
        Type T1.
        Input tensor representing a real or complex valued signal. For real
        input, the following shape is expected: [batch_size][signal_length][1].
        For complex input, the following shape is expected:
        [batch_size][signal_length][2], where [batch_size][signal_length][0]
        represents the real component and [batch_size][signal_length][1]
        represents the imaginary component of the signal.
    frame_step
        Type T2.
        The number of samples to step between successive DFTs.
    window
        Type T1.
        A tensor representing the window that will be slid over the signal.The
        window must have rank 1 with shape: [window_shape]. It's an optional
        value.
    frame_length
        Type T2.
        A scalar representing the size of the DFT. It's an optional value.
    onesided
        Attribute.
        If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) +
        1] are returned because the real-to-complex Fourier transform satisfies
        the conjugate symmetry, i.e., X[m, w] = X[m,w]=X[m,n_fft-w]\*. Note if
        the input or window tensors are complex, then onesided output is not
        possible. Enabling onesided with real inputs performs a Real-valued fast
        Fourier transform (RFFT).When invoked with real or complex valued input,
        the default value is 1. Values can be 0 or 1.

    Returns
    =======
    output : Var
        Type T1.
        The Short-time Fourier Transform of the signals.If onesided is 1, the
        output has the shape: [batch_size][frames][dft_unique_bins][2], where
        dft_unique_bins is frame_length // 2 + 1 (the unique components of the
        DFT) If onesided is 0, the output has the shape:
        [batch_size][frames][frame_length][2], where frame_length is the length
        of the DFT.

    Notes
    =====
    Signature: ``ai.onnx@17::STFT``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(int32)`, `tensor(int64)`
    """
    return _STFT(
        _STFT.Attributes(
            onesided=AttrInt64(onesided, name="onesided"),
        ),
        _STFT.Inputs(
            signal=signal,
            frame_step=frame_step,
            window=window,
            frame_length=frame_length,
        ),
    ).outputs.output


def scan(
    initial_state_and_scan_inputs: Sequence[Var],
    *,
    body: Callable[..., Iterable[Var]],
    num_scan_inputs: int,
    scan_input_axes: Optional[Iterable[int]] = None,
    scan_input_directions: Optional[Iterable[int]] = None,
    scan_output_axes: Optional[Iterable[int]] = None,
    scan_output_directions: Optional[Iterable[int]] = None,
) -> Sequence[Var]:
    r"""
    Scan can be used to iterate over one or more scan_input tensors,
    constructing zero or more scan_output tensors. It combines ideas from
    general recurrences, functional programming constructs such as scan,
    fold, map, and zip, and is intended to enable generalizations of
    RNN-like constructs for sequence-to-sequence processing. Other tensors
    (referred to as state_variables here) can be used to carry a state when
    iterating from one element to another (similar to hidden-state in RNNs,
    also referred to as loop-carried dependences in the context of loops).
    Many common usages involve a single scan_input tensor (where
    functionality similar to scan, fold and map can be obtained). When more
    than one scan_input is used, a behavior similar to zip is obtained.

    The attribute body must be a graph, specifying the computation to be
    performed in every iteration. It takes as input the current values of
    the state_variables and the current iterated element of the scan_inputs.
    It must return the (updated) values of the state_variables and zero or
    more scan_output_element tensors. The values of the scan_output_element
    tensors are concatenated over all the iterations to produce the
    scan_output values of the scan construct (similar to the concatenated
    intermediate hidden-state values of RNN-like constructs). All the output
    tensors (state_variables as well as scan_output_element tensors) are
    required to have the same shape in each iteration of the loop (a
    restriction imposed to enable efficient memory allocation).

    Note that the iterated element passed to the body subgraph does not have
    a sequence axis. It will have a rank one less than the rank of the
    corresponding scan_input.

    The scan operation returns the final values of the state_variables as
    well as the scan_outputs.

    The optional attribute scan_input_directions specifies the direction
    (forward or backward) for each scan input. If this attribute is omitted,
    all sequences are scanned in the forward direction. A bidirectional scan
    may be performed by specifying the same tensor input twice in the
    scan_inputs, once with a forward direction, and once with a backward
    direction.

    The scan_output of the operation is produced by concatenating the
    scan_output_element values produced by the body in each iteration. The
    optional attribute scan_output_directions specifies the direction in
    which scan_output is constructed (by appending or prepending the
    scan_output_element to scan_output in each iteration) for each
    scan_output. If this attribute is omitted, the scan_output_element is
    appended to the scan_output in each iteration.

    The optional attribute scan_input_axes specifies the axis to be scanned
    for each scan_input. If omitted, every scan_input will be scanned in
    axis 0. For example, if axis 0 is the batch axis and axis 1 is the time
    axis (to be scanned), specify an axis value of 1. Note that scanning a
    non-zero axis may be less efficient than scanning axis zero.

    The optional attribute scan_output_axes specifies the axis along which
    the scan_outputs are accumulated for each scan_output. For example, if
    axis 1 is the time axis (to be scanned) for both inputs and outputs,
    specify a scan_input axis and scan_output axis value of 1.

    Note that because of the ONNX restriction that only the last parameter
    of an operator can be variadic, the initial-states and scan-inputs are
    listed together as one input parameter. Similarly, the final-states and
    scan-outputs are listed together as one output parameter. The attribute
    num_scan_inputs indicates the number M of scan-inputs.

    The behavior of

    ::

       Scan <
           num_scan_inputs = m,
           body = loop-body,
           scan_input_axes = [axis_1, ..., axis_m]
       > (init_1, ..., init_n, scan_1, ..., scan_m)

    is equivalent to the following pseudo-code:

    ::

       // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
       // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
       sequence_length = scan_1.shape[axis_1];

       // initialize state-variables
       st_1 = init_1; ... st_n = init_n;
       // initialize scan-output variables: [] denotes an empty tensor
       scan_out_1 = []; ...; scan_out_k = [];
       // identify number of iterations:

       // execute loop
       for (int t = 0; t < sequence_length; ++t) {
           // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
           // of rank one less than T obtained by indexing T at position t along axis k.
           si_1 = scan_1<axis=axis_1>[t];
           ... ;
           si_m = scan_m<axis=axis_m>[t];
           // execute loop-body
           st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
           // accumulate the scan-output elements
           scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
       }

       return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

    *Sample usage: Encoding RNN using a Scan*

    The following example shows how a simple RNN over an input tensor %X,
    with weight tensor %Wi, recurrence weight tensor %Ri, bias tensors %Wbi
    and %Rbi, and initial hidden-state %H_0 can be encoded as a ScanLoop.
    Note that the loop-body is a nested graph, and it directly computes %Wi,
    %Ri, %Wbi, and %Rbi (typically constants or initializers in the body
    graph). If these values are computed in the outer graph, they need to be
    passed in as extra state_variables.

    ::

       graph rnn-encoding {
         %H_0 = ...
         %X = ...
         %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
         return %Y, %Y_h
       }

       graph rnn-cell-1 (
         %H_tminus1[FLOAT, tensor]
         %X_t[FLOAT, tensor]
       ) {
         %Wi = ...
         %Ri = ...
         %Wbi = ...
         %Rbi = ...
         %t1 = X_t * (Wi^T)
         %t2 = H_tminus1*(Ri^T)
         %t3 = Add(%t1, %t2)
         %t4 = Add(%t3, %Wbi)
         %t5 = Add(%t4, %Rbi)
         %Ht = Tanh(%t5)
         %Accumulate = Identity(%Ht)
         return %Ht, %Accumulate
       }

    Parameters
    ==========
    initial_state_and_scan_inputs
        Type V.
        Initial values of the loop's N state variables followed by M scan_inputs
    body
        Attribute.
        The graph run each iteration. It has N+M inputs: (loop state
        variables..., scan_input_elts...). It has N+K outputs: (loop state
        variables..., scan_output_elts...). Each scan_output is created by
        concatenating the value of the specified scan_output_elt value at the
        end of each iteration of the loop. It is an error if the dimensions of
        these values change across loop iterations.
    num_scan_inputs
        Attribute.
        An attribute specifying the number of scan_inputs M.
    scan_input_axes
        Attribute.
        An optional list of M flags. The i-th element of the list specifies the
        axis to be scanned (the sequence axis) for the i-th scan_input. If
        omitted, 0 will be used as the scan axis for every scan_input. Negative
        value for an axis means counting dimensions from the back. Accepted
        range is [-r, r-1] where r = rank(input).
    scan_input_directions
        Attribute.
        An optional list of M flags. The i-th element of the list specifies the
        direction to be scanned for the i-th scan_input tensor: 0 indicates
        forward direction and 1 indicates reverse direction. If omitted, all
        scan_input tensors will be scanned in the forward direction.
    scan_output_axes
        Attribute.
        An optional list of K flags. The i-th element of the list specifies the
        axis for the i-th scan_output. The scan outputs are accumulated along
        the specified axis. If omitted, 0 will be used as the scan axis for
        every scan_output. Negative value for an axis means counting dimensions
        from the back. Accepted range is [-r, r-1].
    scan_output_directions
        Attribute.
        An optional list of K flags, one for each scan_output. The i-th element
        of the list specifies whether the i-th scan_output should be constructed
        by appending or prepending a new value in each iteration: 0 indicates
        appending and 1 indicates prepending. If omitted, all scan_output
        tensors will be produced by appending a value in each iteration.

    Returns
    =======
    final_state_and_scan_outputs : Sequence[Var]
        Type V.
        Final values of the loop's N state variables followed by K scan_outputs

    Notes
    =====
    Signature: ``ai.onnx@16::Scan``.

    Type constraints:
     - V: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    _body_subgraph: Graph = subgraph(
        [
            Tensor(
                var.unwrap_tensor().dtype,
                (lambda x: x[1:] if x is not None else None)(var.unwrap_tensor().shape),
            )
            for var in initial_state_and_scan_inputs[:num_scan_inputs]
        ]
        + [
            Tensor(var.unwrap_tensor().dtype)
            for var in initial_state_and_scan_inputs[num_scan_inputs:]
        ],
        body,
    )
    return _Scan(
        _Scan.Attributes(
            body=AttrGraph(_body_subgraph, name="body"),
            num_scan_inputs=AttrInt64(num_scan_inputs, name="num_scan_inputs"),
            scan_input_axes=AttrInt64s.maybe(scan_input_axes, name="scan_input_axes"),
            scan_input_directions=AttrInt64s.maybe(
                scan_input_directions, name="scan_input_directions"
            ),
            scan_output_axes=AttrInt64s.maybe(
                scan_output_axes, name="scan_output_axes"
            ),
            scan_output_directions=AttrInt64s.maybe(
                scan_output_directions, name="scan_output_directions"
            ),
        ),
        _Scan.Inputs(
            initial_state_and_scan_inputs=initial_state_and_scan_inputs,
        ),
        out_variadic=len(_body_subgraph.requested_results),
    ).outputs.final_state_and_scan_outputs


def scatter_elements(
    data: Var,
    indices: Var,
    updates: Var,
    *,
    axis: int = 0,
    reduction: str = "none",
) -> Var:
    r"""
    ScatterElements takes three inputs ``data``, ``updates``, and
    ``indices`` of the same rank r >= 1 and an optional attribute axis that
    identifies an axis of ``data`` (by default, the outer-most axis, that is
    axis 0). The output of the operation is produced by creating a copy of
    the input ``data``, and then updating its value to values specified by
    ``updates`` at specific index positions specified by ``indices``. Its
    output shape is the same as the shape of ``data``. For each entry in
    ``updates``, the target index in ``data`` is obtained by combining the
    corresponding entry in ``indices`` with the index of the entry itself:
    the index-value for dimension = axis is obtained from the value of the
    corresponding entry in ``indices`` and the index-value for dimension !=
    axis is obtained from the index of the entry itself. ``reduction``
    allows specification of an optional reduction operation, which is
    applied to all values in ``updates`` tensor into ``output`` at the
    specified ``indices``. In cases where ``reduction`` is set to "none",
    indices should not have duplicate entries: that is, if idx1 != idx2,
    then indices[idx1] != indices[idx2]. For instance, in a 2-D tensor case,
    the update corresponding to the [i][j] entry is performed as below:

    ::

         output[indices[i][j]][j] = updates[i][j] if axis = 0,
         output[i][indices[i][j]] = updates[i][j] if axis = 1,

    When ``reduction`` is set to "add", the update corresponding to the
    [i][j] entry is performed as below:

    ::

         output[indices[i][j]][j] += updates[i][j] if axis = 0,
         output[i][indices[i][j]] += updates[i][j] if axis = 1,

    When ``reduction`` is set to "mul", the update corresponding to the
    [i][j] entry is performed as below:

    ::

         output[indices[i][j]][j] *= updates[i][j] if axis = 0,
         output[i][indices[i][j]] *= updates[i][j] if axis = 1,

    This operator is the inverse of GatherElements. It is similar to Torch's
    Scatter operation. Example 1:

    ::

         data = [
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
         ]
         indices = [
             [1, 0, 2],
             [0, 2, 1],
         ]
         updates = [
             [1.0, 1.1, 1.2],
             [2.0, 2.1, 2.2],
         ]
         output = [
             [2.0, 1.1, 0.0]
             [1.0, 0.0, 2.2]
             [0.0, 2.1, 1.2]
         ]

    Example 2:

    ::

         data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
         indices = [[1, 3]]
         updates = [[1.1, 2.1]]
         axis = 1
         output = [[1.0, 1.1, 3.0, 2.1, 5.0]]

    Parameters
    ==========
    data
        Type T.
        Tensor of rank r >= 1.
    indices
        Type Tind.
        Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index
        values are expected to be within bounds [-s, s-1] along axis of size s.
        It is an error if any of the index values are out of bounds.
    updates
        Type T.
        Tensor of rank r >=1 (same rank and shape as indices)
    axis
        Attribute.
        Which axis to scatter on. Negative value means counting dimensions from
        the back. Accepted range is [-r, r-1] where r = rank(data).
    reduction
        Attribute.
        Type of reduction to apply: none (default), add, mul. 'none': no
        reduction applied. 'add': reduction using the addition operation. 'mul':
        reduction using the multiplication operation.

    Returns
    =======
    output : Var
        Type T.
        Tensor of rank r >= 1 (same rank as input).

    Notes
    =====
    Signature: ``ai.onnx@16::ScatterElements``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - Tind: `tensor(int32)`, `tensor(int64)`
    """
    return _ScatterElements(
        _ScatterElements.Attributes(
            axis=AttrInt64(axis, name="axis"),
            reduction=AttrString(reduction, name="reduction"),
        ),
        _ScatterElements.Inputs(
            data=data,
            indices=indices,
            updates=updates,
        ),
    ).outputs.output


def scatter_nd(
    data: Var,
    indices: Var,
    updates: Var,
    *,
    reduction: str = "none",
) -> Var:
    r"""
    ScatterND takes three inputs ``data`` tensor of rank r >= 1, ``indices``
    tensor of rank q >= 1, and ``updates`` tensor of rank q + r -
    indices.shape[-1] - 1. The output of the operation is produced by
    creating a copy of the input ``data``, and then updating its value to
    values specified by ``updates`` at specific index positions specified by
    ``indices``. Its output shape is the same as the shape of ``data``.

    ``indices`` is an integer tensor. Let k denote indices.shape[-1], the
    last dimension in the shape of ``indices``. ``indices`` is treated as a
    (q-1)-dimensional tensor of k-tuples, where each k-tuple is a
    partial-index into ``data``. Hence, k can be a value at most the rank of
    ``data``. When k equals rank(data), each update entry specifies an
    update to a single element of the tensor. When k is less than rank(data)
    each update entry specifies an update to a slice of the tensor. Index
    values are allowed to be negative, as per the usual convention for
    counting backwards from the end, but are expected in the valid range.

    ``updates`` is treated as a (q-1)-dimensional tensor of
    replacement-slice-values. Thus, the first (q-1) dimensions of
    updates.shape must match the first (q-1) dimensions of indices.shape.
    The remaining dimensions of ``updates`` correspond to the dimensions of
    the replacement-slice-values. Each replacement-slice-value is a (r-k)
    dimensional tensor, corresponding to the trailing (r-k) dimensions of
    ``data``. Thus, the shape of ``updates`` must equal indices.shape[0:q-1]
    ++ data.shape[k:r-1], where ++ denotes the concatenation of shapes.

    The ``output`` is calculated via the following equation: output =
    np.copy(data) update_indices = indices.shape[:-1] for idx in
    np.ndindex(update_indices): output[indices[idx]] = updates[idx] The
    order of iteration in the above loop is not specified. In particular,
    indices should not have duplicate entries: that is, if idx1 != idx2,
    then indices[idx1] != indices[idx2]. This ensures that the output value
    does not depend on the iteration order.

    ``reduction`` allows specification of an optional reduction operation,
    which is applied to all values in ``updates`` tensor into ``output`` at
    the specified ``indices``. In cases where ``reduction`` is set to
    "none", indices should not have duplicate entries: that is, if idx1 !=
    idx2, then indices[idx1] != indices[idx2]. This ensures that the output
    value does not depend on the iteration order. When ``reduction`` is set
    to "add", ``output`` is calculated as follows: output = np.copy(data)
    update_indices = indices.shape[:-1] for idx in
    np.ndindex(update_indices): output[indices[idx]] += updates[idx] When
    ``reduction`` is set to "mul", ``output`` is calculated as follows:
    output = np.copy(data) update_indices = indices.shape[:-1] for idx in
    np.ndindex(update_indices): output[indices[idx]] \*= updates[idx] This
    operator is the inverse of GatherND. Example 1:

    ::

         data    = [1, 2, 3, 4, 5, 6, 7, 8]
         indices = [[4], [3], [1], [7]]
         updates = [9, 10, 11, 12]
         output  = [1, 11, 3, 10, 9, 6, 7, 12]

    Example 2:

    ::

         data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
         indices = [[0], [2]]
         updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
         output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]

    Parameters
    ==========
    data
        Type T.
        Tensor of rank r >= 1.
    indices
        Type tensor(int64).
        Tensor of rank q >= 1.
    updates
        Type T.
        Tensor of rank q + r - indices_shape[-1] - 1.
    reduction
        Attribute.
        Type of reduction to apply: none (default), add, mul. 'none': no
        reduction applied. 'add': reduction using the addition operation. 'mul':
        reduction using the multiplication operation.

    Returns
    =======
    output : Var
        Type T.
        Tensor of rank r >= 1.

    Notes
    =====
    Signature: ``ai.onnx@16::ScatterND``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ScatterND(
        _ScatterND.Attributes(
            reduction=AttrString(reduction, name="reduction"),
        ),
        _ScatterND.Inputs(
            data=data,
            indices=indices,
            updates=updates,
        ),
    ).outputs.output


def selu(
    X: Var,
    *,
    alpha: float = 1.6732631921768188,
    gamma: float = 1.0507010221481323,
) -> Var:
    r"""
    Selu takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the scaled exponential linear unit function,
    ``y = gamma * (alpha * e^x - alpha) for x <= 0``,
    ``y = gamma * x for x > 0``, is applied to the tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor
    alpha
        Attribute.
        Coefficient of SELU default to 1.67326319217681884765625 (i.e., float32
        approximation of 1.6732632423543772848170429916717).
    gamma
        Attribute.
        Coefficient of SELU default to 1.05070102214813232421875 (i.e., float32
        approximation of 1.0507009873554804934193349852946).

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@6::Selu``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Selu(
        _Selu.Attributes(
            alpha=AttrFloat32(alpha, name="alpha"),
            gamma=AttrFloat32(gamma, name="gamma"),
        ),
        _Selu.Inputs(
            X=X,
        ),
    ).outputs.Y


def sequence_at(
    input_sequence: Var,
    position: Var,
) -> Var:
    r"""
    Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
    Accepted range for 'position' is in ``[-n, n - 1]``, where ``n`` is the
    number of tensors in 'input_sequence'. Negative value means counting
    positions from the back.

    Parameters
    ==========
    input_sequence
        Type S.
        Input sequence.
    position
        Type I.
        Position of the tensor in the sequence. Negative value means counting
        positions from the back. Accepted range in ``[-n, n - 1]``, where ``n``
        is the number of tensors in 'input_sequence'. It is an error if any of
        the index values are out of bounds. It must be a scalar(tensor of empty
        shape).

    Returns
    =======
    tensor : Var
        Type T.
        Output tensor at the specified position in the input sequence.

    Notes
    =====
    Signature: ``ai.onnx@11::SequenceAt``.

    Type constraints:
     - S: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`
     - I: `tensor(int32)`, `tensor(int64)`
     - T: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _SequenceAt(
        _SequenceAt.Attributes(),
        _SequenceAt.Inputs(
            input_sequence=input_sequence,
            position=position,
        ),
    ).outputs.tensor


def sequence_construct(
    inputs: Sequence[Var],
) -> Var:
    r"""
    Construct a tensor sequence containing 'inputs' tensors. All tensors in
    'inputs' must have the same data type.

    Parameters
    ==========
    inputs
        Type T.
        Tensors.

    Returns
    =======
    output_sequence : Var
        Type S.
        Sequence enclosing the input tensors.

    Notes
    =====
    Signature: ``ai.onnx@11::SequenceConstruct``.

    Type constraints:
     - T: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - S: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`
    """
    return _SequenceConstruct(
        _SequenceConstruct.Attributes(),
        _SequenceConstruct.Inputs(
            inputs=inputs,
        ),
    ).outputs.output_sequence


def sequence_empty(
    *,
    dtype: Optional[npt.DTypeLike] = None,
) -> Var:
    r"""
    Construct an empty tensor sequence, with given data type.

    Parameters
    ==========
    dtype
        Attribute.
        (Optional) The data type of the tensors in the output sequence. The
        default type is 'float'.

    Returns
    =======
    output : Var
        Type S.
        Empty sequence.

    Notes
    =====
    Signature: ``ai.onnx@11::SequenceEmpty``.

    Type constraints:
     - S: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`
    """
    return _SequenceEmpty(
        _SequenceEmpty.Attributes(
            dtype=AttrDtype.maybe(dtype, name="dtype"),
        ),
        _SequenceEmpty.Inputs(),
    ).outputs.output


def sequence_erase(
    input_sequence: Var,
    position: Optional[Var] = None,
) -> Var:
    r"""
    Outputs a tensor sequence that removes the tensor at 'position' from
    'input_sequence'. Accepted range for 'position' is in ``[-n, n - 1]``,
    where ``n`` is the number of tensors in 'input_sequence'. Negative value
    means counting positions from the back. 'position' is optional, by
    default it erases the last tensor from 'input_sequence'.

    Parameters
    ==========
    input_sequence
        Type S.
        Input sequence.
    position
        Type I.
        Position of the tensor in the sequence. Negative value means counting
        positions from the back. Accepted range in ``[-n, n - 1]``, where ``n``
        is the number of tensors in 'input_sequence'. It is an error if any of
        the index values are out of bounds. It must be a scalar(tensor of empty
        shape).

    Returns
    =======
    output_sequence : Var
        Type S.
        Output sequence that has the tensor at the specified position removed.

    Notes
    =====
    Signature: ``ai.onnx@11::SequenceErase``.

    Type constraints:
     - S: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`
     - I: `tensor(int32)`, `tensor(int64)`
    """
    return _SequenceErase(
        _SequenceErase.Attributes(),
        _SequenceErase.Inputs(
            input_sequence=input_sequence,
            position=position,
        ),
    ).outputs.output_sequence


def sequence_insert(
    input_sequence: Var,
    tensor: Var,
    position: Optional[Var] = None,
) -> Var:
    r"""
    Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at
    'position'. 'tensor' must have the same data type as 'input_sequence'.
    Accepted range for 'position' is in ``[-n, n]``, where ``n`` is the
    number of tensors in 'input_sequence'. Negative value means counting
    positions from the back. 'position' is optional, by default it inserts
    'tensor' to the back of 'input_sequence'.

    Parameters
    ==========
    input_sequence
        Type S.
        Input sequence.
    tensor
        Type T.
        Input tensor to be inserted into the input sequence.
    position
        Type I.
        Position in the sequence where the new tensor is inserted. It is
        optional and default is to insert to the back of the sequence. Negative
        value means counting positions from the back. Accepted range in
        ``[-n, n]``, where ``n`` is the number of tensors in 'input_sequence'.
        It is an error if any of the index values are out of bounds. It must be
        a scalar(tensor of empty shape).

    Returns
    =======
    output_sequence : Var
        Type S.
        Output sequence that contains the inserted tensor at given position.

    Notes
    =====
    Signature: ``ai.onnx@11::SequenceInsert``.

    Type constraints:
     - S: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`
     - T: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - I: `tensor(int32)`, `tensor(int64)`
    """
    return _SequenceInsert(
        _SequenceInsert.Attributes(),
        _SequenceInsert.Inputs(
            input_sequence=input_sequence,
            tensor=tensor,
            position=position,
        ),
    ).outputs.output_sequence


def sequence_length(
    input_sequence: Var,
) -> Var:
    r"""
    Produces a scalar(tensor of empty shape) containing the number of
    tensors in 'input_sequence'.

    Parameters
    ==========
    input_sequence
        Type S.
        Input sequence.

    Returns
    =======
    length : Var
        Type I.
        Length of input sequence. It must be a scalar(tensor of empty shape).

    Notes
    =====
    Signature: ``ai.onnx@11::SequenceLength``.

    Type constraints:
     - S: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`
     - I: `tensor(int64)`
    """
    return _SequenceLength(
        _SequenceLength.Attributes(),
        _SequenceLength.Inputs(
            input_sequence=input_sequence,
        ),
    ).outputs.length


def sequence_map(
    input_sequence: Var,
    additional_inputs: Sequence[Var] = (),
    *,
    body: Callable[..., Iterable[Var]],
) -> Sequence[Var]:
    r"""
    Applies a sub-graph to each sample in the input sequence(s).

    Inputs can be either tensors or sequences, with the exception of the
    first input which must be a sequence. The length of the first input
    sequence will determine the number of samples in the outputs. Any other
    sequence inputs should have the same number of samples. The number of
    inputs and outputs, should match the one of the subgraph.

    For each i-th element in the output, a sample will be extracted from the
    input sequence(s) at the i-th position and the sub-graph will be applied
    to it. The outputs will contain the outputs of the sub-graph for each
    sample, in the same order as in the input.

    This operator assumes that processing each sample is independent and
    could executed in parallel or in any order. Users cannot expect any
    specific ordering in which each subgraph is computed.

    Parameters
    ==========
    input_sequence
        Type S.
        Input sequence.
    additional_inputs
        Type V.
        Additional inputs to the graph
    body
        Attribute.
        The graph to be run for each sample in the sequence(s). It should have
        as many inputs and outputs as inputs and outputs to the SequenceMap
        function.

    Returns
    =======
    out_sequence : Sequence[Var]
        Type S.
        Output sequence(s)

    Notes
    =====
    Signature: ``ai.onnx@17::SequenceMap``.

    Type constraints:
     - S: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`
     - V: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    _body_subgraph: Graph = subgraph(
        [typing_cast(SpoxSequence, input_sequence.unwrap_type()).elem_type]
        + [
            typing_cast(SpoxSequence, var.unwrap_type()).elem_type
            for var in additional_inputs
        ],
        body,
    )
    return _SequenceMap(
        _SequenceMap.Attributes(
            body=AttrGraph(_body_subgraph, name="body"),
        ),
        _SequenceMap.Inputs(
            input_sequence=input_sequence,
            additional_inputs=additional_inputs,
        ),
        out_variadic=len(_body_subgraph.requested_results),
    ).outputs.out_sequence


def shape(
    data: Var,
    *,
    end: Optional[int] = None,
    start: int = 0,
) -> Var:
    r"""
    Takes a tensor as input and outputs an 1D int64 tensor containing the
    shape of the input tensor. Optional attributes start and end can be used
    to compute a slice of the input tensor's shape. If start axis is
    omitted, the slice starts from axis 0. The end axis, if specified, is
    exclusive (and the returned value will not include the size of that
    axis). If the end axis is omitted, the axes upto the last one will be
    included. Negative axes indicate counting back from the last axis. Note
    that axes will be clamped to the range [0, r-1], where r is the rank of
    the input tensor if they are out-of-range (after adding r in the case of
    negative axis). Thus, specifying any end value > r is equivalent to
    specifying an end value of r, and specifying any start value < -r is
    equivalent to specifying a start value of 0.

    Examples:

    ::

       Input tensor with shape: [2, 3, 4]
       No attributes specified.
       Output: [2, 3, 4]

    ::

       Input tensor with shape: [2, 3, 4]
       start: -1
       Output: [4]

    ::

       Input tensor with shape: [2, 3, 4]
       end: -1
       Output: [2, 3]

    ::

       Input tensor with shape: [2, 3, 4]
       start: 1
       end: 2
       Output: [3]

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    end
        Attribute.
        (Optional) Ending axis for slicing the shape. Negative value means
        counting dimensions from the back. If omitted, sizes of all axes upto
        (including) the last one will be included.
    start
        Attribute.
        (Optional) Starting axis for slicing the shape. Default value is
        0.Negative value means counting dimensions from the back.

    Returns
    =======
    shape : Var
        Type T1.
        Shape of the input tensor

    Notes
    =====
    Signature: ``ai.onnx@15::Shape``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(int64)`
    """
    return _Shape(
        _Shape.Attributes(
            end=AttrInt64.maybe(end, name="end"),
            start=AttrInt64(start, name="start"),
        ),
        _Shape.Inputs(
            data=data,
        ),
    ).outputs.shape


def shrink(
    input: Var,
    *,
    bias: float = 0.0,
    lambd: float = 0.5,
) -> Var:
    r"""
    Shrink takes one input data (Tensor) and produces one Tensor output,
    having same datatype and shape with input. It has two attributes, lambd
    and bias. The formula of this operator is: If x < -lambd, y = x + bias;
    If x > lambd, y = x - bias; Otherwise, y = 0.

    Parameters
    ==========
    input
        Type T.
        The input data as Tensor.
    bias
        Attribute.
        The bias value added to output. Default is 0.
    lambd
        Attribute.
        The lambd value for the Shrink formulation. Default is 0.5.

    Returns
    =======
    output : Var
        Type T.
        The output.

    Notes
    =====
    Signature: ``ai.onnx@9::Shrink``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Shrink(
        _Shrink.Attributes(
            bias=AttrFloat32(bias, name="bias"),
            lambd=AttrFloat32(lambd, name="lambd"),
        ),
        _Shrink.Inputs(
            input=input,
        ),
    ).outputs.output


def sigmoid(
    X: Var,
) -> Var:
    r"""
    Sigmoid takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is
    applied to the tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Sigmoid``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Sigmoid(
        _Sigmoid.Attributes(),
        _Sigmoid.Inputs(
            X=X,
        ),
    ).outputs.Y


def sign(
    input: Var,
) -> Var:
    r"""
    Calculate the sign of the given input tensor element-wise. If input > 0,
    output 1. if input < 0, output -1. if input == 0, output 0.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The sign of the input tensor computed element-wise. It has the same
        shape and type of the input.

    Notes
    =====
    Signature: ``ai.onnx@13::Sign``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Sign(
        _Sign.Attributes(),
        _Sign.Inputs(
            input=input,
        ),
    ).outputs.output


def sin(
    input: Var,
) -> Var:
    r"""
    Calculates the sine of the given input tensor, element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The sine of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@7::Sin``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Sin(
        _Sin.Attributes(),
        _Sin.Inputs(
            input=input,
        ),
    ).outputs.output


def sinh(
    input: Var,
) -> Var:
    r"""
    Calculates the hyperbolic sine of the given input tensor element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The hyperbolic sine values of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@9::Sinh``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Sinh(
        _Sinh.Attributes(),
        _Sinh.Inputs(
            input=input,
        ),
    ).outputs.output


def size(
    data: Var,
) -> Var:
    r"""
    Takes a tensor as input and outputs a int64 scalar that equals to the
    total number of elements of the input tensor.

    Parameters
    ==========
    data
        Type T.
        An input tensor.

    Returns
    =======
    size : Var
        Type T1.
        Total number of elements of the input tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Size``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(int64)`
    """
    return _Size(
        _Size.Attributes(),
        _Size.Inputs(
            data=data,
        ),
    ).outputs.size


def slice(
    data: Var,
    starts: Var,
    ends: Var,
    axes: Optional[Var] = None,
    steps: Optional[Var] = None,
) -> Var:
    r"""
    Produces a slice of the input tensor along multiple axes. Similar to
    numpy:
    https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding

    Slice uses the ``starts``, ``ends``, ``axes`` and ``steps`` inputs to
    select a sub-tensor of its input ``data`` tensor.

    An effective ``starts[i]``, ``ends[i]``, and ``steps[i]`` must be
    computed for each ``i`` in ``[0, ... r-1]`` where ``r = rank(input)`` as
    follows:

    If ``axes`` are omitted, they are set to ``[0, ..., r-1]``. If ``steps``
    are omitted, they are set to ``[1, ..., 1]`` of length ``len(starts)``

    The effective values are initialized as ``start[i] = 0``,
    ``ends[i] = dims[i]`` where ``dims`` are the dimensions of ``input`` and
    ``steps[i] = 1``.

    All negative elements of ``axes`` are made non-negative by adding ``r``
    to them, where ``r =rank(input)``.

    All negative values in ``starts[i]`` and ``ends[i]`` have
    ``dims[axes[i]]`` added to them, where ``dims`` are the dimensions of
    ``input``. Then ``start[axes[i]]`` is the adjusted ``starts[i]`` is
    clamped into the range ``[0, dims[axes[i]]]`` for positive stepping and
    ``[0, dims[axes[i]]-1]`` for negative stepping.

    The clamping for the adjusted ``ends[i]`` depends on the sign of
    ``steps[i]`` and must accommodate copying 0 through ``dims[axes[i]]``
    elements, so for positive stepping ``ends[axes[i]]`` is clamped to
    ``[0, dims[axes[i]]]``, while for negative stepping it is clamped to
    ``[-1, dims[axes[i]]-1]``.

    Finally, ``steps[axes[i]] = steps[i]``.

    For slicing to the end of a dimension with unknown size, it is
    recommended to pass in ``INT_MAX`` when slicing forward and 'INT_MIN'
    when slicing backward.

    Example 1:

    ::

       data = [
           [1, 2, 3, 4],
           [5, 6, 7, 8],
       ]
       axes = [0, 1]
       starts = [1, 0]
       ends = [2, 3]
       steps = [1, 2]
       result = [
           [5, 7],
       ]

    Example 2:

    ::

       data = [
           [1, 2, 3, 4],
           [5, 6, 7, 8],
       ]
       starts = [0, 1]
       ends = [-1, 1000]
       result = [
           [2, 3, 4],
       ]

    Parameters
    ==========
    data
        Type T.
        Tensor of data to extract slices from.
    starts
        Type Tind.
        1-D tensor of starting indices of corresponding axis in ``axes``
    ends
        Type Tind.
        1-D tensor of ending indices (exclusive) of corresponding axis in
        ``axes``
    axes
        Type Tind.
        1-D tensor of axes that ``starts`` and ``ends`` apply to. Negative value
        means counting dimensions from the back. Accepted range is [-r, r-1]
        where r = rank(data). Behavior is undefined if an axis is repeated.
    steps
        Type Tind.
        1-D tensor of slice step of corresponding axis in ``axes``. Negative
        value means slicing backward. 'steps' cannot be 0. Defaults to 1s.

    Returns
    =======
    output : Var
        Type T.
        Sliced data tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Slice``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - Tind: `tensor(int32)`, `tensor(int64)`
    """
    return _Slice(
        _Slice.Attributes(),
        _Slice.Inputs(
            data=data,
            starts=starts,
            ends=ends,
            axes=axes,
            steps=steps,
        ),
    ).outputs.output


def softmax(
    input: Var,
    *,
    axis: int = -1,
) -> Var:
    r"""
    The operator computes the normalized exponential values for the given
    input:

    Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis,
    keepdims=1)

    The "axis" attribute indicates the dimension along which Softmax will be
    performed. The output tensor has the same shape and contains the Softmax
    values of the corresponding input.

    Parameters
    ==========
    input
        Type T.
        The input tensor of rank >= axis.
    axis
        Attribute.
        Describes the dimension Softmax will be performed on. Negative value
        means counting dimensions from the back. Accepted range is [-r, r-1]
        where r = rank(input).

    Returns
    =======
    output : Var
        Type T.
        The output values with the same shape as the input tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Softmax``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Softmax(
        _Softmax.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _Softmax.Inputs(
            input=input,
        ),
    ).outputs.output


def softmax_cross_entropy_loss(
    scores: Var,
    labels: Var,
    weights: Optional[Var] = None,
    *,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> Tuple[Var, Var]:
    r"""
    Loss function that measures the softmax cross entropy between 'scores'
    and 'labels'. This operator first computes a loss tensor whose shape is
    identical to the labels input. If the input is 2-D with shape (N, C),
    the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N). If
    the input is N-D tensor with shape (N, C, D1, D2, ..., Dk), the loss
    tensor L may have (N, D1, D2, ..., Dk) as its shape and
    L[i,][j_1][j_2]...[j_k] denotes a scalar element in L. After L is
    available, this operator can optionally do a reduction operator.

    -  shape(scores): (N, C) where C is the number of classes, or (N, C, D1,
       D2,..., Dk), with K >= 1 in case of K-dimensional loss.
    -  shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N,
       D1, D2,..., Dk), with K >= 1 in case of K-dimensional loss.

    The loss for one sample, l_i, can calculated as follows:

    ::

       l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.

    or

    ::

       l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

    loss is zero for the case when label-value equals ignore_index.

    ::

       l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index

    where:

    ::

       p = Softmax(scores)
       y = Log(p)
       c = labels[i][d1][d2]...[dk]

    Finally, L is optionally reduced:

    -  If reduction = 'none', the output is L with shape (N, D1, D2, ...,
       Dk).
    -  If reduction = 'sum', the output is scalar: Sum(L).
    -  If reduction = 'mean', the output is scalar: ReduceMean(L), or if
       weight is provided: ``ReduceSum(L) / ReduceSum(W)``, where tensor W
       is of shape ``(N, D1, D2, ..., Dk)`` and
       ``W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]``.

    Parameters
    ==========
    scores
        Type T.
        The predicted outputs with shape [batch_size, class_size], or
        [batch_size, class_size, D1, D2 , ..., Dk], where K is the number of
        dimensions.
    labels
        Type Tind.
        The ground truth output tensor, with shape [batch_size], or [batch_size,
        D1, D2, ..., Dk], where K is the number of dimensions. Labels element
        value shall be in range of [0, C). If ignore_index is specified, it may
        have a value outside [0, C) and the label values should either be in the
        range [0, C) or have the value ignore_index.
    weights
        Type T.
        A manual rescaling weight given to each class. If given, it has to be a
        1D Tensor assigning weight to each of the classes. Otherwise, it is
        treated as if having all ones.
    ignore_index
        Attribute.
        Specifies a target value that is ignored and does not contribute to the
        input gradient. It's an optional value.
    reduction
        Attribute.
        Type of reduction to apply to loss: none, sum, mean(default). 'none': no
        reduction will be applied, 'sum': the output will be summed. 'mean': the
        sum of the output will be divided by the number of elements in the
        output.

    Returns
    =======
    output : Var
        Type T.
        Weighted loss float Tensor. If reduction is 'none', this has the shape
        of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of
        K-dimensional loss. Otherwise, it is a scalar.
    log_prob : Var
        Type T.
        Log probability tensor. If the output of softmax is prob, its value is
        log(prob).

    Notes
    =====
    Signature: ``ai.onnx@13::SoftmaxCrossEntropyLoss``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - Tind: `tensor(int32)`, `tensor(int64)`
    """
    return _SoftmaxCrossEntropyLoss(
        _SoftmaxCrossEntropyLoss.Attributes(
            ignore_index=AttrInt64.maybe(ignore_index, name="ignore_index"),
            reduction=AttrString(reduction, name="reduction"),
        ),
        _SoftmaxCrossEntropyLoss.Inputs(
            scores=scores,
            labels=labels,
            weights=weights,
        ),
    ).outputs._unpack_to_any()


def softplus(
    X: Var,
) -> Var:
    r"""
    Softplus takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied
    to the tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        1D input tensor

    Returns
    =======
    Y : Var
        Type T.
        1D input tensor

    Notes
    =====
    Signature: ``ai.onnx@1::Softplus``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Softplus(
        _Softplus.Attributes(),
        _Softplus.Inputs(
            X=X,
        ),
    ).outputs.Y


def softsign(
    input: Var,
) -> Var:
    r"""
    Calculates the softsign (x/(1+|x\|)) of the given input tensor
    element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The softsign (x/(1+|x\|)) values of the input tensor computed
        element-wise

    Notes
    =====
    Signature: ``ai.onnx@1::Softsign``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Softsign(
        _Softsign.Attributes(),
        _Softsign.Inputs(
            input=input,
        ),
    ).outputs.output


def space_to_depth(
    input: Var,
    *,
    blocksize: int,
) -> Var:
    r"""
    SpaceToDepth rearranges blocks of spatial data into depth. More
    specifically, this op outputs a copy of the input tensor where values
    from the height and width dimensions are moved to the depth dimension.

    Parameters
    ==========
    input
        Type T.
        Input tensor of [N,C,H,W], where N is the batch axis, C is the channel
        or depth, H is the height and W is the width.
    blocksize
        Attribute.
        Blocks of [blocksize, blocksize] are moved.

    Returns
    =======
    output : Var
        Type T.
        Output tensor of [N, C \* blocksize \* blocksize, H/blocksize,
        W/blocksize].

    Notes
    =====
    Signature: ``ai.onnx@13::SpaceToDepth``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _SpaceToDepth(
        _SpaceToDepth.Attributes(
            blocksize=AttrInt64(blocksize, name="blocksize"),
        ),
        _SpaceToDepth.Inputs(
            input=input,
        ),
    ).outputs.output


def split(
    input: Var,
    split: Optional[Var] = None,
    *,
    outputs_count: int,
    axis: int = 0,
) -> Sequence[Var]:
    r"""
    Split a tensor into a list of tensors, along the specified 'axis'.
    Lengths of the parts can be specified using input 'split'. Otherwise,
    the tensor is split to equal sized parts.

    Parameters
    ==========
    input
        Type T.
        The tensor to split
    split
        Type tensor(int64).
        Optional length of each output. Values should be >= 0.Sum of the values
        must be equal to the dim value at 'axis' specified.
    axis
        Attribute.
        Which axis to split on. A negative value means counting dimensions from
        the back. Accepted range is [-rank, rank-1] where r = rank(input).
    outputs_count
        Specifies the number of variadic outputs of this operator.
        Non-standard parameter created by the opset generator, as inference (a solution) it was not implemented or is impossible.

    Returns
    =======
    outputs : Sequence[Var]
        Type T.
        One or more outputs forming list of tensors after splitting

    Notes
    =====
    Signature: ``ai.onnx@13::Split``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Split(
        _Split.Attributes(
            axis=AttrInt64(axis, name="axis"),
        ),
        _Split.Inputs(
            input=input,
            split=split,
        ),
        out_variadic=outputs_count,
    ).outputs.outputs


def split_to_sequence(
    input: Var,
    split: Optional[Var] = None,
    *,
    axis: int = 0,
    keepdims: int = 1,
) -> Var:
    r"""
    Split a tensor into a sequence of tensors, along the specified 'axis'.
    Lengths of the parts can be specified using the optional argument
    'split'. If the argument
    ``split' is not specified, a default scalar value of 1 is used as the value of``\ split'.
    'split' must contain only positive numbers. 'split' is either a scalar
    (tensor of empty shape), or a 1-D tensor. If 'split' is a scalar, then
    'input' will be split into chunks all of size 'split' if possible. The
    last chunk alone may be smaller than 'split' if the 'input' size along
    the given axis 'axis' is not divisible by 'split'. If 'split' is a
    1-dimensional tensor, the input tensor is split into 'size(split)'
    chunks, with lengths of the parts on 'axis' specified in 'split'. In
    this scenario, the sum of entries in 'split' must be equal to the
    dimension size of input tensor on 'axis'.

    Parameters
    ==========
    input
        Type T.
        The tensor to split
    split
        Type I.
        Length of each output. It can be either a scalar(tensor of empty shape),
        or a 1-D tensor. All values must be >= 0.
    axis
        Attribute.
        Which axis to split on. A negative value means counting dimensions from
        the back. Accepted range is [-rank, rank-1].
    keepdims
        Attribute.
        Keep the split dimension or not. Default 1, which means we keep split
        dimension. If input 'split' is specified, this attribute is ignored.

    Returns
    =======
    output_sequence : Var
        Type S.
        One or more outputs forming a sequence of tensors after splitting

    Notes
    =====
    Signature: ``ai.onnx@11::SplitToSequence``.

    Type constraints:
     - T: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - I: `tensor(int32)`, `tensor(int64)`
     - S: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`
    """
    return _SplitToSequence(
        _SplitToSequence.Attributes(
            axis=AttrInt64(axis, name="axis"),
            keepdims=AttrInt64(keepdims, name="keepdims"),
        ),
        _SplitToSequence.Inputs(
            input=input,
            split=split,
        ),
    ).outputs.output_sequence


def sqrt(
    X: Var,
) -> Var:
    r"""
    Square root takes one input data (Tensor<T>) and produces one output
    data (Tensor<T>) where the square root is, y = x^0.5, is applied to the
    tensor elementwise. If x is negative, then it will return NaN.

    Parameters
    ==========
    X
        Type T.
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@13::Sqrt``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Sqrt(
        _Sqrt.Attributes(),
        _Sqrt.Inputs(
            X=X,
        ),
    ).outputs.Y


def squeeze(
    data: Var,
    axes: Optional[Var] = None,
) -> Var:
    r"""
    Remove single-dimensional entries from the shape of a tensor. Takes an
    input ``axes`` with a list of axes to squeeze. If ``axes`` is not
    provided, all the single dimensions will be removed from the shape. If
    an axis is selected with shape entry not equal to one, an error is
    raised.

    Parameters
    ==========
    data
        Type T.
        Tensors with at least max(dims) dimensions.
    axes
        Type tensor(int64).
        List of integers indicating the dimensions to squeeze. Negative value
        means counting dimensions from the back. Accepted range is [-r, r-1]
        where r = rank(data).

    Returns
    =======
    squeezed : Var
        Type T.
        Reshaped tensor with same data as input.

    Notes
    =====
    Signature: ``ai.onnx@13::Squeeze``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Squeeze(
        _Squeeze.Attributes(),
        _Squeeze.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.squeezed


def string_normalizer(
    X: Var,
    *,
    case_change_action: str = "NONE",
    is_case_sensitive: int = 0,
    locale: Optional[str] = None,
    stopwords: Optional[Iterable[str]] = None,
) -> Var:
    r"""
    StringNormalization performs string operations for basic cleaning. This
    operator has only one input (denoted by X) and only one output (denoted
    by Y). This operator first examines the elements in the X, and removes
    elements specified in "stopwords" attribute. After removing stop words,
    the intermediate result can be further lowercased, uppercased, or just
    returned depending the "case_change_action" attribute. This operator
    only accepts [C]- and [1, C]-tensor. If all elements in X are dropped,
    the output will be the empty value of string tensor with shape [1] if
    input shape is [C] and shape [1, 1] if input shape is [1, C].

    Parameters
    ==========
    X
        Type tensor(string).
        UTF-8 strings to normalize
    case_change_action
        Attribute.
        string enum that cases output to be lowercased/uppercases/unchanged.
        Valid values are "LOWER", "UPPER", "NONE". Default is "NONE"
    is_case_sensitive
        Attribute.
        Boolean. Whether the identification of stop words in X is
        case-sensitive. Default is false
    locale
        Attribute.
        Environment dependent string that denotes the locale according to which
        output strings needs to be upper/lowercased.Default en_US or platform
        specific equivalent as decided by the implementation.
    stopwords
        Attribute.
        List of stop words. If not set, no word would be removed from X.

    Returns
    =======
    Y : Var
        Type tensor(string).
        UTF-8 Normalized strings

    Notes
    =====
    Signature: ``ai.onnx@10::StringNormalizer``.

    """
    return _StringNormalizer(
        _StringNormalizer.Attributes(
            case_change_action=AttrString(
                case_change_action, name="case_change_action"
            ),
            is_case_sensitive=AttrInt64(is_case_sensitive, name="is_case_sensitive"),
            locale=AttrString.maybe(locale, name="locale"),
            stopwords=AttrStrings.maybe(stopwords, name="stopwords"),
        ),
        _StringNormalizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def sub(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Performs element-wise binary subtraction (with Numpy-style broadcasting
    support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    (Opset 14 change): Extend supported types to include uint8, int8,
    uint16, and int16.

    Parameters
    ==========
    A
        Type T.
        First operand.
    B
        Type T.
        Second operand.

    Returns
    =======
    C : Var
        Type T.
        Result, has same element type as two inputs

    Notes
    =====
    Signature: ``ai.onnx@14::Sub``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Sub(
        _Sub.Attributes(),
        _Sub.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def sum(
    data_0: Sequence[Var],
) -> Var:
    r"""
    Element-wise sum of each of the input tensors (with Numpy-style
    broadcasting support). All inputs and outputs must have the same data
    type. This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    data_0
        Type T.
        List of tensors for sum.

    Returns
    =======
    sum : Var
        Type T.
        Output tensor.

    Notes
    =====
    Signature: ``ai.onnx@13::Sum``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Sum(
        _Sum.Attributes(),
        _Sum.Inputs(
            data_0=data_0,
        ),
    ).outputs.sum


def tan(
    input: Var,
) -> Var:
    r"""
    Calculates the tangent of the given input tensor, element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The tangent of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@7::Tan``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Tan(
        _Tan.Attributes(),
        _Tan.Inputs(
            input=input,
        ),
    ).outputs.output


def tanh(
    input: Var,
) -> Var:
    r"""
    Calculates the hyperbolic tangent of the given input tensor
    element-wise.

    Parameters
    ==========
    input
        Type T.
        Input tensor

    Returns
    =======
    output : Var
        Type T.
        The hyperbolic tangent values of the input tensor computed element-wise

    Notes
    =====
    Signature: ``ai.onnx@13::Tanh``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Tanh(
        _Tanh.Attributes(),
        _Tanh.Inputs(
            input=input,
        ),
    ).outputs.output


def tf_idf_vectorizer(
    X: Var,
    *,
    max_gram_length: int,
    max_skip_count: int,
    min_gram_length: int,
    mode: str,
    ngram_counts: Iterable[int],
    ngram_indexes: Iterable[int],
    pool_int64s: Optional[Iterable[int]] = None,
    pool_strings: Optional[Iterable[str]] = None,
    weights: Optional[Iterable[float]] = None,
) -> Var:
    r"""
    This transform extracts n-grams from the input sequence and save them as
    a vector. Input can be either a 1-D or 2-D tensor. For 1-D input, output
    is the n-gram representation of that input. For 2-D input, the output is
    also a 2-D tensor whose i-th row is the n-gram representation of the
    i-th input row. More specifically, if input shape is [C], the
    corresponding output shape would be [max(ngram_indexes) + 1]. If input
    shape is [N, C], this operator produces a [N, max(ngram_indexes) +
    1]-tensor.

    In contrast to standard n-gram extraction, here, the indexes of
    extracting an n-gram from the original sequence are not necessarily
    consecutive numbers. The discontinuity between indexes are controlled by
    the number of skips. If the number of skips is 2, we should skip two
    tokens when scanning through the original sequence. Let's consider an
    example. Assume that input sequence is [94, 17, 36, 12, 28] and the
    number of skips is 2. The associated 2-grams are [94, 12] and [17, 28]
    respectively indexed by [0, 3] and [1, 4]. If the number of skips
    becomes 0, the 2-grams generated are [94, 17], [17, 36], [36, 12], [12,
    28] indexed by [0, 1], [1, 2], [2, 3], [3, 4], respectively.

    The output vector (denoted by Y) stores the count of each n-gram;
    Y[ngram_indexes[i]] indicates the times that the i-th n-gram is found.
    The attribute ngram_indexes is used to determine the mapping between
    index i and the corresponding n-gram's output coordinate. If pool_int64s
    is [94, 17, 17, 36], ngram_indexes is [1, 0], ngram_counts=[0, 0], then
    the Y[0] (first element in Y) and Y[1] (second element in Y) are the
    counts of [17, 36] and [94, 17], respectively. An n-gram which cannot be
    found in pool_strings/pool_int64s should be ignored and has no effect on
    the output. Note that we may consider all skips up to S when generating
    the n-grams.

    The examples used above are true if mode is "TF". If mode is "IDF", all
    the counts larger than 1 would be truncated to 1 and the i-th element in
    weights would be used to scale (by multiplication) the count of the i-th
    n-gram in pool. If mode is "TFIDF", this operator first computes the
    counts of all n-grams and then scale them by the associated values in
    the weights attribute.

    Only one of pool_strings and pool_int64s can be set. If pool_int64s is
    set, the input should be an integer tensor. If pool_strings is set, the
    input must be a string tensor.

    Parameters
    ==========
    X
        Type T.
        Input for n-gram extraction
    max_gram_length
        Attribute.
        Maximum n-gram length. If this value is 3, 3-grams will be used to
        generate the output.
    max_skip_count
        Attribute.
        Maximum number of items (integers/strings) to be skipped when
        constructing an n-gram from X. If max_skip_count=1, min_gram_length=2,
        max_gram_length=3, this operator may generate 2-grams with skip_count=0
        and skip_count=1, and 3-grams with skip_count=0 and skip_count=1
    min_gram_length
        Attribute.
        Minimum n-gram length. If this value is 2 and max_gram_length is 3,
        output may contain counts of 2-grams and 3-grams.
    mode
        Attribute.
        The weighting criteria. It can be one of "TF" (term frequency), "IDF"
        (inverse document frequency), and "TFIDF" (the combination of TF and
        IDF)
    ngram_counts
        Attribute.
        The starting indexes of 1-grams, 2-grams, and so on in pool. It is
        useful when determining the boundary between two consecutive collections
        of n-grams. For example, if ngram_counts is [0, 17, 36], the first index
        (zero-based) of 1-gram/2-gram/3-gram in pool are 0/17/36. This format is
        essentially identical to CSR (or CSC) sparse matrix format, and we
        choose to use this due to its popularity.
    ngram_indexes
        Attribute.
        list of int64s (type: AttributeProto::INTS). This list is parallel to
        the specified 'pool\_\*' attribute. The i-th element in ngram_indexes
        indicate the coordinate of the i-th n-gram in the output tensor.
    pool_int64s
        Attribute.
        List of int64 n-grams learned from the training set. Either this or
        pool_strings attributes must be present but not both. It's an 1-D tensor
        starting with the collections of all 1-grams and ending with the
        collections of n-grams. The i-th element in pool stores the n-gram that
        should be mapped to coordinate ngram_indexes[i] in the output vector.
    pool_strings
        Attribute.
        List of strings n-grams learned from the training set. Either this or
        pool_int64s attributes must be present but not both. It's an 1-D tensor
        starting with the collections of all 1-grams and ending with the
        collections of n-grams. The i-th element in pool stores the n-gram that
        should be mapped to coordinate ngram_indexes[i] in the output vector.
    weights
        Attribute.
        list of floats. This attribute stores the weight of each n-gram in pool.
        The i-th element in weights is the weight of the i-th n-gram in pool.
        Its length equals to the size of ngram_indexes. By default, weights is
        an all-one tensor.This attribute is used when mode is "IDF" or "TFIDF"
        to scale the associated word counts.

    Returns
    =======
    Y : Var
        Type T1.
        Ngram results

    Notes
    =====
    Signature: ``ai.onnx@9::TfIdfVectorizer``.

    Type constraints:
     - T: `tensor(int32)`, `tensor(int64)`, `tensor(string)`
     - T1: `tensor(float)`
    """
    return _TfIdfVectorizer(
        _TfIdfVectorizer.Attributes(
            max_gram_length=AttrInt64(max_gram_length, name="max_gram_length"),
            max_skip_count=AttrInt64(max_skip_count, name="max_skip_count"),
            min_gram_length=AttrInt64(min_gram_length, name="min_gram_length"),
            mode=AttrString(mode, name="mode"),
            ngram_counts=AttrInt64s(ngram_counts, name="ngram_counts"),
            ngram_indexes=AttrInt64s(ngram_indexes, name="ngram_indexes"),
            pool_int64s=AttrInt64s.maybe(pool_int64s, name="pool_int64s"),
            pool_strings=AttrStrings.maybe(pool_strings, name="pool_strings"),
            weights=AttrFloat32s.maybe(weights, name="weights"),
        ),
        _TfIdfVectorizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def thresholded_relu(
    X: Var,
    *,
    alpha: float = 1.0,
) -> Var:
    r"""
    ThresholdedRelu takes one input data (Tensor<T>) and produces one output
    data (Tensor<T>) where the rectified linear function, y = x for x >
    alpha, y = 0 otherwise, is applied to the tensor elementwise.

    Parameters
    ==========
    X
        Type T.
        Input tensor
    alpha
        Attribute.
        Threshold value

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@10::ThresholdedRelu``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _ThresholdedRelu(
        _ThresholdedRelu.Attributes(
            alpha=AttrFloat32(alpha, name="alpha"),
        ),
        _ThresholdedRelu.Inputs(
            X=X,
        ),
    ).outputs.Y


def tile(
    input: Var,
    repeats: Var,
) -> Var:
    r"""
    Constructs a tensor by tiling a given tensor. This is the same as
    function ``tile`` in Numpy, but no broadcast. For example A = [[1, 2],
    [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

    Parameters
    ==========
    input
        Type T.
        Input tensor of any shape.
    repeats
        Type T1.
        1D int64 tensor of the same length as input's dimension number, includes
        numbers of repeated copies along input's dimensions.

    Returns
    =======
    output : Var
        Type T.
        Output tensor of the same dimensions and type as tensor input.
        output_dim[i] = input_dim[i] \* repeats[i]

    Notes
    =====
    Signature: ``ai.onnx@13::Tile``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(int64)`
    """
    return _Tile(
        _Tile.Attributes(),
        _Tile.Inputs(
            input=input,
            repeats=repeats,
        ),
    ).outputs.output


def top_k(
    X: Var,
    K: Var,
    *,
    axis: int = -1,
    largest: int = 1,
    sorted: int = 1,
) -> Tuple[Var, Var]:
    r"""
    Retrieve the top-K largest or smallest elements along a specified axis.
    Given an input tensor of shape [a_0, a_1, ..., a\_{n-1}] and integer
    argument k, return two outputs:

    -  Value tensor of shape [a_0, a_1, ..., a\_{axis-1}, k, a\_{axis+1},
       ... a\_{n-1}] which contains the values of the top k elements along
       the specified axis

    -  Index tensor of shape [a_0, a_1, ..., a\_{axis-1}, k, a\_{axis+1},
       ... a\_{n-1}] which contains the indices of the top k elements
       (original indices from the input tensor).

    -  If "largest" is 1 (the default value) then the k largest elements are
       returned.

    -  If "sorted" is 1 (the default value) then the resulting k elements
       will be sorted.

    -  If "sorted" is 0, order of returned 'Values' and 'Indices' are
       undefined.

    Given two equivalent values, this operator uses the indices along the
    axis as a tiebreaker. That is, the element with the lower index will
    appear first.

    Parameters
    ==========
    X
        Type T.
        Tensor of shape [a_0, a_1, ..., a\_{n-1}]
    K
        Type tensor(int64).
        A 1-D tensor containing a single positive value corresponding to the
        number of top elements to retrieve
    axis
        Attribute.
        Dimension on which to do the sort. Negative value means counting
        dimensions from the back. Accepted range is [-r, r-1] where r =
        rank(input).
    largest
        Attribute.
        Whether to return the top-K largest or smallest elements.
    sorted
        Attribute.
        Whether to return the elements in sorted order.

    Returns
    =======
    Values : Var
        Type T.
        Tensor of shape [a_0, a_1, ..., a\_{axis-1}, k, a\_{axis+1}, ...
        a\_{n-1}] containing top K values from the input tensor
    Indices : Var
        Type I.
        Tensor of shape [a_0, a_1, ..., a\_{axis-1}, k, a\_{axis+1}, ...
        a\_{n-1}] containing the corresponding input tensor indices for the top
        K values.

    Notes
    =====
    Signature: ``ai.onnx@11::TopK``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - I: `tensor(int64)`
    """
    return _TopK(
        _TopK.Attributes(
            axis=AttrInt64(axis, name="axis"),
            largest=AttrInt64(largest, name="largest"),
            sorted=AttrInt64(sorted, name="sorted"),
        ),
        _TopK.Inputs(
            X=X,
            K=K,
        ),
    ).outputs._unpack_to_any()


def transpose(
    data: Var,
    *,
    perm: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    Transpose the input tensor similar to numpy.transpose. For example, when
    perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output
    shape will be (2, 1, 3).

    Parameters
    ==========
    data
        Type T.
        An input tensor.
    perm
        Attribute.
        A list of integers. By default, reverse the dimensions, otherwise
        permute the axes according to the values given.

    Returns
    =======
    transposed : Var
        Type T.
        Transposed output.

    Notes
    =====
    Signature: ``ai.onnx@13::Transpose``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Transpose(
        _Transpose.Attributes(
            perm=AttrInt64s.maybe(perm, name="perm"),
        ),
        _Transpose.Inputs(
            data=data,
        ),
    ).outputs.transposed


def trilu(
    input: Var,
    k: Optional[Var] = None,
    *,
    upper: int = 1,
) -> Var:
    r"""
    Given a 2-D matrix or batches of 2-D matrices, returns the upper or
    lower triangular part of the tensor(s). The attribute "upper" determines
    whether the upper or lower part is retained. If set to true, the upper
    triangular matrix is retained. Lower triangular matrix is retained
    otherwise. Default value for the "upper" attribute is true. Trilu takes
    one input tensor of shape [\*, N, M], where \* is zero or more batch
    dimensions. The upper triangular part consists of the elements on and
    above the given diagonal (k). The lower triangular part consists of
    elements on and below the diagonal. All other elements in the matrix are
    set to zero. If k = 0, the triangular part on and above/below the main
    diagonal is retained. If upper is set to true, a positive k retains the
    upper triangular matrix excluding the main diagonal and (k-1) diagonals
    above it. A negative k value retains the main diagonal and \|k\|
    diagonals below it. If upper is set to false, a positive k retains the
    lower triangular matrix including the main diagonal and k diagonals
    above it. A negative k value excludes the main diagonal and (\|k\|-1)
    diagonals below it.

    Parameters
    ==========
    input
        Type T.
        Input tensor of rank 2 or higher.
    k
        Type tensor(int64).
        A 0-D tensor containing a single value corresponding to the number
        diagonals above or below the main diagonal to exclude or include.
        Default value is 0 if it's not specified.
    upper
        Attribute.
        Boolean. Indicates whether upper or lower part of matrix is retained.
        Default is true.

    Returns
    =======
    output : Var
        Type T.
        Output tensor of the same type and shape as the input tensor.

    Notes
    =====
    Signature: ``ai.onnx@14::Trilu``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Trilu(
        _Trilu.Attributes(
            upper=AttrInt64(upper, name="upper"),
        ),
        _Trilu.Inputs(
            input=input,
            k=k,
        ),
    ).outputs.output


def unique(
    X: Var,
    *,
    axis: Optional[int] = None,
    sorted: int = 1,
) -> Tuple[Var, Var, Var, Var]:
    r"""
    Find the unique elements of a tensor. When an optional attribute 'axis'
    is provided, unique subtensors sliced along the 'axis' are returned.
    Otherwise the input tensor is flattened and unique values of the
    flattened tensor are returned.

    This operator returns the unique values or sliced unique subtensors of
    the input tensor and three optional outputs. The first output tensor 'Y'
    contains all unique values or subtensors of the input. The second
    optional output tensor 'indices' contains indices of 'Y' elements' first
    occurrence in 'X'. The third optional output tensor 'inverse_indices'
    contains, for elements of 'X', its corresponding indices in 'Y'. The
    fourth optional output tensor 'counts' contains the count of each
    element of 'Y' in the input.

    Outputs are either sorted in ascending order or optionally in the order
    of the first occurrence of the values in the input.

    https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

    Example 1:

    ::

       input_X = [2, 1, 1, 3, 4, 3]
       attribute_sorted = 0
       attribute_axis = None
       output_Y = [2, 1, 3, 4]
       output_indices = [0, 1, 3, 4]
       output_inverse_indices = [0, 1, 1, 2, 3, 2]
       output_counts = [1, 2, 2, 1]

    Example 2:

    ::

       input_X = [[1, 3], [2, 3]]
       attribute_sorted = 1
       attribute_axis = None
       output_Y = [1, 2, 3]
       output_indices = [0, 2, 1]
       output_inverse_indices = [0, 2, 1, 2]
       output_counts = [1, 1, 2]

    Example 3:

    ::

       input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
       attribute_sorted = 1
       attribute_axis = 0
       output_Y = [[1, 0, 0], [2, 3, 4]]
       output_indices = [0, 2]
       output_inverse_indices = [0, 0, 1]
       output_counts = [2, 1]

    Example 4:

    ::

       input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
                   [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
       attribute_sorted = 1
       attribute_axis = 1

    intermediate data are presented below for better understanding: there
    are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):

    ::

       A: [[1, 1], [1, 1]],
          [[0, 1], [0, 1]],
          [[2, 1], [2, 1]],
          [[0, 1], [0, 1]].

    there are 3 unique subtensors:

    ::

       [[1, 1], [1, 1]],
       [[0, 1], [0, 1]],
       [[2, 1], [2, 1]].

    sorted unique subtensors:

    ::

       B: [[0, 1], [0, 1]],
          [[1, 1], [1, 1]],
          [[2, 1], [2, 1]].

    output_Y is constructed from B:

    ::

       [[[0. 1.], [1. 1.], [2. 1.]],
        [[0. 1.], [1. 1.], [2. 1.]]]

    output_indices is to map from B to A:

    ::

       [1, 0, 2]

    output_inverse_indices is to map from A to B:

    ::

       [1, 0, 2, 0]

    output_counts:

    ::

       [2, 1, 1]

    Parameters
    ==========
    X
        Type T.
        A N-D input tensor that is to be processed.
    axis
        Attribute.
        (Optional) The dimension to apply unique. If not specified, the unique
        elements of the flattened input are returned. Negative value means
        counting dimensions from the back. Accepted range is [-r, r-1] where r =
        rank(input).
    sorted
        Attribute.
        (Optional) Whether to sort the unique elements in ascending order before
        returning as output. Must be one of 0, or 1 (default).

    Returns
    =======
    Y : Var
        Type T.
        A tensor of the same type as 'X' containing all the unique values or
        subtensors sliced along a provided 'axis' in 'X', either sorted or
        maintained in the same order they occur in input 'X'
    indices : Var
        Type tensor(int64).
        A 1-D INT64 tensor containing indices of 'Y' elements' first occurrence
        in 'X'. When 'axis' is provided, it contains indices to subtensors in
        input 'X' on the 'axis'. When 'axis' is not provided, it contains
        indices to values in the flattened input tensor.
    inverse_indices : Var
        Type tensor(int64).
        A 1-D INT64 tensor containing, for elements of 'X', its corresponding
        indices in 'Y'. When 'axis' is provided, it contains indices to
        subtensors in output 'Y' on the 'axis'. When 'axis' is not provided, it
        contains indices to values in output 'Y'.
    counts : Var
        Type tensor(int64).
        A 1-D INT64 tensor containing the count of each element of 'Y' in input
        'X'

    Notes
    =====
    Signature: ``ai.onnx@11::Unique``.

    Type constraints:
     - T: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Unique(
        _Unique.Attributes(
            axis=AttrInt64.maybe(axis, name="axis"),
            sorted=AttrInt64(sorted, name="sorted"),
        ),
        _Unique.Inputs(
            X=X,
        ),
    ).outputs._unpack_to_any()


def unsqueeze(
    data: Var,
    axes: Var,
) -> Var:
    r"""
    Insert single-dimensional entries to the shape of an input tensor
    (``data``). Takes one required input ``axes`` - which contains a list of
    dimension indices and this operator will insert a dimension of value
    ``1`` into the corresponding index of the output tensor (``expanded``).

    For example, given an input tensor (``data``) of shape [3, 4, 5], then
    Unsqueeze(data, axes=[0, 4]) outputs a tensor (``expanded``) containing
    same data as ``data`` but with shape [1, 3, 4, 5, 1].

    The input ``axes`` should not contain any duplicate entries. It is an
    error if it contains duplicates. The rank of the output tensor
    (``output_rank``) is the rank of the input tensor (``data``) plus the
    number of values in ``axes``. Each value in ``axes`` should be within
    the (inclusive) range [-output_rank , output_rank - 1]. The order of
    values in ``axes`` does not matter and can come in any order.

    Parameters
    ==========
    data
        Type T.
        Original tensor
    axes
        Type tensor(int64).
        List of integers indicating the dimensions to be inserted. Negative
        value means counting dimensions from the back. Accepted range is [-r,
        r-1] where r = rank(expanded).

    Returns
    =======
    expanded : Var
        Type T.
        Reshaped tensor with same data as input.

    Notes
    =====
    Signature: ``ai.onnx@13::Unsqueeze``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Unsqueeze(
        _Unsqueeze.Attributes(),
        _Unsqueeze.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.expanded


def where(
    condition: Var,
    X: Var,
    Y: Var,
) -> Var:
    r"""
    Return elements, either from X or Y, depending on condition. Where
    behaves like
    `numpy.where <https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html>`__
    with three parameters.

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    condition
        Type B.
        When True (nonzero), yield X, otherwise yield Y
    X
        Type T.
        values selected at indices where condition is True
    Y
        Type T.
        values selected at indices where condition is False

    Returns
    =======
    output : Var
        Type T.
        Tensor of shape equal to the broadcasted shape of condition, X, and Y.

    Notes
    =====
    Signature: ``ai.onnx@16::Where``.

    Type constraints:
     - B: `tensor(bool)`
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Where(
        _Where.Attributes(),
        _Where.Inputs(
            condition=condition,
            X=X,
            Y=Y,
        ),
    ).outputs.output


def xor(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulted from performing the ``xor`` logical
    operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the logical operator.
    B
        Type T.
        Second input operand for the logical operator.

    Returns
    =======
    C : Var
        Type T1.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@7::Xor``.

    Type constraints:
     - T: `tensor(bool)`
     - T1: `tensor(bool)`
    """
    return _Xor(
        _Xor.Attributes(),
        _Xor.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def const(value: npt.ArrayLike, dtype: npt.DTypeLike = None) -> Var:
    """
    Convenience function for creating constants.

    Shorthand for ``constant(value=np.array(value, dtype))``. The types follow numpy rules.
    """

    return constant(value=np.array(value, dtype))


cum_sum = cumsum
_OPERATORS = {
    "Abs": _Abs,
    "Acos": _Acos,
    "Acosh": _Acosh,
    "Add": _Add,
    "And": _And,
    "ArgMax": _ArgMax,
    "ArgMin": _ArgMin,
    "Asin": _Asin,
    "Asinh": _Asinh,
    "Atan": _Atan,
    "Atanh": _Atanh,
    "AveragePool": _AveragePool,
    "BatchNormalization": _BatchNormalization,
    "Bernoulli": _Bernoulli,
    "BitShift": _BitShift,
    "BlackmanWindow": _BlackmanWindow,
    "Cast": _Cast,
    "CastLike": _CastLike,
    "Ceil": _Ceil,
    "Celu": _Celu,
    "Clip": _Clip,
    "Compress": _Compress,
    "Concat": _Concat,
    "ConcatFromSequence": _ConcatFromSequence,
    "Constant": _Constant,
    "ConstantOfShape": _ConstantOfShape,
    "Conv": _Conv,
    "ConvInteger": _ConvInteger,
    "ConvTranspose": _ConvTranspose,
    "Cos": _Cos,
    "Cosh": _Cosh,
    "CumSum": _CumSum,
    "DFT": _DFT,
    "DepthToSpace": _DepthToSpace,
    "DequantizeLinear": _DequantizeLinear,
    "Det": _Det,
    "Div": _Div,
    "Dropout": _Dropout,
    "DynamicQuantizeLinear": _DynamicQuantizeLinear,
    "Einsum": _Einsum,
    "Elu": _Elu,
    "Equal": _Equal,
    "Erf": _Erf,
    "Exp": _Exp,
    "Expand": _Expand,
    "EyeLike": _EyeLike,
    "Flatten": _Flatten,
    "Floor": _Floor,
    "GRU": _GRU,
    "Gather": _Gather,
    "GatherElements": _GatherElements,
    "GatherND": _GatherND,
    "Gemm": _Gemm,
    "GlobalAveragePool": _GlobalAveragePool,
    "GlobalLpPool": _GlobalLpPool,
    "GlobalMaxPool": _GlobalMaxPool,
    "Greater": _Greater,
    "GreaterOrEqual": _GreaterOrEqual,
    "GridSample": _GridSample,
    "HammingWindow": _HammingWindow,
    "HannWindow": _HannWindow,
    "HardSigmoid": _HardSigmoid,
    "HardSwish": _HardSwish,
    "Hardmax": _Hardmax,
    "Identity": _Identity,
    "If": _If,
    "InstanceNormalization": _InstanceNormalization,
    "IsInf": _IsInf,
    "IsNaN": _IsNaN,
    "LRN": _LRN,
    "LSTM": _LSTM,
    "LayerNormalization": _LayerNormalization,
    "LeakyRelu": _LeakyRelu,
    "Less": _Less,
    "LessOrEqual": _LessOrEqual,
    "Log": _Log,
    "LogSoftmax": _LogSoftmax,
    "Loop": _Loop,
    "LpNormalization": _LpNormalization,
    "LpPool": _LpPool,
    "MatMul": _MatMul,
    "MatMulInteger": _MatMulInteger,
    "Max": _Max,
    "MaxPool": _MaxPool,
    "MaxRoiPool": _MaxRoiPool,
    "MaxUnpool": _MaxUnpool,
    "Mean": _Mean,
    "MeanVarianceNormalization": _MeanVarianceNormalization,
    "MelWeightMatrix": _MelWeightMatrix,
    "Min": _Min,
    "Mod": _Mod,
    "Mul": _Mul,
    "Multinomial": _Multinomial,
    "Neg": _Neg,
    "NegativeLogLikelihoodLoss": _NegativeLogLikelihoodLoss,
    "NonMaxSuppression": _NonMaxSuppression,
    "NonZero": _NonZero,
    "Not": _Not,
    "OneHot": _OneHot,
    "Optional": _Optional,
    "OptionalGetElement": _OptionalGetElement,
    "OptionalHasElement": _OptionalHasElement,
    "Or": _Or,
    "PRelu": _PRelu,
    "Pad": _Pad,
    "Pow": _Pow,
    "QLinearConv": _QLinearConv,
    "QLinearMatMul": _QLinearMatMul,
    "QuantizeLinear": _QuantizeLinear,
    "RNN": _RNN,
    "RandomNormal": _RandomNormal,
    "RandomNormalLike": _RandomNormalLike,
    "RandomUniform": _RandomUniform,
    "RandomUniformLike": _RandomUniformLike,
    "Range": _Range,
    "Reciprocal": _Reciprocal,
    "ReduceL1": _ReduceL1,
    "ReduceL2": _ReduceL2,
    "ReduceLogSum": _ReduceLogSum,
    "ReduceLogSumExp": _ReduceLogSumExp,
    "ReduceMax": _ReduceMax,
    "ReduceMean": _ReduceMean,
    "ReduceMin": _ReduceMin,
    "ReduceProd": _ReduceProd,
    "ReduceSum": _ReduceSum,
    "ReduceSumSquare": _ReduceSumSquare,
    "Relu": _Relu,
    "Reshape": _Reshape,
    "Resize": _Resize,
    "ReverseSequence": _ReverseSequence,
    "RoiAlign": _RoiAlign,
    "Round": _Round,
    "STFT": _STFT,
    "Scan": _Scan,
    "ScatterElements": _ScatterElements,
    "ScatterND": _ScatterND,
    "Selu": _Selu,
    "SequenceAt": _SequenceAt,
    "SequenceConstruct": _SequenceConstruct,
    "SequenceEmpty": _SequenceEmpty,
    "SequenceErase": _SequenceErase,
    "SequenceInsert": _SequenceInsert,
    "SequenceLength": _SequenceLength,
    "SequenceMap": _SequenceMap,
    "Shape": _Shape,
    "Shrink": _Shrink,
    "Sigmoid": _Sigmoid,
    "Sign": _Sign,
    "Sin": _Sin,
    "Sinh": _Sinh,
    "Size": _Size,
    "Slice": _Slice,
    "Softmax": _Softmax,
    "SoftmaxCrossEntropyLoss": _SoftmaxCrossEntropyLoss,
    "Softplus": _Softplus,
    "Softsign": _Softsign,
    "SpaceToDepth": _SpaceToDepth,
    "Split": _Split,
    "SplitToSequence": _SplitToSequence,
    "Sqrt": _Sqrt,
    "Squeeze": _Squeeze,
    "StringNormalizer": _StringNormalizer,
    "Sub": _Sub,
    "Sum": _Sum,
    "Tan": _Tan,
    "Tanh": _Tanh,
    "TfIdfVectorizer": _TfIdfVectorizer,
    "ThresholdedRelu": _ThresholdedRelu,
    "Tile": _Tile,
    "TopK": _TopK,
    "Transpose": _Transpose,
    "Trilu": _Trilu,
    "Unique": _Unique,
    "Unsqueeze": _Unsqueeze,
    "Where": _Where,
    "Xor": _Xor,
}

_CONSTRUCTORS = {
    "Abs": abs,
    "Acos": acos,
    "Acosh": acosh,
    "Add": add,
    "And": and_,
    "ArgMax": arg_max,
    "ArgMin": arg_min,
    "Asin": asin,
    "Asinh": asinh,
    "Atan": atan,
    "Atanh": atanh,
    "AveragePool": average_pool,
    "BatchNormalization": batch_normalization,
    "Bernoulli": bernoulli,
    "BitShift": bit_shift,
    "BlackmanWindow": blackman_window,
    "Cast": cast,
    "CastLike": cast_like,
    "Ceil": ceil,
    "Celu": celu,
    "Clip": clip,
    "Compress": compress,
    "Concat": concat,
    "ConcatFromSequence": concat_from_sequence,
    "Constant": constant,
    "ConstantOfShape": constant_of_shape,
    "Conv": conv,
    "ConvInteger": conv_integer,
    "ConvTranspose": conv_transpose,
    "Cos": cos,
    "Cosh": cosh,
    "CumSum": cumsum,
    "DFT": dft,
    "DepthToSpace": depth_to_space,
    "DequantizeLinear": dequantize_linear,
    "Det": det,
    "Div": div,
    "Dropout": dropout,
    "DynamicQuantizeLinear": dynamic_quantize_linear,
    "Einsum": einsum,
    "Elu": elu,
    "Equal": equal,
    "Erf": erf,
    "Exp": exp,
    "Expand": expand,
    "EyeLike": eye_like,
    "Flatten": flatten,
    "Floor": floor,
    "GRU": gru,
    "Gather": gather,
    "GatherElements": gather_elements,
    "GatherND": gather_nd,
    "Gemm": gemm,
    "GlobalAveragePool": global_average_pool,
    "GlobalLpPool": global_lp_pool,
    "GlobalMaxPool": global_max_pool,
    "Greater": greater,
    "GreaterOrEqual": greater_or_equal,
    "GridSample": grid_sample,
    "HammingWindow": hamming_window,
    "HannWindow": hann_window,
    "HardSigmoid": hard_sigmoid,
    "HardSwish": hard_swish,
    "Hardmax": hardmax,
    "Identity": identity,
    "If": if_,
    "InstanceNormalization": instance_normalization,
    "IsInf": isinf,
    "IsNaN": isnan,
    "LRN": lrn,
    "LSTM": lstm,
    "LayerNormalization": layer_normalization,
    "LeakyRelu": leaky_relu,
    "Less": less,
    "LessOrEqual": less_or_equal,
    "Log": log,
    "LogSoftmax": log_softmax,
    "Loop": loop,
    "LpNormalization": lp_normalization,
    "LpPool": lp_pool,
    "MatMul": matmul,
    "MatMulInteger": matmul_integer,
    "Max": max,
    "MaxPool": max_pool,
    "MaxRoiPool": max_roi_pool,
    "MaxUnpool": max_unpool,
    "Mean": mean,
    "MeanVarianceNormalization": mean_variance_normalization,
    "MelWeightMatrix": mel_weight_matrix,
    "Min": min,
    "Mod": mod,
    "Mul": mul,
    "Multinomial": multinomial,
    "Neg": neg,
    "NegativeLogLikelihoodLoss": negative_log_likelihood_loss,
    "NonMaxSuppression": non_max_suppression,
    "NonZero": non_zero,
    "Not": not_,
    "OneHot": one_hot,
    "Optional": optional,
    "OptionalGetElement": optional_get_element,
    "OptionalHasElement": optional_has_element,
    "Or": or_,
    "PRelu": prelu,
    "Pad": pad,
    "Pow": pow,
    "QLinearConv": qlinear_conv,
    "QLinearMatMul": qlinear_matmul,
    "QuantizeLinear": quantize_linear,
    "RNN": rnn,
    "RandomNormal": random_normal,
    "RandomNormalLike": random_normal_like,
    "RandomUniform": random_uniform,
    "RandomUniformLike": random_uniform_like,
    "Range": range,
    "Reciprocal": reciprocal,
    "ReduceL1": reduce_l1,
    "ReduceL2": reduce_l2,
    "ReduceLogSum": reduce_log_sum,
    "ReduceLogSumExp": reduce_log_sum_exp,
    "ReduceMax": reduce_max,
    "ReduceMean": reduce_mean,
    "ReduceMin": reduce_min,
    "ReduceProd": reduce_prod,
    "ReduceSum": reduce_sum,
    "ReduceSumSquare": reduce_sum_square,
    "Relu": relu,
    "Reshape": reshape,
    "Resize": resize,
    "ReverseSequence": reverse_sequence,
    "RoiAlign": roi_align,
    "Round": round,
    "STFT": stft,
    "Scan": scan,
    "ScatterElements": scatter_elements,
    "ScatterND": scatter_nd,
    "Selu": selu,
    "SequenceAt": sequence_at,
    "SequenceConstruct": sequence_construct,
    "SequenceEmpty": sequence_empty,
    "SequenceErase": sequence_erase,
    "SequenceInsert": sequence_insert,
    "SequenceLength": sequence_length,
    "SequenceMap": sequence_map,
    "Shape": shape,
    "Shrink": shrink,
    "Sigmoid": sigmoid,
    "Sign": sign,
    "Sin": sin,
    "Sinh": sinh,
    "Size": size,
    "Slice": slice,
    "Softmax": softmax,
    "SoftmaxCrossEntropyLoss": softmax_cross_entropy_loss,
    "Softplus": softplus,
    "Softsign": softsign,
    "SpaceToDepth": space_to_depth,
    "Split": split,
    "SplitToSequence": split_to_sequence,
    "Sqrt": sqrt,
    "Squeeze": squeeze,
    "StringNormalizer": string_normalizer,
    "Sub": sub,
    "Sum": sum,
    "Tan": tan,
    "Tanh": tanh,
    "TfIdfVectorizer": tf_idf_vectorizer,
    "ThresholdedRelu": thresholded_relu,
    "Tile": tile,
    "TopK": top_k,
    "Transpose": transpose,
    "Trilu": trilu,
    "Unique": unique,
    "Unsqueeze": unsqueeze,
    "Where": where,
    "Xor": xor,
}

__all__ = [fun.__name__ for fun in _CONSTRUCTORS.values()]
