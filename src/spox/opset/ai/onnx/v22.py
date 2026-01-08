# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: E741 -- Allow ambiguous variable name
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spox._attributes import (
    AttrDtype,
    AttrFloat32,
    AttrFloat32s,
    AttrInt64,
    AttrInt64s,
    AttrString,
    AttrStrings,
)
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._node import OpType
from spox._standard import StandardNode
from spox._var import (
    Var,
    _VarInfo,
    create_prop_dict,
    unwrap_vars,
)
from spox.opset.ai.onnx.v21 import (
    _DFT,
    _LRN,
    _STFT,
    _Abs,
    _Add,
    _AffineGrid,
    _And,
    _ArgMax,
    _ArgMin,
    _BatchNormalization,
    _BitShift,
    _BitwiseAnd,
    _BitwiseNot,
    _BitwiseOr,
    _BitwiseXor,
    _BlackmanWindow,
    _Cast,
    _CastLike,
    _Ceil,
    _Celu,
    _CenterCropPad,
    _Clip,
    _Col2Im,
    _Compress,
    _Concat,
    _ConcatFromSequence,
    _Constant,
    _ConstantOfShape,
    _ConvInteger,
    _CumSum,
    _DepthToSpace,
    _DequantizeLinear,
    _Div,
    _DynamicQuantizeLinear,
    _Einsum,
    _Equal,
    _Erf,
    _Exp,
    _Expand,
    _Flatten,
    _Floor,
    _Gather,
    _GatherElements,
    _GatherND,
    _Gelu,
    _Gemm,
    _Greater,
    _GreaterOrEqual,
    _GroupNormalization,
    _HammingWindow,
    _HannWindow,
    _Hardmax,
    _Identity,
    _If,
    _ImageDecoder,
    _IsInf,
    _IsNaN,
    _LayerNormalization,
    _LeakyRelu,
    _Less,
    _LessOrEqual,
    _Log,
    _LogSoftmax,
    _Loop,
    _MatMul,
    _MatMulInteger,
    _Max,
    _Mean,
    _MeanVarianceNormalization,
    _MelWeightMatrix,
    _Min,
    _Mod,
    _Mul,
    _Neg,
    _NonMaxSuppression,
    _NonZero,
    _Not,
    _OneHot,
    _Optional,
    _OptionalGetElement,
    _OptionalHasElement,
    _Or,
    _Pad,
    _Pow,
    _PRelu,
    _QLinearConv,
    _QLinearMatMul,
    _QuantizeLinear,
    _Range,
    _Reciprocal,
    _ReduceL1,
    _ReduceL2,
    _ReduceLogSum,
    _ReduceLogSumExp,
    _ReduceMax,
    _ReduceMean,
    _ReduceMin,
    _ReduceProd,
    _ReduceSum,
    _ReduceSumSquare,
    _RegexFullMatch,
    _Relu,
    _Reshape,
    _Resize,
    _ReverseSequence,
    _Scan,
    _ScatterElements,
    _ScatterND,
    _SequenceAt,
    _SequenceConstruct,
    _SequenceEmpty,
    _SequenceErase,
    _SequenceInsert,
    _SequenceLength,
    _SequenceMap,
    _Shape,
    _Shrink,
    _Sigmoid,
    _Sign,
    _Size,
    _Slice,
    _Softmax,
    _SoftmaxCrossEntropyLoss,
    _SpaceToDepth,
    _Split,
    _SplitToSequence,
    _Sqrt,
    _Squeeze,
    _StringConcat,
    _StringNormalizer,
    _StringSplit,
    _Sub,
    _Sum,
    _Tanh,
    _TfIdfVectorizer,
    _Tile,
    _TopK,
    _Transpose,
    _Trilu,
    _Unique,
    _Unsqueeze,
    _Where,
    _Xor,
    abs,
    add,
    affine_grid,
    and_,
    arg_max,
    arg_min,
    batch_normalization,
    bit_shift,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    blackman_window,
    cast,
    cast_like,
    ceil,
    celu,
    center_crop_pad,
    clip,
    col2_im,
    compress,
    concat,
    concat_from_sequence,
    constant,
    constant_of_shape,
    conv_integer,
    cumsum,
    depth_to_space,
    dequantize_linear,
    dft,
    div,
    dynamic_quantize_linear,
    einsum,
    equal,
    erf,
    exp,
    expand,
    flatten,
    floor,
    gather,
    gather_elements,
    gather_nd,
    gelu,
    gemm,
    greater,
    greater_or_equal,
    group_normalization,
    hamming_window,
    hann_window,
    hardmax,
    identity,
    if_,
    image_decoder,
    isinf,
    isnan,
    layer_normalization,
    leaky_relu,
    less,
    less_or_equal,
    log,
    log_softmax,
    loop,
    lrn,
    matmul,
    matmul_integer,
    max,
    mean,
    mean_variance_normalization,
    mel_weight_matrix,
    min,
    mod,
    mul,
    neg,
    non_max_suppression,
    non_zero,
    not_,
    one_hot,
    optional,
    optional_get_element,
    optional_has_element,
    or_,
    pad,
    pow,
    prelu,
    qlinear_conv,
    qlinear_matmul,
    quantize_linear,
    range,
    reciprocal,
    reduce_l1,
    reduce_l2,
    reduce_log_sum,
    reduce_log_sum_exp,
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_prod,
    reduce_sum,
    reduce_sum_square,
    regex_full_match,
    relu,
    reshape,
    resize,
    reverse_sequence,
    scan,
    scatter_elements,
    scatter_nd,
    sequence_at,
    sequence_construct,
    sequence_empty,
    sequence_erase,
    sequence_insert,
    sequence_length,
    sequence_map,
    shape,
    shrink,
    sigmoid,
    sign,
    size,
    slice,
    softmax,
    softmax_cross_entropy_loss,
    space_to_depth,
    split,
    split_to_sequence,
    sqrt,
    squeeze,
    stft,
    string_concat,
    string_normalizer,
    string_split,
    sub,
    sum,
    tanh,
    tf_idf_vectorizer,
    tile,
    top_k,
    transpose,
    trilu,
    unique,
    unsqueeze,
    where,
    xor,
)


class _Acos(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Acos", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Acosh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Acosh", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Asin(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Asin", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Asinh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Asinh", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Atan(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Atan", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Atanh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Atanh", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _AveragePool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        ceil_mode: AttrInt64
        count_include_pad: AttrInt64
        dilations: AttrInt64s | None
        kernel_shape: AttrInt64s
        pads: AttrInt64s | None
        strides: AttrInt64s | None

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("AveragePool", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Bernoulli(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype | None
        seed: AttrFloat32 | None

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Bernoulli", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Conv(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        dilations: AttrInt64s | None
        group: AttrInt64
        kernel_shape: AttrInt64s | None
        pads: AttrInt64s | None
        strides: AttrInt64s | None

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo
        W: _VarInfo
        B: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("Conv", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ConvTranspose(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        dilations: AttrInt64s | None
        group: AttrInt64
        kernel_shape: AttrInt64s | None
        output_padding: AttrInt64s | None
        output_shape: AttrInt64s | None
        pads: AttrInt64s | None
        strides: AttrInt64s | None

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo
        W: _VarInfo
        B: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("ConvTranspose", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Cos(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Cos", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Cosh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Cosh", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _DeformConv(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dilations: AttrInt64s | None
        group: AttrInt64
        kernel_shape: AttrInt64s | None
        offset_group: AttrInt64
        pads: AttrInt64s | None
        strides: AttrInt64s | None

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo
        W: _VarInfo
        offset: _VarInfo
        B: _VarInfo | None
        mask: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("DeformConv", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Det(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("Det", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Dropout(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        seed: AttrInt64 | None

    @dataclass
    class Inputs(BaseInputs):
        data: _VarInfo
        ratio: _VarInfo | None
        training_mode: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo
        mask: _VarInfo | None

    op_type = OpType("Dropout", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Elu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("Elu", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _EyeLike(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype | None
        k: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("EyeLike", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GRU(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        activation_alpha: AttrFloat32s | None
        activation_beta: AttrFloat32s | None
        activations: AttrStrings | None
        clip: AttrFloat32 | None
        direction: AttrString
        hidden_size: AttrInt64 | None
        layout: AttrInt64
        linear_before_reset: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo
        W: _VarInfo
        R: _VarInfo
        B: _VarInfo | None
        sequence_lens: _VarInfo | None
        initial_h: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo | None
        Y_h: _VarInfo | None

    op_type = OpType("GRU", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GlobalAveragePool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("GlobalAveragePool", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GlobalLpPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        p: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("GlobalLpPool", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GlobalMaxPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("GlobalMaxPool", "", 22)

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
        X: _VarInfo
        grid: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("GridSample", "", 22)

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
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("HardSigmoid", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _HardSwish(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("HardSwish", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _InstanceNormalization(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        epsilon: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo
        scale: _VarInfo
        B: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("InstanceNormalization", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LSTM(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        activation_alpha: AttrFloat32s | None
        activation_beta: AttrFloat32s | None
        activations: AttrStrings | None
        clip: AttrFloat32 | None
        direction: AttrString
        hidden_size: AttrInt64 | None
        input_forget: AttrInt64
        layout: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo
        W: _VarInfo
        R: _VarInfo
        B: _VarInfo | None
        sequence_lens: _VarInfo | None
        initial_h: _VarInfo | None
        initial_c: _VarInfo | None
        P: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo | None
        Y_h: _VarInfo | None
        Y_c: _VarInfo | None

    op_type = OpType("LSTM", "", 22)

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
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("LpNormalization", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LpPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        ceil_mode: AttrInt64
        dilations: AttrInt64s | None
        kernel_shape: AttrInt64s
        p: AttrInt64
        pads: AttrInt64s | None
        strides: AttrInt64s | None

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("LpPool", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _MaxPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        ceil_mode: AttrInt64
        dilations: AttrInt64s | None
        kernel_shape: AttrInt64s
        pads: AttrInt64s | None
        storage_order: AttrInt64
        strides: AttrInt64s | None

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo
        Indices: _VarInfo | None

    op_type = OpType("MaxPool", "", 22)

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
        X: _VarInfo
        rois: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("MaxRoiPool", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _MaxUnpool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        kernel_shape: AttrInt64s
        pads: AttrInt64s | None
        strides: AttrInt64s | None

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo
        I: _VarInfo
        output_shape: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("MaxUnpool", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Mish(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("Mish", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Multinomial(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype
        sample_size: AttrInt64
        seed: AttrFloat32 | None

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Multinomial", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _NegativeLogLikelihoodLoss(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        ignore_index: AttrInt64 | None
        reduction: AttrString

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo
        target: _VarInfo
        weight: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        loss: _VarInfo

    op_type = OpType("NegativeLogLikelihoodLoss", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _RNN(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        activation_alpha: AttrFloat32s | None
        activation_beta: AttrFloat32s | None
        activations: AttrStrings
        clip: AttrFloat32 | None
        direction: AttrString
        hidden_size: AttrInt64 | None
        layout: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo
        W: _VarInfo
        R: _VarInfo
        B: _VarInfo | None
        sequence_lens: _VarInfo | None
        initial_h: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo | None
        Y_h: _VarInfo | None

    op_type = OpType("RNN", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _RandomNormal(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype
        mean: AttrFloat32
        scale: AttrFloat32
        seed: AttrFloat32 | None
        shape: AttrInt64s

    Inputs = BaseInputs

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("RandomNormal", "", 22)

    attrs: Attributes
    inputs: BaseInputs
    outputs: Outputs


class _RandomNormalLike(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype | None
        mean: AttrFloat32
        scale: AttrFloat32
        seed: AttrFloat32 | None

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("RandomNormalLike", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _RandomUniform(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype
        high: AttrFloat32
        low: AttrFloat32
        seed: AttrFloat32 | None
        shape: AttrInt64s

    Inputs = BaseInputs

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("RandomUniform", "", 22)

    attrs: Attributes
    inputs: BaseInputs
    outputs: Outputs


class _RandomUniformLike(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dtype: AttrDtype | None
        high: AttrFloat32
        low: AttrFloat32
        seed: AttrFloat32 | None

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("RandomUniformLike", "", 22)

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
        X: _VarInfo
        rois: _VarInfo
        batch_indices: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("RoiAlign", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Round(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("Round", "", 22)

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
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("Selu", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Sin(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Sin", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Sinh(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Sinh", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Softplus(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("Softplus", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Softsign(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Softsign", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Tan(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Tan", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ThresholdedRelu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        alpha: AttrFloat32

    @dataclass
    class Inputs(BaseInputs):
        X: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: _VarInfo

    op_type = OpType("ThresholdedRelu", "", 22)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


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
    Signature: ``ai.onnx@22::Acos``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Acos(
            _Acos.Attributes(),
            _Acos.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Acosh``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Acosh(
            _Acosh.Attributes(),
            _Acosh.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Asin``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Asin(
            _Asin.Attributes(),
            _Asin.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Asinh``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Asinh(
            _Asinh.Attributes(),
            _Asinh.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Atan``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Atan(
            _Atan.Attributes(),
            _Atan.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Atanh``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Atanh(
            _Atanh.Attributes(),
            _Atanh.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def average_pool(
    X: Var,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    count_include_pad: int = 0,
    dilations: Iterable[int] | None = None,
    kernel_shape: Iterable[int],
    pads: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
) -> Var:
    r"""
    AveragePool consumes an input tensor X and applies average pooling
    across the tensor according to kernel sizes, stride sizes, and pad
    lengths. average pooling consisting of computing the average on all
    values of a subset of the input tensor according to the kernel size and
    downsampling the data into the output tensor Y for further processing.
    The output spatial shape is calculated differently depending on whether
    explicit padding is used, where pads is employed, or auto padding is
    used, where auto_pad is utilized. With explicit padding
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
    Signature: ``ai.onnx@22::AveragePool``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _AveragePool(
            _AveragePool.Attributes(
                auto_pad=AttrString(auto_pad, name="auto_pad"),
                ceil_mode=AttrInt64(ceil_mode, name="ceil_mode"),
                count_include_pad=AttrInt64(
                    count_include_pad, name="count_include_pad"
                ),
                dilations=AttrInt64s.maybe(dilations, name="dilations"),
                kernel_shape=AttrInt64s(kernel_shape, name="kernel_shape"),
                pads=AttrInt64s.maybe(pads, name="pads"),
                strides=AttrInt64s.maybe(strides, name="strides"),
            ),
            _AveragePool.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


def bernoulli(
    input: Var,
    *,
    dtype: npt.DTypeLike | None = None,
    seed: float | None = None,
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
    Signature: ``ai.onnx@22::Bernoulli``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Bernoulli(
            _Bernoulli.Attributes(
                dtype=AttrDtype.maybe(dtype, name="dtype"),
                seed=AttrFloat32.maybe(seed, name="seed"),
            ),
            _Bernoulli.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def conv(
    X: Var,
    W: Var,
    B: Var | None = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: Iterable[int] | None = None,
    group: int = 1,
    kernel_shape: Iterable[int] | None = None,
    pads: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
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
    Signature: ``ai.onnx@22::Conv``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        W=W,
        B=B,
    )
    output_vars = (
        _Conv(
            _Conv.Attributes(
                auto_pad=AttrString(auto_pad, name="auto_pad"),
                dilations=AttrInt64s.maybe(dilations, name="dilations"),
                group=AttrInt64(group, name="group"),
                kernel_shape=AttrInt64s.maybe(kernel_shape, name="kernel_shape"),
                pads=AttrInt64s.maybe(pads, name="pads"),
                strides=AttrInt64s.maybe(strides, name="strides"),
            ),
            _Conv.Inputs(
                X=unwrap_vars(X),
                W=unwrap_vars(W),
                B=unwrap_vars(B),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


def conv_transpose(
    X: Var,
    W: Var,
    B: Var | None = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: Iterable[int] | None = None,
    group: int = 1,
    kernel_shape: Iterable[int] | None = None,
    output_padding: Iterable[int] | None = None,
    output_shape: Iterable[int] | None = None,
    pads: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
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
    Signature: ``ai.onnx@22::ConvTranspose``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        W=W,
        B=B,
    )
    output_vars = (
        _ConvTranspose(
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
                X=unwrap_vars(X),
                W=unwrap_vars(W),
                B=unwrap_vars(B),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Cos``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Cos(
            _Cos.Attributes(),
            _Cos.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Cosh``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Cosh(
            _Cosh.Attributes(),
            _Cosh.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def deform_conv(
    X: Var,
    W: Var,
    offset: Var,
    B: Var | None = None,
    mask: Var | None = None,
    *,
    dilations: Iterable[int] | None = None,
    group: int = 1,
    kernel_shape: Iterable[int] | None = None,
    offset_group: int = 1,
    pads: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
) -> Var:
    r"""
    Performs deformable convolution as described in
    https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
    This operator specification supports the general N-D case. Note that
    most common use cases have 2D or 3D data.

    Parameters
    ==========
    X
        Type T.
        Input data tensor. For 2D image data, it has shape (N, C, H, W) where N
        is the batch size, C is the number of input channels, and H and W are
        the height and width. In general, the shape is (N, C, D1, D2, ... , Dn)
        for n-dimensional data, where D1 to Dn are the spatial dimension sizes.
        Most common use cases have n = 2 or 3.
    W
        Type T.
        Weight tensor that will be used in the convolutions. It has shape (oC,
        C/group, kH, kW), where oC is the number of output channels and kH and
        kW are the kernel height and width. For more than 2 dimensions, it has
        shape (oC, C/group, k1, k2, ... , kn).
    offset
        Type T.
        Offset tensor denoting the offset for the sampling locations in the
        convolution kernel. It has shape (N, offset_group \* kH \* kW \* 2, oH,
        oW) for 2D data or (N, offset_group \* k1 \* k2 \* ... \* kn \* n, o1,
        o2, ... , on) for nD data. Use linear interpolationfor fractional offset
        values. Sampling locations outside of the padded input tensor gives
        zero.
    B
        Type T.
        Optional 1D bias of length oC to be added to the convolution. Default is
        a tensor of zeros.
    mask
        Type T.
        The mask tensor to be applied to each position in the convolution
        kernel. It has shape (N, offset_group \* kH \* kW, oH, oW) for 2D data
        or (N, offset_group \* k1 \* k2 \* ... \* kn \* n, o1, o2, ... , on) for
        nD data. Default is a tensor of ones.
    dilations
        Attribute.
        Dilation value along each spatial axis of the kernel. Default is 1 along
        each axis.
    group
        Attribute.
        Number of groups the input and output channels, C and oC, are divided
        into. C and oC must both be divisible by group. Default is 1.
    kernel_shape
        Attribute.
        Shape of the convolution kernel. If not present, it is inferred from the
        shape of input W.
    offset_group
        Attribute.
        Number of groups of offset. C must be divisible by offset_group. Default
        is 1.
    pads
        Attribute.
        Padding for the beginning and end along each spatial axis. The values
        represent the number of pixels added to the beginning and end of the
        corresponding axis and can take any nonnegative value. The format should
        be as follows: [x1_begin, x2_begin, ..., x1_end, x2_end, ...], where
        xi_begin is the number of pixels added at the beginning of axis ``i``
        and xi_end is the number of pixels added at the end of axis ``i``.
        Default is 0 along each axis.
    strides
        Attribute.
        Stride along each spatial axis. Default is 1 along each axis.

    Returns
    =======
    Y : Var
        Type T.
        Output data tensor that contains the result of convolution. It has shape
        (N, oC, oH, oW) for 2D data or (N, oC, o1, o2, ..., on) for nD data

    Notes
    =====
    Signature: ``ai.onnx@22::DeformConv``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        W=W,
        offset=offset,
        B=B,
        mask=mask,
    )
    output_vars = (
        _DeformConv(
            _DeformConv.Attributes(
                dilations=AttrInt64s.maybe(dilations, name="dilations"),
                group=AttrInt64(group, name="group"),
                kernel_shape=AttrInt64s.maybe(kernel_shape, name="kernel_shape"),
                offset_group=AttrInt64(offset_group, name="offset_group"),
                pads=AttrInt64s.maybe(pads, name="pads"),
                strides=AttrInt64s.maybe(strides, name="strides"),
            ),
            _DeformConv.Inputs(
                X=unwrap_vars(X),
                W=unwrap_vars(W),
                offset=unwrap_vars(offset),
                B=unwrap_vars(B),
                mask=unwrap_vars(mask),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Det``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _Det(
            _Det.Attributes(),
            _Det.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


def dropout(
    data: Var,
    ratio: Var | None = None,
    training_mode: Var | None = None,
    *,
    seed: int | None = None,
) -> tuple[Var, Var]:
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
        The ratio of random dropout, with value in [0, 1). If set to 0, the
        output would be a simple copy of the input. If it's non-zero, output
        will be a random dropout of the scaled input, which is typically the
        case during training. It is an optional value, if not specified it will
        default to 0.5.
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
    Signature: ``ai.onnx@22::Dropout``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
     - T2: `tensor(bool)`
    """
    input_prop_values = create_prop_dict(
        data=data,
        ratio=ratio,
        training_mode=training_mode,
    )
    output_vars = (
        _Dropout(
            _Dropout.Attributes(
                seed=AttrInt64.maybe(seed, name="seed"),
            ),
            _Dropout.Inputs(
                data=unwrap_vars(data),
                ratio=unwrap_vars(ratio),
                training_mode=unwrap_vars(training_mode),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        ._unpack_to_any()
    )
    return output_vars  # type: ignore


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
        Input tensor
    alpha
        Attribute.
        Coefficient of ELU.

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@22::Elu``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _Elu(
            _Elu.Attributes(
                alpha=AttrFloat32(alpha, name="alpha"),
            ),
            _Elu.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


def eye_like(
    input: Var,
    *,
    dtype: npt.DTypeLike | None = None,
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
        specified, the data type of the input tensor T1 is used.
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
    Signature: ``ai.onnx@22::EyeLike``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _EyeLike(
            _EyeLike.Attributes(
                dtype=AttrDtype.maybe(dtype, name="dtype"),
                k=AttrInt64(k, name="k"),
            ),
            _EyeLike.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def gru(
    X: Var,
    W: Var,
    R: Var,
    B: Var | None = None,
    sequence_lens: Var | None = None,
    initial_h: Var | None = None,
    *,
    activation_alpha: Iterable[float] | None = None,
    activation_beta: Iterable[float] | None = None,
    activations: Iterable[str] | None = None,
    clip: float | None = None,
    direction: str = "forward",
    hidden_size: int | None = None,
    layout: int = 0,
    linear_before_reset: int = 0,
) -> tuple[Var, Var]:
    r"""
    Computes an one-layer GRU. This operator is usually supported via some
    custom implementation such as CuDNN.

    Notations:

    - ``X`` - input tensor
    - ``z`` - update gate
    - ``r`` - reset gate
    - ``h`` - hidden gate
    - ``t`` - time step (t-1 means previous time step)
    - ``W[zrh]`` - W parameter weight matrix for update, reset, and hidden
      gates
    - ``R[zrh]`` - R recurrence weight matrix for update, reset, and hidden
      gates
    - ``Wb[zrh]`` - W bias vectors for update, reset, and hidden gates
    - ``Rb[zrh]`` - R bias vectors for update, reset, and hidden gates
    - ``WB[zrh]`` - W parameter weight matrix for backward update, reset,
      and hidden gates
    - ``RB[zrh]`` - R recurrence weight matrix for backward update, reset,
      and hidden gates
    - ``WBb[zrh]`` - W bias vectors for backward update, reset, and hidden
      gates
    - ``RBb[zrh]`` - R bias vectors for backward update, reset, and hidden
      gates
    - ``H`` - Hidden state
    - ``num_directions`` - 2 if direction == bidirectional else 1

    Activation functions:

    - Relu(x) - max(0, x)
    - Tanh(x) - (1 - e^{-2x})/(1 + e^{-2x})
    - Sigmoid(x) - 1/(1 + e^{-x})

    NOTE: Below are optional

    - Affine(x) - alpha \* x + beta
    - LeakyRelu(x) - x if x >= 0 else alpha \* x
    - ThresholdedRelu(x) - x if x >= alpha else 0
    - ScaledTanh(x) - alpha \* Tanh(beta \* x)
    - HardSigmoid(x) - min(max(alpha \* x + beta, 0), 1)
    - Elu(x) - x if x >= 0 else alpha \* (e^x - 1)
    - Softsign(x) - x/(1 + \|x\|)
    - Softplus(x) - log(1 + e^x)

    Equations (Default: f=Sigmoid, g=Tanh):

    - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when
      linear_before_reset = 0
    - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when
      linear_before_reset != 0
    - Ht = (1 - zt) (.) ht + zt (.) Ht-1 This operator has **optional**
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
    Signature: ``ai.onnx@22::GRU``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T1: `tensor(int32)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        W=W,
        R=R,
        B=B,
        sequence_lens=sequence_lens,
        initial_h=initial_h,
    )
    output_vars = (
        _GRU(
            _GRU.Attributes(
                activation_alpha=AttrFloat32s.maybe(
                    activation_alpha, name="activation_alpha"
                ),
                activation_beta=AttrFloat32s.maybe(
                    activation_beta, name="activation_beta"
                ),
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
                X=unwrap_vars(X),
                W=unwrap_vars(W),
                R=unwrap_vars(R),
                B=unwrap_vars(B),
                sequence_lens=unwrap_vars(sequence_lens),
                initial_h=unwrap_vars(initial_h),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        ._unpack_to_any()
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::GlobalAveragePool``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _GlobalAveragePool(
            _GlobalAveragePool.Attributes(),
            _GlobalAveragePool.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::GlobalLpPool``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _GlobalLpPool(
            _GlobalLpPool.Attributes(
                p=AttrInt64(p, name="p"),
            ),
            _GlobalLpPool.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::GlobalMaxPool``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _GlobalMaxPool(
            _GlobalMaxPool.Attributes(),
            _GlobalMaxPool.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


def grid_sample(
    X: Var,
    grid: Var,
    *,
    align_corners: int = 0,
    mode: str = "linear",
    padding_mode: str = "zeros",
) -> Var:
    r"""
    Given an input ``X`` and a flow-field ``grid``, computes the output
    ``Y`` using ``X`` values and pixel locations from the ``grid``. For
    spatial input ``X`` with shape (N, C, H, W), the ``grid`` will have
    shape (N, H_out, W_out, 2), the output ``Y`` will have shape (N, C,
    H_out, W_out). For volumetric input ``X`` with shape (N, C, D, H, W),
    the ``grid`` will have shape (N, D_out, H_out, W_out, 3), the output
    ``Y`` will have shape (N, C, D_out, H_out, W_out). More generally, for
    an input ``X`` of rank r+2 with shape (N, C, d1, d2, ..., dr), the
    ``grid`` will have shape (N, D1_out, D2_out, ..., Dr_out, r), the output
    ``Y`` will have shape (N, C, D1_out, D2_out, ..., Dr_out).

    The tensor ``X`` contains values at centers of square pixels (voxels,
    etc) locations such as (n, c, d1_in, d2_in, ..., dr_in). The (n, d1_out,
    d2_out, ..., dr_out, :) values from the tensor ``grid`` are the
    normalized positions for interpolating the values at the (n, c, d1_out,
    d2_out, ..., dr_out) locations from the output tensor ``Y`` using a
    specified interpolation method (the mode) and a padding mode (for
    ``grid`` positions falling outside the 2-dimensional image).

    For example, the values in ``grid[n, h_out, w_out, :]`` are size-2
    vectors specifying normalized positions in the 2-dimensional space of
    ``X``. They are used to interpolate output values of
    ``Y[n, c, h_out, w_out]``.

    The GridSample operator is often used in doing grid generator and
    sampler in the `Spatial Transformer
    Networks <https://arxiv.org/abs/1506.02025>`__. See also in
    `torch.nn.functional.grid_sample <https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html>`__.

    Parameters
    ==========
    X
        Type T1.
        Input tensor of rank r+2 that has shape (N, C, D1, D2, ..., Dr), where N
        is the batch size, C is the number of channels, D1, D2, ..., Dr are the
        spatial dimensions.
    grid
        Type T2.
        Input offset of shape (N, D1_out, D2_out, ..., Dr_out, r), where D1_out,
        D2_out, ..., Dr_out are the spatial dimensions of the grid and output,
        and r is the number of spatial dimensions. Grid specifies the sampling
        locations normalized by the input spatial dimensions. Therefore, it
        should have most values in the range of [-1, 1]. If the grid has values
        outside the range of [-1, 1], the corresponding outputs will be handled
        as defined by padding_mode. Following computer vision convention, the
        coordinates in the length-r location vector are listed from the
        innermost tensor dimension to the outermost, the opposite of regular
        tensor indexing.
    align_corners
        Attribute.
        If align_corners=1, the extrema (-1 and 1) are considered as referring
        to the center points of the input's corner pixels (voxels, etc.). If
        align_corners=0, they are instead considered as referring to the corner
        points of the input's corner pixels (voxels, etc.), making the sampling
        more resolution agnostic.
    mode
        Attribute.
        Three interpolation modes: linear (default), nearest and cubic. The
        "linear" mode includes linear and N-linear interpolation modes depending
        on the number of spatial dimensions of the input tensor (i.e. linear for
        1 spatial dimension, bilinear for 2 spatial dimensions, etc.). The
        "cubic" mode also includes N-cubic interpolation modes following the
        same rules. The "nearest" mode rounds to the nearest even index when the
        sampling point falls halfway between two indices.
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
        Output tensor of rank r+2 that has shape (N, C, D1_out, D2_out, ...,
        Dr_out) of the sampled values. For integer input types, intermediate
        values are computed as floating point and cast to integer at the end.

    Notes
    =====
    Signature: ``ai.onnx@22::GridSample``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        grid=grid,
    )
    output_vars = (
        _GridSample(
            _GridSample.Attributes(
                align_corners=AttrInt64(align_corners, name="align_corners"),
                mode=AttrString(mode, name="mode"),
                padding_mode=AttrString(padding_mode, name="padding_mode"),
            ),
            _GridSample.Inputs(
                X=unwrap_vars(X),
                grid=unwrap_vars(grid),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::HardSigmoid``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _HardSigmoid(
            _HardSigmoid.Attributes(
                alpha=AttrFloat32(alpha, name="alpha"),
                beta=AttrFloat32(beta, name="beta"),
            ),
            _HardSigmoid.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::HardSwish``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _HardSwish(
            _HardSwish.Attributes(),
            _HardSwish.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::InstanceNormalization``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
        scale=scale,
        B=B,
    )
    output_vars = (
        _InstanceNormalization(
            _InstanceNormalization.Attributes(
                epsilon=AttrFloat32(epsilon, name="epsilon"),
            ),
            _InstanceNormalization.Inputs(
                input=unwrap_vars(input),
                scale=unwrap_vars(scale),
                B=unwrap_vars(B),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def lstm(
    X: Var,
    W: Var,
    R: Var,
    B: Var | None = None,
    sequence_lens: Var | None = None,
    initial_h: Var | None = None,
    initial_c: Var | None = None,
    P: Var | None = None,
    *,
    activation_alpha: Iterable[float] | None = None,
    activation_beta: Iterable[float] | None = None,
    activations: Iterable[str] | None = None,
    clip: float | None = None,
    direction: str = "forward",
    hidden_size: int | None = None,
    input_forget: int = 0,
    layout: int = 0,
) -> tuple[Var, Var, Var]:
    r"""
    Computes an one-layer LSTM. This operator is usually supported via some
    custom implementation such as CuDNN.

    Notations:

    - ``X`` - input tensor
    - ``i`` - input gate
    - ``o`` - output gate
    - ``f`` - forget gate
    - ``c`` - cell gate
    - ``t`` - time step (t-1 means previous time step)
    - ``W[iofc]`` - W parameter weight matrix for input, output, forget, and
      cell gates
    - ``R[iofc]`` - R recurrence weight matrix for input, output, forget,
      and cell gates
    - ``Wb[iofc]`` - W bias vectors for input, output, forget, and cell
      gates
    - ``Rb[iofc]`` - R bias vectors for input, output, forget, and cell
      gates
    - ``P[iof]`` - P peephole weight vector for input, output, and forget
      gates
    - ``WB[iofc]`` - W parameter weight matrix for backward input, output,
      forget, and cell gates
    - ``RB[iofc]`` - R recurrence weight matrix for backward input, output,
      forget, and cell gates
    - ``WBb[iofc]`` - W bias vectors for backward input, output, forget, and
      cell gates
    - ``RBb[iofc]`` - R bias vectors for backward input, output, forget, and
      cell gates
    - ``PB[iof]`` - P peephole weight vector for backward input, output, and
      forget gates
    - ``H`` - Hidden state
    - ``num_directions`` - 2 if direction == bidirectional else 1

    Activation functions:

    - Relu(x) - max(0, x)
    - Tanh(x) - (1 - e^{-2x})/(1 + e^{-2x})
    - Sigmoid(x) - 1/(1 + e^{-x})

    NOTE: Below are optional

    - Affine(x) - alpha*x + beta
    - LeakyRelu(x) - x if x >= 0 else alpha \* x
    - ThresholdedRelu(x) - x if x >= alpha else 0
    - ScaledTanh(x) - alpha\ *Tanh(beta*\ x)
    - HardSigmoid(x) - min(max(alpha*x + beta, 0), 1)
    - Elu(x) - x if x >= 0 else alpha*(e^x - 1)
    - Softsign(x) - x/(1 + \|x\|)
    - Softplus(x) - log(1 + e^x)

    Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

    - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    - Ct = ft (.) Ct-1 + it (.) ct
    - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    - Ht = ot (.) h(Ct) This operator has **optional** inputs/outputs. See
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
    Signature: ``ai.onnx@22::LSTM``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T1: `tensor(int32)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        W=W,
        R=R,
        B=B,
        sequence_lens=sequence_lens,
        initial_h=initial_h,
        initial_c=initial_c,
        P=P,
    )
    output_vars = (
        _LSTM(
            _LSTM.Attributes(
                activation_alpha=AttrFloat32s.maybe(
                    activation_alpha, name="activation_alpha"
                ),
                activation_beta=AttrFloat32s.maybe(
                    activation_beta, name="activation_beta"
                ),
                activations=AttrStrings.maybe(activations, name="activations"),
                clip=AttrFloat32.maybe(clip, name="clip"),
                direction=AttrString(direction, name="direction"),
                hidden_size=AttrInt64.maybe(hidden_size, name="hidden_size"),
                input_forget=AttrInt64(input_forget, name="input_forget"),
                layout=AttrInt64(layout, name="layout"),
            ),
            _LSTM.Inputs(
                X=unwrap_vars(X),
                W=unwrap_vars(W),
                R=unwrap_vars(R),
                B=unwrap_vars(B),
                sequence_lens=unwrap_vars(sequence_lens),
                initial_h=unwrap_vars(initial_h),
                initial_c=unwrap_vars(initial_c),
                P=unwrap_vars(P),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        ._unpack_to_any()
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::LpNormalization``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _LpNormalization(
            _LpNormalization.Attributes(
                axis=AttrInt64(axis, name="axis"),
                p=AttrInt64(p, name="p"),
            ),
            _LpNormalization.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def lp_pool(
    X: Var,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    dilations: Iterable[int] | None = None,
    kernel_shape: Iterable[int],
    p: int = 2,
    pads: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
) -> Var:
    r"""
    LpPool consumes an input tensor X and applies Lp pooling across the
    tensor according to kernel sizes, stride sizes, and pad lengths. Lp
    pooling consisting of computing the Lp norm on all values of a subset of
    the input tensor according to the kernel size and downsampling the data
    into the output tensor Y for further processing. The output spatial
    shape will be following:

    ::

       output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)

    or

    ::

       output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)

    if ceil_mode is enabled ``pad_shape[i]`` is the sum of pads along axis
    ``i``.

    ``auto_pad`` is a DEPRECATED attribute. If you are using them currently,
    the output spatial shape will be following:

    ::

       VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
       SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])

    And pad shape will be following if ``SAME_UPPER`` or ``SAME_LOWER``:

    ::

       pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]

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
    ceil_mode
        Attribute.
        Whether to use ceil or floor (default) to compute the output shape.
    dilations
        Attribute.
        dilation value along each spatial axis of the filter. If not present,
        the dilation defaults is 1 along each spatial axis.
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
    Signature: ``ai.onnx@22::LpPool``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _LpPool(
            _LpPool.Attributes(
                auto_pad=AttrString(auto_pad, name="auto_pad"),
                ceil_mode=AttrInt64(ceil_mode, name="ceil_mode"),
                dilations=AttrInt64s.maybe(dilations, name="dilations"),
                kernel_shape=AttrInt64s(kernel_shape, name="kernel_shape"),
                p=AttrInt64(p, name="p"),
                pads=AttrInt64s.maybe(pads, name="pads"),
                strides=AttrInt64s.maybe(strides, name="strides"),
            ),
            _LpPool.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


def max_pool(
    X: Var,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    dilations: Iterable[int] | None = None,
    kernel_shape: Iterable[int],
    pads: Iterable[int] | None = None,
    storage_order: int = 0,
    strides: Iterable[int] | None = None,
) -> tuple[Var, Var]:
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
    Signature: ``ai.onnx@22::MaxPool``.

    Type constraints:
     - I: `tensor(int64)`
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int8)`, `tensor(uint8)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _MaxPool(
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
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        ._unpack_to_any()
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::MaxRoiPool``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        rois=rois,
    )
    output_vars = (
        _MaxRoiPool(
            _MaxRoiPool.Attributes(
                pooled_shape=AttrInt64s(pooled_shape, name="pooled_shape"),
                spatial_scale=AttrFloat32(spatial_scale, name="spatial_scale"),
            ),
            _MaxRoiPool.Inputs(
                X=unwrap_vars(X),
                rois=unwrap_vars(rois),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


def max_unpool(
    X: Var,
    I: Var,
    output_shape: Var | None = None,
    *,
    kernel_shape: Iterable[int],
    pads: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
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
    Signature: ``ai.onnx@22::MaxUnpool``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(int64)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        I=I,
        output_shape=output_shape,
    )
    output_vars = (
        _MaxUnpool(
            _MaxUnpool.Attributes(
                kernel_shape=AttrInt64s(kernel_shape, name="kernel_shape"),
                pads=AttrInt64s.maybe(pads, name="pads"),
                strides=AttrInt64s.maybe(strides, name="strides"),
            ),
            _MaxUnpool.Inputs(
                X=unwrap_vars(X),
                I=unwrap_vars(I),
                output_shape=unwrap_vars(output_shape),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def mish(
    X: Var,
) -> Var:
    r"""
    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    Perform the linear unit element-wise on the input tensor X using
    formula:

    ::

       mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

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
    Signature: ``ai.onnx@22::Mish``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _Mish(
            _Mish.Attributes(),
            _Mish.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


def multinomial(
    input: Var,
    *,
    dtype: npt.DTypeLike = np.int32,
    sample_size: int = 1,
    seed: float | None = None,
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
    Signature: ``ai.onnx@22::Multinomial``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(int32)`, `tensor(int64)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Multinomial(
            _Multinomial.Attributes(
                dtype=AttrDtype(dtype, name="dtype"),
                sample_size=AttrInt64(sample_size, name="sample_size"),
                seed=AttrFloat32.maybe(seed, name="seed"),
            ),
            _Multinomial.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def negative_log_likelihood_loss(
    input: Var,
    target: Var,
    weight: Var | None = None,
    *,
    ignore_index: int | None = None,
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
    Signature: ``ai.onnx@22::NegativeLogLikelihoodLoss``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - Tind: `tensor(int32)`, `tensor(int64)`
    """
    input_prop_values = create_prop_dict(
        input=input,
        target=target,
        weight=weight,
    )
    output_vars = (
        _NegativeLogLikelihoodLoss(
            _NegativeLogLikelihoodLoss.Attributes(
                ignore_index=AttrInt64.maybe(ignore_index, name="ignore_index"),
                reduction=AttrString(reduction, name="reduction"),
            ),
            _NegativeLogLikelihoodLoss.Inputs(
                input=unwrap_vars(input),
                target=unwrap_vars(target),
                weight=unwrap_vars(weight),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .loss
    )
    return output_vars  # type: ignore


def rnn(
    X: Var,
    W: Var,
    R: Var,
    B: Var | None = None,
    sequence_lens: Var | None = None,
    initial_h: Var | None = None,
    *,
    activation_alpha: Iterable[float] | None = None,
    activation_beta: Iterable[float] | None = None,
    activations: Iterable[str] = ("Tanh", "Tanh"),
    clip: float | None = None,
    direction: str = "forward",
    hidden_size: int | None = None,
    layout: int = 0,
) -> tuple[Var, Var]:
    r"""
    Computes an one-layer simple RNN. This operator is usually supported via
    some custom implementation such as CuDNN.

    Notations:

    - ``X`` - input tensor
    - ``i`` - input gate
    - ``t`` - time step (t-1 means previous time step)
    - ``Wi`` - W parameter weight matrix for input gate
    - ``Ri`` - R recurrence weight matrix for input gate
    - ``Wbi`` - W parameter bias vector for input gate
    - ``Rbi`` - R parameter bias vector for input gate
    - ``WBi`` - W parameter weight matrix for backward input gate
    - ``RBi`` - R recurrence weight matrix for backward input gate
    - ``WBbi`` - WR bias vectors for backward input gate
    - ``RBbi`` - RR bias vectors for backward input gate
    - ``H`` - Hidden state
    - ``num_directions`` - 2 if direction == bidirectional else 1

    Activation functions:

    - Relu(x) - max(0, x)
    - Tanh(x) - (1 - e^{-2x})/(1 + e^{-2x})
    - Sigmoid(x) - 1/(1 + e^{-x})

    NOTE: Below are optional

    - Affine(x) - alpha*x + beta
    - LeakyRelu(x) - x if x >= 0 else alpha \* x
    - ThresholdedRelu(x) - x if x >= alpha else 0
    - ScaledTanh(x) - alpha\ *Tanh(beta*\ x)
    - HardSigmoid(x) - min(max(alpha*x + beta, 0), 1)
    - Elu(x) - x if x >= 0 else alpha*(e^x - 1)
    - Softsign(x) - x/(1 + \|x\|)
    - Softplus(x) - log(1 + e^x)

    Equations (Default: f=Tanh):

    - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi) This operator has
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
    Signature: ``ai.onnx@22::RNN``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T1: `tensor(int32)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        W=W,
        R=R,
        B=B,
        sequence_lens=sequence_lens,
        initial_h=initial_h,
    )
    output_vars = (
        _RNN(
            _RNN.Attributes(
                activation_alpha=AttrFloat32s.maybe(
                    activation_alpha, name="activation_alpha"
                ),
                activation_beta=AttrFloat32s.maybe(
                    activation_beta, name="activation_beta"
                ),
                activations=AttrStrings(activations, name="activations"),
                clip=AttrFloat32.maybe(clip, name="clip"),
                direction=AttrString(direction, name="direction"),
                hidden_size=AttrInt64.maybe(hidden_size, name="hidden_size"),
                layout=AttrInt64(layout, name="layout"),
            ),
            _RNN.Inputs(
                X=unwrap_vars(X),
                W=unwrap_vars(W),
                R=unwrap_vars(R),
                B=unwrap_vars(B),
                sequence_lens=unwrap_vars(sequence_lens),
                initial_h=unwrap_vars(initial_h),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        ._unpack_to_any()
    )
    return output_vars  # type: ignore


def random_normal(
    *,
    dtype: npt.DTypeLike = np.float32,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float | None = None,
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
    Signature: ``ai.onnx@22::RandomNormal``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict()
    output_vars = (
        _RandomNormal(
            _RandomNormal.Attributes(
                dtype=AttrDtype(dtype, name="dtype"),
                mean=AttrFloat32(mean, name="mean"),
                scale=AttrFloat32(scale, name="scale"),
                seed=AttrFloat32.maybe(seed, name="seed"),
                shape=AttrInt64s(shape, name="shape"),
            ),
            _RandomNormal.Inputs(),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def random_normal_like(
    input: Var,
    *,
    dtype: npt.DTypeLike | None = None,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float | None = None,
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
    Signature: ``ai.onnx@22::RandomNormalLike``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _RandomNormalLike(
            _RandomNormalLike.Attributes(
                dtype=AttrDtype.maybe(dtype, name="dtype"),
                mean=AttrFloat32(mean, name="mean"),
                scale=AttrFloat32(scale, name="scale"),
                seed=AttrFloat32.maybe(seed, name="seed"),
            ),
            _RandomNormalLike.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def random_uniform(
    *,
    dtype: npt.DTypeLike = np.float32,
    high: float = 1.0,
    low: float = 0.0,
    seed: float | None = None,
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
    Signature: ``ai.onnx@22::RandomUniform``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict()
    output_vars = (
        _RandomUniform(
            _RandomUniform.Attributes(
                dtype=AttrDtype(dtype, name="dtype"),
                high=AttrFloat32(high, name="high"),
                low=AttrFloat32(low, name="low"),
                seed=AttrFloat32.maybe(seed, name="seed"),
                shape=AttrInt64s(shape, name="shape"),
            ),
            _RandomUniform.Inputs(),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def random_uniform_like(
    input: Var,
    *,
    dtype: npt.DTypeLike | None = None,
    high: float = 1.0,
    low: float = 0.0,
    seed: float | None = None,
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
    Signature: ``ai.onnx@22::RandomUniformLike``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _RandomUniformLike(
            _RandomUniformLike.Attributes(
                dtype=AttrDtype.maybe(dtype, name="dtype"),
                high=AttrFloat32(high, name="high"),
                low=AttrFloat32(low, name="low"),
                seed=AttrFloat32.maybe(seed, name="seed"),
            ),
            _RandomUniformLike.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::RoiAlign``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
     - T2: `tensor(int64)`
    """
    input_prop_values = create_prop_dict(
        X=X,
        rois=rois,
        batch_indices=batch_indices,
    )
    output_vars = (
        _RoiAlign(
            _RoiAlign.Attributes(
                coordinate_transformation_mode=AttrString(
                    coordinate_transformation_mode,
                    name="coordinate_transformation_mode",
                ),
                mode=AttrString(mode, name="mode"),
                output_height=AttrInt64(output_height, name="output_height"),
                output_width=AttrInt64(output_width, name="output_width"),
                sampling_ratio=AttrInt64(sampling_ratio, name="sampling_ratio"),
                spatial_scale=AttrFloat32(spatial_scale, name="spatial_scale"),
            ),
            _RoiAlign.Inputs(
                X=unwrap_vars(X),
                rois=unwrap_vars(rois),
                batch_indices=unwrap_vars(batch_indices),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Round``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _Round(
            _Round.Attributes(),
            _Round.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Selu``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _Selu(
            _Selu.Attributes(
                alpha=AttrFloat32(alpha, name="alpha"),
                gamma=AttrFloat32(gamma, name="gamma"),
            ),
            _Selu.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Sin``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Sin(
            _Sin.Attributes(),
            _Sin.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Sinh``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Sinh(
            _Sinh.Attributes(),
            _Sinh.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
        Input tensor

    Returns
    =======
    Y : Var
        Type T.
        Output tensor

    Notes
    =====
    Signature: ``ai.onnx@22::Softplus``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _Softplus(
            _Softplus.Attributes(),
            _Softplus.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Softsign``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Softsign(
            _Softsign.Attributes(),
            _Softsign.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::Tan``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _Tan(
            _Tan.Attributes(),
            _Tan.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


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
    Signature: ``ai.onnx@22::ThresholdedRelu``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        X=X,
    )
    output_vars = (
        _ThresholdedRelu(
            _ThresholdedRelu.Attributes(
                alpha=AttrFloat32(alpha, name="alpha"),
            ),
            _ThresholdedRelu.Inputs(
                X=unwrap_vars(X),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .Y
    )
    return output_vars  # type: ignore


def const(value: npt.ArrayLike, dtype: npt.DTypeLike | None = None) -> Var:
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
    "AffineGrid": _AffineGrid,
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
    "BitwiseAnd": _BitwiseAnd,
    "BitwiseNot": _BitwiseNot,
    "BitwiseOr": _BitwiseOr,
    "BitwiseXor": _BitwiseXor,
    "BlackmanWindow": _BlackmanWindow,
    "Cast": _Cast,
    "CastLike": _CastLike,
    "Ceil": _Ceil,
    "Celu": _Celu,
    "CenterCropPad": _CenterCropPad,
    "Clip": _Clip,
    "Col2Im": _Col2Im,
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
    "DeformConv": _DeformConv,
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
    "Gelu": _Gelu,
    "Gemm": _Gemm,
    "GlobalAveragePool": _GlobalAveragePool,
    "GlobalLpPool": _GlobalLpPool,
    "GlobalMaxPool": _GlobalMaxPool,
    "Greater": _Greater,
    "GreaterOrEqual": _GreaterOrEqual,
    "GridSample": _GridSample,
    "GroupNormalization": _GroupNormalization,
    "HammingWindow": _HammingWindow,
    "HannWindow": _HannWindow,
    "HardSigmoid": _HardSigmoid,
    "HardSwish": _HardSwish,
    "Hardmax": _Hardmax,
    "Identity": _Identity,
    "If": _If,
    "ImageDecoder": _ImageDecoder,
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
    "Mish": _Mish,
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
    "RegexFullMatch": _RegexFullMatch,
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
    "StringConcat": _StringConcat,
    "StringNormalizer": _StringNormalizer,
    "StringSplit": _StringSplit,
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
    "AffineGrid": affine_grid,
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
    "BitwiseAnd": bitwise_and,
    "BitwiseNot": bitwise_not,
    "BitwiseOr": bitwise_or,
    "BitwiseXor": bitwise_xor,
    "BlackmanWindow": blackman_window,
    "Cast": cast,
    "CastLike": cast_like,
    "Ceil": ceil,
    "Celu": celu,
    "CenterCropPad": center_crop_pad,
    "Clip": clip,
    "Col2Im": col2_im,
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
    "DeformConv": deform_conv,
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
    "Gelu": gelu,
    "Gemm": gemm,
    "GlobalAveragePool": global_average_pool,
    "GlobalLpPool": global_lp_pool,
    "GlobalMaxPool": global_max_pool,
    "Greater": greater,
    "GreaterOrEqual": greater_or_equal,
    "GridSample": grid_sample,
    "GroupNormalization": group_normalization,
    "HammingWindow": hamming_window,
    "HannWindow": hann_window,
    "HardSigmoid": hard_sigmoid,
    "HardSwish": hard_swish,
    "Hardmax": hardmax,
    "Identity": identity,
    "If": if_,
    "ImageDecoder": image_decoder,
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
    "Mish": mish,
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
    "RegexFullMatch": regex_full_match,
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
    "StringConcat": string_concat,
    "StringNormalizer": string_normalizer,
    "StringSplit": string_split,
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

__all__ = [fun.__name__ for fun in _CONSTRUCTORS.values()] + ["const"]
