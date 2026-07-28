# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: E741 -- Allow ambiguous variable name
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spox._attributes import (
    AttrInt64,
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
from spox.opset.ai.onnx.v25 import (
    _DFT,
    _GRU,
    _LRN,
    _LSTM,
    _RNN,
    _STFT,
    _Abs,
    _Acos,
    _Acosh,
    _Add,
    _AffineGrid,
    _And,
    _ArgMax,
    _ArgMin,
    _Asin,
    _Asinh,
    _Atan,
    _Atanh,
    _Attention,
    _AveragePool,
    _BatchNormalization,
    _Bernoulli,
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
    _Conv,
    _ConvInteger,
    _ConvTranspose,
    _Cos,
    _Cosh,
    _CumSum,
    _DeformConv,
    _DepthToSpace,
    _DequantizeLinear,
    _Det,
    _Div,
    _Dropout,
    _DynamicQuantizeLinear,
    _Einsum,
    _Elu,
    _Equal,
    _Erf,
    _Exp,
    _Expand,
    _EyeLike,
    _Flatten,
    _Floor,
    _Gather,
    _GatherElements,
    _GatherND,
    _Gelu,
    _Gemm,
    _GlobalAveragePool,
    _GlobalLpPool,
    _GlobalMaxPool,
    _Greater,
    _GreaterOrEqual,
    _GridSample,
    _GroupNormalization,
    _HammingWindow,
    _HannWindow,
    _Hardmax,
    _HardSigmoid,
    _HardSwish,
    _Identity,
    _If,
    _ImageDecoder,
    _InstanceNormalization,
    _IsInf,
    _IsNaN,
    _LayerNormalization,
    _LeakyRelu,
    _Less,
    _LessOrEqual,
    _Log,
    _LogSoftmax,
    _Loop,
    _LpNormalization,
    _LpPool,
    _MatMul,
    _MatMulInteger,
    _Max,
    _MaxPool,
    _MaxRoiPool,
    _MaxUnpool,
    _Mean,
    _MeanVarianceNormalization,
    _MelWeightMatrix,
    _Min,
    _Mish,
    _Mod,
    _Mul,
    _Multinomial,
    _Neg,
    _NegativeLogLikelihoodLoss,
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
    _RandomNormal,
    _RandomNormalLike,
    _RandomUniform,
    _RandomUniformLike,
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
    _RMSNormalization,
    _RoiAlign,
    _RotaryEmbedding,
    _Round,
    _Scan,
    _ScatterElements,
    _ScatterND,
    _Selu,
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
    _Sin,
    _Sinh,
    _Size,
    _Slice,
    _Softmax,
    _SoftmaxCrossEntropyLoss,
    _Softplus,
    _Softsign,
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
    _Swish,
    _Tan,
    _Tanh,
    _TensorScatter,
    _TfIdfVectorizer,
    _ThresholdedRelu,
    _Tile,
    _TopK,
    _Transpose,
    _Trilu,
    _Unique,
    _Unsqueeze,
    _Where,
    _Xor,
    abs,
    acos,
    acosh,
    add,
    affine_grid,
    and_,
    arg_max,
    arg_min,
    asin,
    asinh,
    atan,
    atanh,
    attention,
    average_pool,
    batch_normalization,
    bernoulli,
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
    conv,
    conv_integer,
    conv_transpose,
    cos,
    cosh,
    cumsum,
    deform_conv,
    depth_to_space,
    dequantize_linear,
    det,
    dft,
    div,
    dropout,
    dynamic_quantize_linear,
    einsum,
    elu,
    equal,
    erf,
    exp,
    expand,
    eye_like,
    flatten,
    floor,
    gather,
    gather_elements,
    gather_nd,
    gelu,
    gemm,
    global_average_pool,
    global_lp_pool,
    global_max_pool,
    greater,
    greater_or_equal,
    grid_sample,
    group_normalization,
    gru,
    hamming_window,
    hann_window,
    hard_sigmoid,
    hard_swish,
    hardmax,
    identity,
    if_,
    image_decoder,
    instance_normalization,
    isinf,
    isnan,
    layer_normalization,
    leaky_relu,
    less,
    less_or_equal,
    log,
    log_softmax,
    loop,
    lp_normalization,
    lp_pool,
    lrn,
    lstm,
    matmul,
    matmul_integer,
    max,
    max_pool,
    max_roi_pool,
    max_unpool,
    mean,
    mean_variance_normalization,
    mel_weight_matrix,
    min,
    mish,
    mod,
    mul,
    multinomial,
    neg,
    negative_log_likelihood_loss,
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
    random_normal,
    random_normal_like,
    random_uniform,
    random_uniform_like,
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
    rmsnormalization,
    rnn,
    roi_align,
    rotary_embedding,
    round,
    scan,
    scatter_elements,
    scatter_nd,
    selu,
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
    sin,
    sinh,
    size,
    slice,
    softmax,
    softmax_cross_entropy_loss,
    softplus,
    softsign,
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
    swish,
    tan,
    tanh,
    tensor_scatter,
    tf_idf_vectorizer,
    thresholded_relu,
    tile,
    top_k,
    transpose,
    trilu,
    unique,
    unsqueeze,
    where,
    xor,
)


class _BitCast(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        to: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("BitCast", "", 26)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _CumProd(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        exclusive: AttrInt64
        reverse: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        x: _VarInfo
        axis: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        y: _VarInfo

    op_type = OpType("CumProd", "", 26)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def bit_cast(
    input: Var,
    *,
    to: int,
) -> Var:
    r"""
    Reinterprets the binary representation of a tensor as a different data
    type, specified by the 'to' attribute. Unlike Cast, BitCast preserves
    the exact bit pattern without any value conversion.

    The target data type must have the same bit-width as the input data
    type. The output tensor has the same shape as the input tensor. All
    types except string are supported. Implementations must treat the
    underlying bytes as little endian.

    Parameters
    ==========
    input
        Type T1.
        Input tensor to be bitcast.
    to
        Attribute.
        The data type to which the input tensor is bitwise reinterpreted. Must
        be one of the non-string types from DataType enum in TensorProto. The
        target type must have the same bit-width as the input type.

    Returns
    =======
    output : Var
        Type T2.
        Output tensor with the same shape as the input.

    Notes
    =====
    Signature: ``ai.onnx@26::BitCast``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)`, `tensor(int16)`, `tensor(int2)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint2)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float4e2m1)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(float8e8m0)`, `tensor(int16)`, `tensor(int2)`, `tensor(int32)`, `tensor(int4)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint2)`, `tensor(uint32)`, `tensor(uint4)`, `tensor(uint64)`, `tensor(uint8)`
    """
    input_prop_values = create_prop_dict(
        input=input,
    )
    output_vars = (
        _BitCast(
            _BitCast.Attributes(
                to=AttrInt64(to, name="to"),
            ),
            _BitCast.Inputs(
                input=unwrap_vars(input),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
    )
    return output_vars  # type: ignore


def cum_prod(
    x: Var,
    axis: Var,
    *,
    exclusive: int = 0,
    reverse: int = 0,
) -> Var:
    r"""
    Performs cumulative product of the input elements along the given axis.
    By default, it will do the product inclusively meaning the first element
    is copied as is. Through an ``exclusive`` attribute, this behavior can
    change to exclude the first element. It can also perform product in the
    opposite direction of the axis. For that, set ``reverse`` attribute to
    1.

    Example:

    ::

       input_x = [1, 2, 3]
       axis=0
       output = [1, 2, 6]
       exclusive=1
       output = [1, 1, 2]
       exclusive=0
       reverse=1
       output = [6, 6, 3]
       exclusive=1
       reverse=1
       output = [6, 3, 1]

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
        If set to 1 will return exclusive product in which the top element is
        not included. In other terms, if set to 1, the j-th output element would
        be the product of the first (j-1) elements. Otherwise, it would be the
        product of the first j elements.
    reverse
        Attribute.
        If set to 1 will perform the products in reverse direction.

    Returns
    =======
    y : Var
        Type T.
        Output tensor of the same type as 'x' with cumulative products of the
        x's elements

    Notes
    =====
    Signature: ``ai.onnx@26::CumProd``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
     - T2: `tensor(int32)`, `tensor(int64)`
    """
    input_prop_values = create_prop_dict(
        x=x,
        axis=axis,
    )
    output_vars = (
        _CumProd(
            _CumProd.Attributes(
                exclusive=AttrInt64(exclusive, name="exclusive"),
                reverse=AttrInt64(reverse, name="reverse"),
            ),
            _CumProd.Inputs(
                x=unwrap_vars(x),
                axis=unwrap_vars(axis),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .y
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
    "Attention": _Attention,
    "AveragePool": _AveragePool,
    "BatchNormalization": _BatchNormalization,
    "Bernoulli": _Bernoulli,
    "BitCast": _BitCast,
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
    "CumProd": _CumProd,
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
    "RMSNormalization": _RMSNormalization,
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
    "RotaryEmbedding": _RotaryEmbedding,
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
    "Swish": _Swish,
    "Tan": _Tan,
    "Tanh": _Tanh,
    "TensorScatter": _TensorScatter,
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
    "Attention": attention,
    "AveragePool": average_pool,
    "BatchNormalization": batch_normalization,
    "Bernoulli": bernoulli,
    "BitCast": bit_cast,
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
    "CumProd": cum_prod,
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
    "RMSNormalization": rmsnormalization,
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
    "RotaryEmbedding": rotary_embedding,
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
    "Swish": swish,
    "Tan": tan,
    "Tanh": tanh,
    "TensorScatter": tensor_scatter,
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
