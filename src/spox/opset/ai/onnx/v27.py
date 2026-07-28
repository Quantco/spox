# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: E741 -- Allow ambiguous variable name
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spox._attributes import (
    AttrFloat32,
    AttrInt64,
    AttrString,
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
from spox.opset.ai.onnx.v26 import (
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
    _BitCast,
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
    _CumProd,
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
    bit_cast,
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
    cum_prod,
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


class _CausalConvWithState(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        activation: AttrString

    @dataclass
    class Inputs(BaseInputs):
        input: _VarInfo
        weight: _VarInfo
        bias: _VarInfo | None
        past_state: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo
        present_state: _VarInfo

    op_type = OpType("CausalConvWithState", "", 27)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LinearAttention(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        chunk_size: AttrInt64
        kv_num_heads: AttrInt64
        q_num_heads: AttrInt64
        scale: AttrFloat32
        update_rule: AttrString

    @dataclass
    class Inputs(BaseInputs):
        query: _VarInfo
        key: _VarInfo
        value: _VarInfo
        past_state: _VarInfo | None
        decay: _VarInfo | None
        beta: _VarInfo | None

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo
        present_state: _VarInfo

    op_type = OpType("LinearAttention", "", 27)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Range(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        stash_type: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        start: _VarInfo
        limit: _VarInfo
        delta: _VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: _VarInfo

    op_type = OpType("Range", "", 27)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def causal_conv_with_state(
    input: Var,
    weight: Var,
    bias: Var | None = None,
    past_state: Var | None = None,
    *,
    activation: str = "none",
) -> tuple[Var, Var]:
    r"""
    Stateful causal 1D depthwise convolution.

    Used by Gated DeltaNet (Qwen3.5) and Mamba (Jamba, FalconMamba) as a
    preprocessing step. Replaces the 3-op pattern (Concat + Conv + Slice)
    with a single fused operation.

    The convolution is causal (looks only at current and past positions) and
    depthwise (each channel is convolved independently with its own kernel).

    The input, weight, past_state, output, and present_state tensors are
    rank-3 with shape (batch_size, channels, length). The optional bias
    input is rank-1 with shape (channels). For higher-dimensional data, use
    Reshape nodes before and after this operator to pack extra dimensions
    into the batch or channel axis.

    Weight layout: (channels, 1, k) for depthwise convolution. The carry
    state stores the last (k-1) positions for incremental decode.

    The optional activation attribute supports fused SiLU/Swish activation.

    Parameters
    ==========
    input
        Type T.
        Input tensor with shape (batch_size, channels, length). Channels-first
        layout.
    weight
        Type T.
        Depthwise convolution kernel with shape (channels, 1, k) where k is the
        kernel size. The middle dim of size 1 follows the ONNX ``Conv`` weight
        layout ``(M, C/group, k1, ..., kn)``: since this op is always depthwise,
        ``group = channels``, so ``C/group = 1``. Keeping this layout makes the
        weight tensor a drop-in for a depthwise ``Conv(group=channels)`` weight,
        so ``Conv`` <-> ``CausalConvWithState`` rewrites require no reshape.
    bias
        Type T.
        Optional per-channel bias with shape (channels).
    past_state
        Type T.
        Carry state from previous step with shape (batch_size, channels, k - 1).
        If not provided, padding is zero.
    activation
        Attribute.
        Fused activation function. One of: 'silu', 'swish', 'none'. Default is
        'none'.

    Returns
    =======
    output : Var
        Type T.
        Convolution output with same shape as input.
    present_state : Var
        Type T.
        Updated carry state with shape (batch_size, channels, k - 1). Contains
        the last (k - 1) values of the effective padded/concatenated sequence
        along the causal axis, including any values from past_state or
        zero-padding when the current input is shorter than k - 1.

    Notes
    =====
    Signature: ``ai.onnx@27::CausalConvWithState``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        input=input,
        weight=weight,
        bias=bias,
        past_state=past_state,
    )
    output_vars = (
        _CausalConvWithState(
            _CausalConvWithState.Attributes(
                activation=AttrString(activation, name="activation"),
            ),
            _CausalConvWithState.Inputs(
                input=unwrap_vars(input),
                weight=unwrap_vars(weight),
                bias=unwrap_vars(bias),
                past_state=unwrap_vars(past_state),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        ._unpack_to_any()
    )
    return output_vars  # type: ignore


def linear_attention(
    query: Var,
    key: Var,
    value: Var,
    past_state: Var | None = None,
    decay: Var | None = None,
    beta: Var | None = None,
    *,
    chunk_size: int = 64,
    kv_num_heads: int,
    q_num_heads: int,
    scale: float = 0.0,
    update_rule: str = "gated_delta",
) -> tuple[Var, Var]:
    r"""
    Unified linear attention operator for autoregressive decoding (T=1) and
    prefill (T>1).

    The query, key, value, and (where applicable) decay/beta inputs use 3D
    packed format [B, T, H*D], where heads are flattened into the last
    dimension; q_num_heads and kv_num_heads are always required and are used
    to unpack to 4D internally for computation. The optional past_state and
    present_state are 4D with shape (B, H_kv, d_k, d_v).

    Group-query attention (GQA) is supported: q_num_heads must be a positive
    multiple of kv_num_heads. When q_num_heads == kv_num_heads this reduces
    to multi-headed linear attention; when q_num_heads > kv_num_heads each
    KV head (and its recurrent state) is shared by
    ``q_num_heads / kv_num_heads`` query heads (multi-query attention is the
    special case kv_num_heads == 1).

    The update_rule attribute selects the recurrence type:

    - "linear": S_t = S\_{t-1} + k_t ⊗ v_t; o_t = scale \* q_t^T S_t
    - "gated": S_t = exp(g_t) \* S\_{t-1} + k_t ⊗ v_t; o_t = scale \* q_t^T
      S_t
    - "delta": S_t = S\_{t-1} + β_t \* k_t ⊗ (v_t - S\_{t-1}^T k_t); o_t =
      scale \* q_t^T S_t
    - "gated_delta": S_t = exp(g_t) \* S\_{t-1} + β_t \* k_t ⊗ (v_t -
      exp(g_t) \* S\_{t-1}^T k_t); o_t = scale \* q_t^T S_t

    where g_t is the decay (in log-space), β_t is the update rate, and ⊗
    denotes outer product.

    Semantics: Equivalent to running the recurrent update sequentially for
    each token, but may be implemented using chunk-parallel algorithms for
    GPU efficiency.

    Parameters
    ==========
    query
        Type T.
        Query vectors with 3D packed shape (B, T, H_q \* d_k). Heads are packed
        into the last dimension.
    key
        Type T.
        Key vectors with 3D packed shape (B, T, H_kv \* d_k). Should be
        L2-normalized for delta/gated_delta modes.
    value
        Type T.
        Value vectors with 3D packed shape (B, T, H_kv \* d_v).
    past_state
        Type S.
        Recurrent state from previous step with shape (B, H_kv, d_k, d_v).
        Always 4D. If not provided, defaults to zeros.
    decay
        Type T.
        Exponential decay gate in log-space. 3D packed shape: (B, T, H_kv \*
        d_k) for per-key-dimension decay (GLA/RWKV-6), or (B, T, H_kv) for
        per-head scalar decay (DeltaNet/RetNet). Required for 'gated' and
        'gated_delta' modes.
    beta
        Type T.
        Update rate (sigmoid output). 3D packed shape: (B, T, H_kv) or (B, T,
        1). Required for 'delta' and 'gated_delta' modes.
    chunk_size
        Attribute.
        Chunk size for the chunk-parallel WY decomposition during prefill (T>1).
        Tuning hint; does not affect output correctness.
    kv_num_heads
        Attribute.
        Number of key/value heads. Always required.
    q_num_heads
        Attribute.
        Number of query heads. Always required.
    scale
        Attribute.
        Output scaling factor. When 0.0 (default), derives d_k = query.shape[-1]
        / q_num_heads and uses 1/sqrt(d_k). Set explicitly to override.
    update_rule
        Attribute.
        The update rule for the linear attention recurrence. One of: 'linear',
        'gated', 'delta', 'gated_delta'. Default is 'gated_delta'.

    Returns
    =======
    output : Var
        Type T.
        Attention output with 3D packed shape (B, T, H_q \* d_v).
    present_state : Var
        Type S.
        Updated recurrent state with shape (B, H_kv, d_k, d_v). Always 4D.

    Notes
    =====
    Signature: ``ai.onnx@27::LinearAttention``.

    Type constraints:
     - S: `tensor(bfloat16)`, `tensor(float)`, `tensor(float16)`
     - T: `tensor(bfloat16)`, `tensor(float)`, `tensor(float16)`
    """
    input_prop_values = create_prop_dict(
        query=query,
        key=key,
        value=value,
        past_state=past_state,
        decay=decay,
        beta=beta,
    )
    output_vars = (
        _LinearAttention(
            _LinearAttention.Attributes(
                chunk_size=AttrInt64(chunk_size, name="chunk_size"),
                kv_num_heads=AttrInt64(kv_num_heads, name="kv_num_heads"),
                q_num_heads=AttrInt64(q_num_heads, name="q_num_heads"),
                scale=AttrFloat32(scale, name="scale"),
                update_rule=AttrString(update_rule, name="update_rule"),
            ),
            _LinearAttention.Inputs(
                query=unwrap_vars(query),
                key=unwrap_vars(key),
                value=unwrap_vars(value),
                past_state=unwrap_vars(past_state),
                decay=unwrap_vars(decay),
                beta=unwrap_vars(beta),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        ._unpack_to_any()
    )
    return output_vars  # type: ignore


def range(
    start: Var,
    limit: Var,
    delta: Var,
    *,
    stash_type: int = 1,
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

    Example 1:

    ::

       Inputs: start = 3, limit = 9, delta = 3
       Output: [3, 6]

    Example 2:

    ::

       Inputs: start = 10, limit = 4, delta = -2
       Output: [10, 8, 6]

    For ``float16`` and ``bfloat16`` inputs, the ``stash_type`` attribute
    controls the precision used for intermediate accumulation. Setting
    ``stash_type`` to ``1`` (float) causes ``start``, ``limit``, and
    ``delta`` to be cast to 32-bit float before the loop, with the output
    cast back to the original type. This avoids precision loss for large
    ranges where successive additions in float16 or bfloat16 would otherwise
    be inexact (e.g. ``x + 1 == x`` for large ``x``).

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
    stash_type
        Attribute.
        The data type used for intermediate computation when T is float16 or
        bfloat16. Defaults to 1 (float). Has no effect for other types.

    Returns
    =======
    output : Var
        Type T.
        A 1-D tensor with same type as the inputs containing generated range of
        values.

    Notes
    =====
    Signature: ``ai.onnx@27::Range``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`
    """
    input_prop_values = create_prop_dict(
        start=start,
        limit=limit,
        delta=delta,
    )
    output_vars = (
        _Range(
            _Range.Attributes(
                stash_type=AttrInt64(stash_type, name="stash_type"),
            ),
            _Range.Inputs(
                start=unwrap_vars(start),
                limit=unwrap_vars(limit),
                delta=unwrap_vars(delta),
            ),
        )
        .get_output_vars(input_prop_values=input_prop_values)
        .output
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
    "CausalConvWithState": _CausalConvWithState,
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
    "LinearAttention": _LinearAttention,
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
    "CausalConvWithState": causal_conv_with_state,
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
    "LinearAttention": linear_attention,
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
