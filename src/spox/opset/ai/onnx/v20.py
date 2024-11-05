# ruff: noqa: E741 -- Allow ambiguous variable name
import typing
import warnings
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)
from typing import cast as typing_cast

import numpy as np
import numpy.typing as npt

from spox._var import Var, VarInfo, result_type, unwrap_vars, get_value
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
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
from spox._graph import Graph, subgraph
from spox._internal_op import intro
from spox._node import OpType
from spox._standard import InferenceError, StandardNode
from spox._type_system import Tensor, Type, Sequence as SpoxSequence
from spox._value_prop import PropValueType


from spox.opset.ai.onnx.v19 import _Abs, abs
from spox.opset.ai.onnx.v19 import _Acos, acos
from spox.opset.ai.onnx.v19 import _Acosh, acosh
from spox.opset.ai.onnx.v19 import _Add, add
from spox.opset.ai.onnx.v19 import _And, and_
from spox.opset.ai.onnx.v19 import _ArgMax, arg_max
from spox.opset.ai.onnx.v19 import _ArgMin, arg_min
from spox.opset.ai.onnx.v19 import _Asin, asin
from spox.opset.ai.onnx.v19 import _Asinh, asinh
from spox.opset.ai.onnx.v19 import _Atan, atan
from spox.opset.ai.onnx.v19 import _Atanh, atanh
from spox.opset.ai.onnx.v19 import _AveragePool, average_pool
from spox.opset.ai.onnx.v19 import _BatchNormalization, batch_normalization
from spox.opset.ai.onnx.v19 import _Bernoulli, bernoulli
from spox.opset.ai.onnx.v19 import _BitShift, bit_shift
from spox.opset.ai.onnx.v19 import _BitwiseAnd, bitwise_and
from spox.opset.ai.onnx.v19 import _BitwiseNot, bitwise_not
from spox.opset.ai.onnx.v19 import _BitwiseOr, bitwise_or
from spox.opset.ai.onnx.v19 import _BitwiseXor, bitwise_xor
from spox.opset.ai.onnx.v19 import _BlackmanWindow, blackman_window
from spox.opset.ai.onnx.v19 import _Cast, cast
from spox.opset.ai.onnx.v19 import _CastLike, cast_like
from spox.opset.ai.onnx.v19 import _Ceil, ceil
from spox.opset.ai.onnx.v19 import _Celu, celu
from spox.opset.ai.onnx.v19 import _CenterCropPad, center_crop_pad
from spox.opset.ai.onnx.v19 import _Clip, clip
from spox.opset.ai.onnx.v19 import _Col2Im, col2_im
from spox.opset.ai.onnx.v19 import _Compress, compress
from spox.opset.ai.onnx.v19 import _Concat, concat
from spox.opset.ai.onnx.v19 import _ConcatFromSequence, concat_from_sequence
from spox.opset.ai.onnx.v19 import _Constant, constant
from spox.opset.ai.onnx.v19 import _Conv, conv
from spox.opset.ai.onnx.v19 import _ConvInteger, conv_integer
from spox.opset.ai.onnx.v19 import _ConvTranspose, conv_transpose
from spox.opset.ai.onnx.v19 import _Cos, cos
from spox.opset.ai.onnx.v19 import _Cosh, cosh
from spox.opset.ai.onnx.v19 import _CumSum, cumsum
from spox.opset.ai.onnx.v19 import _DeformConv, deform_conv
from spox.opset.ai.onnx.v19 import _DepthToSpace, depth_to_space
from spox.opset.ai.onnx.v19 import _DequantizeLinear, dequantize_linear
from spox.opset.ai.onnx.v19 import _Det, det
from spox.opset.ai.onnx.v19 import _Div, div
from spox.opset.ai.onnx.v19 import _Dropout, dropout
from spox.opset.ai.onnx.v19 import _DynamicQuantizeLinear, dynamic_quantize_linear
from spox.opset.ai.onnx.v19 import _Einsum, einsum
from spox.opset.ai.onnx.v19 import _Elu, elu
from spox.opset.ai.onnx.v19 import _Equal, equal
from spox.opset.ai.onnx.v19 import _Erf, erf
from spox.opset.ai.onnx.v19 import _Exp, exp
from spox.opset.ai.onnx.v19 import _Expand, expand
from spox.opset.ai.onnx.v19 import _EyeLike, eye_like
from spox.opset.ai.onnx.v19 import _Flatten, flatten
from spox.opset.ai.onnx.v19 import _Floor, floor
from spox.opset.ai.onnx.v19 import _GRU, gru
from spox.opset.ai.onnx.v19 import _Gather, gather
from spox.opset.ai.onnx.v19 import _GatherElements, gather_elements
from spox.opset.ai.onnx.v19 import _GatherND, gather_nd
from spox.opset.ai.onnx.v19 import _Gemm, gemm
from spox.opset.ai.onnx.v19 import _GlobalAveragePool, global_average_pool
from spox.opset.ai.onnx.v19 import _GlobalLpPool, global_lp_pool
from spox.opset.ai.onnx.v19 import _GlobalMaxPool, global_max_pool
from spox.opset.ai.onnx.v19 import _Greater, greater
from spox.opset.ai.onnx.v19 import _GreaterOrEqual, greater_or_equal
from spox.opset.ai.onnx.v19 import _GroupNormalization, group_normalization
from spox.opset.ai.onnx.v19 import _HammingWindow, hamming_window
from spox.opset.ai.onnx.v19 import _HannWindow, hann_window
from spox.opset.ai.onnx.v19 import _HardSigmoid, hard_sigmoid
from spox.opset.ai.onnx.v19 import _HardSwish, hard_swish
from spox.opset.ai.onnx.v19 import _Hardmax, hardmax
from spox.opset.ai.onnx.v19 import _Identity, identity
from spox.opset.ai.onnx.v19 import _If, if_
from spox.opset.ai.onnx.v19 import _InstanceNormalization, instance_normalization
from spox.opset.ai.onnx.v19 import _LRN, lrn
from spox.opset.ai.onnx.v19 import _LSTM, lstm
from spox.opset.ai.onnx.v19 import _LayerNormalization, layer_normalization
from spox.opset.ai.onnx.v19 import _LeakyRelu, leaky_relu
from spox.opset.ai.onnx.v19 import _Less, less
from spox.opset.ai.onnx.v19 import _LessOrEqual, less_or_equal
from spox.opset.ai.onnx.v19 import _Log, log
from spox.opset.ai.onnx.v19 import _LogSoftmax, log_softmax
from spox.opset.ai.onnx.v19 import _Loop, loop
from spox.opset.ai.onnx.v19 import _LpNormalization, lp_normalization
from spox.opset.ai.onnx.v19 import _LpPool, lp_pool
from spox.opset.ai.onnx.v19 import _MatMul, matmul
from spox.opset.ai.onnx.v19 import _MatMulInteger, matmul_integer
from spox.opset.ai.onnx.v19 import _Max, max
from spox.opset.ai.onnx.v19 import _MaxPool, max_pool
from spox.opset.ai.onnx.v19 import _MaxRoiPool, max_roi_pool
from spox.opset.ai.onnx.v19 import _MaxUnpool, max_unpool
from spox.opset.ai.onnx.v19 import _Mean, mean
from spox.opset.ai.onnx.v19 import _MeanVarianceNormalization, mean_variance_normalization
from spox.opset.ai.onnx.v19 import _MelWeightMatrix, mel_weight_matrix
from spox.opset.ai.onnx.v19 import _Min, min
from spox.opset.ai.onnx.v19 import _Mish, mish
from spox.opset.ai.onnx.v19 import _Mod, mod
from spox.opset.ai.onnx.v19 import _Mul, mul
from spox.opset.ai.onnx.v19 import _Multinomial, multinomial
from spox.opset.ai.onnx.v19 import _Neg, neg
from spox.opset.ai.onnx.v19 import _NegativeLogLikelihoodLoss, negative_log_likelihood_loss
from spox.opset.ai.onnx.v19 import _NonMaxSuppression, non_max_suppression
from spox.opset.ai.onnx.v19 import _NonZero, non_zero
from spox.opset.ai.onnx.v19 import _Not, not_
from spox.opset.ai.onnx.v19 import _OneHot, one_hot
from spox.opset.ai.onnx.v19 import _Optional, optional
from spox.opset.ai.onnx.v19 import _OptionalGetElement, optional_get_element
from spox.opset.ai.onnx.v19 import _OptionalHasElement, optional_has_element
from spox.opset.ai.onnx.v19 import _Or, or_
from spox.opset.ai.onnx.v19 import _PRelu, prelu
from spox.opset.ai.onnx.v19 import _Pad, pad
from spox.opset.ai.onnx.v19 import _Pow, pow
from spox.opset.ai.onnx.v19 import _QLinearConv, qlinear_conv
from spox.opset.ai.onnx.v19 import _QLinearMatMul, qlinear_matmul
from spox.opset.ai.onnx.v19 import _QuantizeLinear, quantize_linear
from spox.opset.ai.onnx.v19 import _RNN, rnn
from spox.opset.ai.onnx.v19 import _RandomNormal, random_normal
from spox.opset.ai.onnx.v19 import _RandomNormalLike, random_normal_like
from spox.opset.ai.onnx.v19 import _RandomUniform, random_uniform
from spox.opset.ai.onnx.v19 import _RandomUniformLike, random_uniform_like
from spox.opset.ai.onnx.v19 import _Range, range
from spox.opset.ai.onnx.v19 import _Reciprocal, reciprocal
from spox.opset.ai.onnx.v19 import _ReduceL1, reduce_l1
from spox.opset.ai.onnx.v19 import _ReduceL2, reduce_l2
from spox.opset.ai.onnx.v19 import _ReduceLogSum, reduce_log_sum
from spox.opset.ai.onnx.v19 import _ReduceLogSumExp, reduce_log_sum_exp
from spox.opset.ai.onnx.v19 import _ReduceMean, reduce_mean
from spox.opset.ai.onnx.v19 import _ReduceProd, reduce_prod
from spox.opset.ai.onnx.v19 import _ReduceSum, reduce_sum
from spox.opset.ai.onnx.v19 import _ReduceSumSquare, reduce_sum_square
from spox.opset.ai.onnx.v19 import _Relu, relu
from spox.opset.ai.onnx.v19 import _Reshape, reshape
from spox.opset.ai.onnx.v19 import _Resize, resize
from spox.opset.ai.onnx.v19 import _ReverseSequence, reverse_sequence
from spox.opset.ai.onnx.v19 import _RoiAlign, roi_align
from spox.opset.ai.onnx.v19 import _Round, round
from spox.opset.ai.onnx.v19 import _STFT, stft
from spox.opset.ai.onnx.v19 import _Scan, scan
from spox.opset.ai.onnx.v19 import _ScatterElements, scatter_elements
from spox.opset.ai.onnx.v19 import _ScatterND, scatter_nd
from spox.opset.ai.onnx.v19 import _Selu, selu
from spox.opset.ai.onnx.v19 import _SequenceAt, sequence_at
from spox.opset.ai.onnx.v19 import _SequenceConstruct, sequence_construct
from spox.opset.ai.onnx.v19 import _SequenceEmpty, sequence_empty
from spox.opset.ai.onnx.v19 import _SequenceErase, sequence_erase
from spox.opset.ai.onnx.v19 import _SequenceInsert, sequence_insert
from spox.opset.ai.onnx.v19 import _SequenceLength, sequence_length
from spox.opset.ai.onnx.v19 import _SequenceMap, sequence_map
from spox.opset.ai.onnx.v19 import _Shape, shape
from spox.opset.ai.onnx.v19 import _Shrink, shrink
from spox.opset.ai.onnx.v19 import _Sigmoid, sigmoid
from spox.opset.ai.onnx.v19 import _Sign, sign
from spox.opset.ai.onnx.v19 import _Sin, sin
from spox.opset.ai.onnx.v19 import _Sinh, sinh
from spox.opset.ai.onnx.v19 import _Size, size
from spox.opset.ai.onnx.v19 import _Slice, slice
from spox.opset.ai.onnx.v19 import _Softmax, softmax
from spox.opset.ai.onnx.v19 import _SoftmaxCrossEntropyLoss, softmax_cross_entropy_loss
from spox.opset.ai.onnx.v19 import _Softplus, softplus
from spox.opset.ai.onnx.v19 import _Softsign, softsign
from spox.opset.ai.onnx.v19 import _SpaceToDepth, space_to_depth
from spox.opset.ai.onnx.v19 import _Split, split
from spox.opset.ai.onnx.v19 import _SplitToSequence, split_to_sequence
from spox.opset.ai.onnx.v19 import _Sqrt, sqrt
from spox.opset.ai.onnx.v19 import _Squeeze, squeeze
from spox.opset.ai.onnx.v19 import _StringNormalizer, string_normalizer
from spox.opset.ai.onnx.v19 import _Sub, sub
from spox.opset.ai.onnx.v19 import _Sum, sum
from spox.opset.ai.onnx.v19 import _Tan, tan
from spox.opset.ai.onnx.v19 import _Tanh, tanh
from spox.opset.ai.onnx.v19 import _TfIdfVectorizer, tf_idf_vectorizer
from spox.opset.ai.onnx.v19 import _ThresholdedRelu, thresholded_relu
from spox.opset.ai.onnx.v19 import _Tile, tile
from spox.opset.ai.onnx.v19 import _TopK, top_k
from spox.opset.ai.onnx.v19 import _Transpose, transpose
from spox.opset.ai.onnx.v19 import _Trilu, trilu
from spox.opset.ai.onnx.v19 import _Unique, unique
from spox.opset.ai.onnx.v19 import _Unsqueeze, unsqueeze
from spox.opset.ai.onnx.v19 import _Where, where
from spox.opset.ai.onnx.v19 import _Xor, xor
class _AffineGrid(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        align_corners: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        theta: VarInfo
        size: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        grid: VarInfo

    op_type = OpType("AffineGrid", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _ConstantOfShape(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        value: Optional[AttrTensor]

    @dataclass
    class Inputs(BaseInputs):
        input: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        output: VarInfo

    op_type = OpType("ConstantOfShape", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _DFT(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        inverse: AttrInt64
        onesided: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: VarInfo
        dft_length: Optional[VarInfo]
        axis: Optional[VarInfo]

    @dataclass
    class Outputs(BaseOutputs):
        output: VarInfo

    op_type = OpType("DFT", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _Gelu(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        approximate: AttrString

    @dataclass
    class Inputs(BaseInputs):
        X: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: VarInfo

    op_type = OpType("Gelu", "", 20)

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
        X: VarInfo
        grid: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: VarInfo

    op_type = OpType("GridSample", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _ImageDecoder(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pixel_format: AttrString

    @dataclass
    class Inputs(BaseInputs):
        encoded_stream: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        image: VarInfo

    op_type = OpType("ImageDecoder", "", 20)

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
        X: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: VarInfo

    op_type = OpType("IsInf", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _IsNaN(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: VarInfo

    op_type = OpType("IsNaN", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _ReduceMax(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        keepdims: AttrInt64
        noop_with_empty_axes: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: VarInfo
        axes: Optional[VarInfo]

    @dataclass
    class Outputs(BaseOutputs):
        reduced: VarInfo

    op_type = OpType("ReduceMax", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _ReduceMin(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        keepdims: AttrInt64
        noop_with_empty_axes: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        data: VarInfo
        axes: Optional[VarInfo]

    @dataclass
    class Outputs(BaseOutputs):
        reduced: VarInfo

    op_type = OpType("ReduceMin", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _RegexFullMatch(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pattern: Optional[AttrString]

    @dataclass
    class Inputs(BaseInputs):
        X: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: VarInfo

    op_type = OpType("RegexFullMatch", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _StringConcat(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: VarInfo
        Y: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Z: VarInfo

    op_type = OpType("StringConcat", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

class _StringSplit(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        delimiter: Optional[AttrString]
        maxsplit: Optional[AttrInt64]

    @dataclass
    class Inputs(BaseInputs):
        X: VarInfo

    @dataclass
    class Outputs(BaseOutputs):
        Y: VarInfo
        Z: VarInfo

    op_type = OpType("StringSplit", "", 20)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs

def affine_grid(theta: Var, size: Var, *,     align_corners: int = 0, ) -> Var:
    r"""
Generates a 2D or 3D flow field (sampling grid), given a batch of affine
matrices theta
(https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
An affine matrix ``theta`` is applied to a position tensor represented
in its homogeneous expression. Here is an example in 3D:

::

   [r00, r01, r02, t0]   [x]   [x']
   [r10, r11, r12, t1] * [y] = [y']
   [r20, r21, r22, t2]   [z]   [z']
   [0,   0,   0,   1 ]   [1]   [1 ]

where ``(x, y, z)`` is the position in the original space,
``(x', y', z')`` is the position in the output space. The last row is
always ``[0, 0, 0, 1]`` and is not stored in the affine matrix.
Therefore we have ``theta`` of shape ``(N, 2, 3)`` for 2D or
``(N, 3, 4)`` for 3D.

Input ``size`` is used to define grid of positions evenly spaced in the
original 2D or 3D space, with dimensions ranging from ``-1`` to ``1``.
The output ``grid`` contains positions in the output space.

When ``align_corners=1``, consider ``-1`` and ``1`` to refer to the
centers of the corner pixels (mark ``v`` in illustration).

::

   v            v            v            v
   |-------------------|------------------|
   -1                  0                  1

When ``align_corners=0``, consider ``-1`` and ``1`` to refer to the
outer edge of the corner pixels.

::

       v        v         v         v
   |------------------|-------------------|
   -1                 0                   1

Parameters
==========
theta
    Type T1.
    input batch of affine matrices with shape (N, 2, 3) for 2D or (N, 3, 4)
    for 3D
size
    Type T2.
    the target output image size (N, C, H, W) for 2D or (N, C, D, H, W) for
    3D
align_corners
    Attribute.
    if align_corners=1, consider -1 and 1 to refer to the centers of the
    corner pixels. if align_corners=0, consider -1 and 1 to refer to the
    outer edge the corner pixels.

Returns
=======
grid : Var
    Type T1.
    output tensor of shape (N, H, W, 2) of 2D sample coordinates or (N, D,
    H, W, 3) of 3D sample coordinates.

Notes
=====
Signature: ``ai.onnx@20::AffineGrid``.

Type constraints:
 - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
 - T2: `tensor(int64)`
    """
    return _AffineGrid(
        _AffineGrid.Attributes(
        align_corners=AttrInt64(align_corners, name="align_corners"),
        ), _AffineGrid.Inputs(
    theta=unwrap_vars(theta), size=unwrap_vars(size),     ), ).get_output_vars(
    theta=get_value(theta), size=get_value(size),     ).grid


def constant_of_shape(input: Var, *,     value: Optional[np.ndarray] = None, ) -> Var:
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
Signature: ``ai.onnx@20::ConstantOfShape``.

Type constraints:
 - T1: `tensor(int64)`
 - T2: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ConstantOfShape(
        _ConstantOfShape.Attributes(
        value=AttrTensor.maybe(value, name="value"),
        ), _ConstantOfShape.Inputs(
    input=unwrap_vars(input),     ), ).get_output_vars(
    input=get_value(input),     ).output


def dft(input: Var, dft_length: Optional[Var] = None, axis: Optional[Var] = None, *,     inverse: int = 0,     onesided: int = 0, ) -> Var:
    r"""
Computes the discrete Fourier Transform (DFT) of the input.

Assuming the input has shape ``[M, N]``, where ``N`` is the dimension
over which the DFT is computed and ``M`` denotes the conceptual "all
other dimensions," the DFT ``y[m, k]`` of shape ``[M, N]`` is defined as

.. math:: y[m, k] = \sum_{n=0}^{N-1} e^{-2 \pi j \frac{k n}{N} } x[m, n] ,

and the inverse transform is defined as

.. math:: x[m, n] = \frac{1}{N} \sum_{k=0}^{N-1} e^{2 \pi j \frac{k n}{N} } y[m, k] ,

where :math:`j` is the imaginary unit.

The actual shape of the output is specified in the "output" section.

Reference: https://docs.scipy.org/doc/scipy/tutorial/fft.html

Parameters
==========
input
    Type T1.
    For real input, the following shape is expected:
    ``[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][1]``. For
    complex input, the following shape is expected:
    ``[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]``. The
    final dimension represents the real and imaginary parts of the value in
    that order.
dft_length
    Type T2.
    The length of the signal as a scalar. If greater than the axis
    dimension, the signal will be zero-padded up to ``dft_length``. If less
    than the axis dimension, only the first ``dft_length`` values will be
    used as the signal.
axis
    Type tensor(int64).
    The axis as a scalar on which to perform the DFT. Default is ``-2``
    (last signal axis). Negative value means counting dimensions from the
    back. Accepted range is :math:`[-r, -2] \cup [0, r-2]` where
    ``r = rank(input)``. The last dimension is for representing complex
    numbers and thus is an invalid axis.
inverse
    Attribute.
    Whether to perform the inverse discrete Fourier Transform. Default is 0,
    which corresponds to ``false``.
onesided
    Attribute.
    If ``onesided`` is ``1`` and input is real, only values for ``k`` in
    ``[0, 1, 2, ..., floor(n_fft/2) + 1]`` are returned because the
    real-to-complex Fourier transform satisfies the conjugate symmetry,
    i.e., ``X[m, k] = X[m, n_fft-k]*``, where ``m`` denotes "all other
    dimensions" DFT was not applied on. If the input tensor is complex,
    onesided output is not possible. Value can be ``0`` or ``1``. Default is
    ``0``.

Returns
=======
output : Var
    Type T1.
    The Fourier Transform of the input vector. If ``onesided`` is ``0``, the
    following shape is expected:
    ``[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]``. If
    ``axis=0`` and ``onesided`` is ``1``, the following shape is expected:
    ``[floor(signal_dim0/2)+1][signal_dim1][signal_dim2]...[signal_dimN][2]``.
    If ``axis=1`` and ``onesided`` is ``1``, the following shape is
    expected:
    ``[signal_dim0][floor(signal_dim1/2)+1][signal_dim2]...[signal_dimN][2]``.
    If ``axis=N`` and ``onesided`` is ``1``, the following shape is
    expected:
    ``[signal_dim0][signal_dim1][signal_dim2]...[floor(signal_dimN/2)+1][2]``.
    The ``signal_dim`` at the specified ``axis`` is equal to the
    ``dft_length``.

Notes
=====
Signature: ``ai.onnx@20::DFT``.

Type constraints:
 - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
 - T2: `tensor(int32)`, `tensor(int64)`
    """
    return _DFT(
        _DFT.Attributes(
        inverse=AttrInt64(inverse, name="inverse"),
        onesided=AttrInt64(onesided, name="onesided"),
        ), _DFT.Inputs(
    input=unwrap_vars(input), dft_length=unwrap_vars(dft_length), axis=unwrap_vars(axis),     ), ).get_output_vars(
    input=get_value(input), dft_length=get_value(dft_length), axis=get_value(axis),     ).output


def gelu(X: Var, *,     approximate: str = "none", ) -> Var:
    r"""
Gelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the gaussian error linear units function,
:math:`y = 0.5 * x * (1 + erf(x/sqrt(2)))` is applied to the tensor
elementwise. If the attribute "approximate" is set to "tanh", the
function estimation,
:math:`y = 0.5 * x * (1 + Tanh(sqrt(2/\pi) * (x + 0.044715 * x^3)))` is
used and applied to the tensor elementwise.

Parameters
==========
X
    Type T.
    Input tensor
approximate
    Attribute.
    Gelu approximation algorithm: ``"tanh"``,
    ``"none"``\ (default).\ ``"none"``: do not use
    approximation.\ ``"tanh"``: use tanh approximation.

Returns
=======
Y : Var
    Type T.
    Output tensor

Notes
=====
Signature: ``ai.onnx@20::Gelu``.

Type constraints:
 - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Gelu(
        _Gelu.Attributes(
        approximate=AttrString(approximate, name="approximate"),
        ), _Gelu.Inputs(
    X=unwrap_vars(X),     ), ).get_output_vars(
    X=get_value(X),     ).Y


def grid_sample(X: Var, grid: Var, *,     align_corners: int = 0,     mode: str = "linear",     padding_mode: str = "zeros", ) -> Var:
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
Signature: ``ai.onnx@20::GridSample``.

Type constraints:
 - T1: `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
 - T2: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _GridSample(
        _GridSample.Attributes(
        align_corners=AttrInt64(align_corners, name="align_corners"),
        mode=AttrString(mode, name="mode"),
        padding_mode=AttrString(padding_mode, name="padding_mode"),
        ), _GridSample.Inputs(
    X=unwrap_vars(X), grid=unwrap_vars(grid),     ), ).get_output_vars(
    X=get_value(X), grid=get_value(grid),     ).Y


def image_decoder(encoded_stream: Var, *,     pixel_format: str = "RGB", ) -> Var:
    r"""
Loads and decodes and image from a file. If it can't decode for any
reason (e.g. corrupted encoded stream, invalid format, it will return an
empty matrix). The following image formats are supported:

-  BMP
-  JPEG (note: Lossless JPEG support is optional)
-  JPEG2000
-  TIFF
-  PNG
-  WebP
-  Portable image format (PBM, PGM, PPM, PXM, PNM) Decoded images follow
   a channel-last layout: (Height, Width, Channels). **JPEG chroma
   upsampling method:** When upsampling the chroma components by a
   factor of 2, the pixels are linearly interpolated so that the centers
   of the output pixels are 1/4 and 3/4 of the way between input pixel
   centers. When rounding, 0.5 is rounded down and up at alternative
   pixels locations to prevent bias towards larger values (ordered
   dither pattern). Considering adjacent input pixels A, B, and C, B is
   upsampled to pixels B0 and B1 so that

::

   B0 = round_half_down((1/4) * A + (3/4) * B)
   B1 = round_half_up((3/4) * B + (1/4) * C)

This method, is the default chroma upsampling method in the
well-established libjpeg-turbo library, also referred as "smooth" or
"fancy" upsampling.

Parameters
==========
encoded_stream
    Type T1.
    Encoded stream
pixel_format
    Attribute.
    Pixel format. Can be one of "RGB", "BGR", or "Grayscale".

Returns
=======
image : Var
    Type T2.
    Decoded image

Notes
=====
Signature: ``ai.onnx@20::ImageDecoder``.

Type constraints:
 - T1: `tensor(uint8)`
 - T2: `tensor(uint8)`
    """
    return _ImageDecoder(
        _ImageDecoder.Attributes(
        pixel_format=AttrString(pixel_format, name="pixel_format"),
        ), _ImageDecoder.Inputs(
    encoded_stream=unwrap_vars(encoded_stream),     ), ).get_output_vars(
    encoded_stream=get_value(encoded_stream),     ).image


def isinf(X: Var, *,     detect_negative: int = 1,     detect_positive: int = 1, ) -> Var:
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
Signature: ``ai.onnx@20::IsInf``.

Type constraints:
 - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
 - T2: `tensor(bool)`
    """
    return _IsInf(
        _IsInf.Attributes(
        detect_negative=AttrInt64(detect_negative, name="detect_negative"),
        detect_positive=AttrInt64(detect_positive, name="detect_positive"),
        ), _IsInf.Inputs(
    X=unwrap_vars(X),     ), ).get_output_vars(
    X=get_value(X),     ).Y


def isnan(X: Var, ) -> Var:
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
Signature: ``ai.onnx@20::IsNaN``.

Type constraints:
 - T1: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`
 - T2: `tensor(bool)`
    """
    return _IsNaN(
        _IsNaN.Attributes(
        ), _IsNaN.Inputs(
    X=unwrap_vars(X),     ), ).get_output_vars(
    X=get_value(X),     ).Y


def reduce_max(data: Var, axes: Optional[Var] = None, *,     keepdims: int = 1,     noop_with_empty_axes: int = 0, ) -> Var:
    r"""
Computes the max of the input tensor's elements along the provided axes.
The resulting tensor has the same rank as the input if ``keepdims``
equals 1. If ``keepdims`` equals 0, then the resulting tensor has the
reduced dimension pruned. Input tensors of rank zero are valid.
Reduction over an empty set of values yields minus infinity (if
supported by the datatype) or the minimum value of the data type
otherwise.

If the input data type is Boolean, the comparison should consider
``False < True``.

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
Signature: ``ai.onnx@20::ReduceMax``.

Type constraints:
 - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ReduceMax(
        _ReduceMax.Attributes(
        keepdims=AttrInt64(keepdims, name="keepdims"),
        noop_with_empty_axes=AttrInt64(noop_with_empty_axes, name="noop_with_empty_axes"),
        ), _ReduceMax.Inputs(
    data=unwrap_vars(data), axes=unwrap_vars(axes),     ), ).get_output_vars(
    data=get_value(data), axes=get_value(axes),     ).reduced


def reduce_min(data: Var, axes: Optional[Var] = None, *,     keepdims: int = 1,     noop_with_empty_axes: int = 0, ) -> Var:
    r"""
Computes the min of the input tensor's elements along the provided axes.
The resulting tensor has the same rank as the input if ``keepdims``
equals 1. If ``keepdims`` equals 0, then the resulting tensor has the
reduced dimension pruned. Input tensors of rank zero are valid.
Reduction over an empty set of values yields plus infinity (if supported
by the datatype) or the maximum value of the data type otherwise.

If the input data type is Boolean, the comparison should consider
``False < True``.

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
Signature: ``ai.onnx@20::ReduceMin``.

Type constraints:
 - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ReduceMin(
        _ReduceMin.Attributes(
        keepdims=AttrInt64(keepdims, name="keepdims"),
        noop_with_empty_axes=AttrInt64(noop_with_empty_axes, name="noop_with_empty_axes"),
        ), _ReduceMin.Inputs(
    data=unwrap_vars(data), axes=unwrap_vars(axes),     ), ).get_output_vars(
    data=get_value(data), axes=get_value(axes),     ).reduced


def regex_full_match(X: Var, *,     pattern: Optional[str] = None, ) -> Var:
    r"""
RegexFullMatch performs a full regex match on each element of the input
tensor. If an element fully matches the regex pattern specified as an
attribute, the corresponding element in the output is True and it is
False otherwise. `RE2 <https://github.com/google/re2/wiki/Syntax>`__
regex syntax is used.

Parameters
==========
X
    Type T1.
    Tensor with strings to match on.
pattern
    Attribute.
    Regex pattern to match on. This must be valid RE2 syntax.

Returns
=======
Y : Var
    Type T2.
    Tensor of bools indicating if each input string fully matches the regex
    pattern specified.

Notes
=====
Signature: ``ai.onnx@20::RegexFullMatch``.

Type constraints:
 - T1: `tensor(string)`
 - T2: `tensor(bool)`
    """
    return _RegexFullMatch(
        _RegexFullMatch.Attributes(
        pattern=AttrString.maybe(pattern, name="pattern"),
        ), _RegexFullMatch.Inputs(
    X=unwrap_vars(X),     ), ).get_output_vars(
    X=get_value(X),     ).Y


def string_concat(X: Var, Y: Var, ) -> Var:
    r"""
StringConcat concatenates string tensors elementwise (with NumPy-style
broadcasting support)

Parameters
==========
X
    Type T.
    Tensor to prepend in concatenation
Y
    Type T.
    Tensor to append in concatenation

Returns
=======
Z : Var
    Type T.
    Concatenated string tensor

Notes
=====
Signature: ``ai.onnx@20::StringConcat``.

Type constraints:
 - T: `tensor(string)`
    """
    return _StringConcat(
        _StringConcat.Attributes(
        ), _StringConcat.Inputs(
    X=unwrap_vars(X), Y=unwrap_vars(Y),     ), ).get_output_vars(
    X=get_value(X), Y=get_value(Y),     ).Z


def string_split(X: Var, *,     delimiter: Optional[str] = None,     maxsplit: Optional[int] = None, ) -> tuple[Var, Var]:
    r"""
StringSplit splits a string tensor's elements into substrings based on a
delimiter attribute and a maxsplit attribute.

The first output of this operator is a tensor of strings representing
the substrings from splitting each input string on the ``delimiter``
substring. This tensor has one additional rank compared to the input
tensor in order to store the substrings for each input element (where
the input tensor is not empty). Note that, in order to ensure the same
number of elements are present in the final dimension, this tensor will
pad empty strings as illustrated in the examples below. Consecutive
delimiters are not grouped together and are deemed to delimit empty
strings, except if the ``delimiter`` is unspecified or is the empty
string (""). In the case where the ``delimiter`` is unspecified or the
empty string, consecutive whitespace characters are regarded as a single
separator and leading or trailing whitespace is removed in the output.

The second output tensor represents the number of substrings generated.
``maxsplit`` can be used to limit the number of splits performed - after
the ``maxsplit``\ th split if the string is not fully split, the
trailing suffix of input string after the final split point is also
added. For elements where fewer splits are possible than specified in
``maxsplit``, it has no effect.

Parameters
==========
X
    Type T1.
    Tensor of strings to split.
delimiter
    Attribute.
    Delimiter to split on. If left unset or set to the empty string (""),
    the input is split on consecutive whitespace.
maxsplit
    Attribute.
    Maximum number of splits (from left to right). If left unset (or if the
    number of possible splits are less than maxsplit), it will make as many
    splits as possible. Note that the maximum possible number of substrings
    returned with ``maxsplit`` specified is ``maxsplit+1`` since the
    remaining suffix after the ``maxsplit``\ th split is included in the
    output.

Returns
=======
Y : Var
    Type T2.
    Tensor of substrings representing the outcome of splitting the strings
    in the input on the delimiter. Note that to ensure the same number of
    elements are present in the final rank, this tensor will pad any
    necessary empty strings.
Z : Var
    Type T3.
    The number of substrings generated for each input element.

Notes
=====
Signature: ``ai.onnx@20::StringSplit``.

Type constraints:
 - T1: `tensor(string)`
 - T2: `tensor(string)`
 - T3: `tensor(int64)`
    """
    return _StringSplit(
        _StringSplit.Attributes(
        delimiter=AttrString.maybe(delimiter, name="delimiter"),
        maxsplit=AttrInt64.maybe(maxsplit, name="maxsplit"),
        ), _StringSplit.Inputs(
    X=unwrap_vars(X),     ), ).get_output_vars(
    X=get_value(X),     )._unpack_to_any()


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

__all__ = [fun.__name__ for fun in _CONSTRUCTORS.values()]  + ["const"] 
