# ruff: noqa: E741 -- Allow ambiguous variable name
from dataclasses import dataclass
from typing import (
    Iterable,
    Optional,
    Sequence,
)

import numpy as np
import numpy.typing as npt

from spox._attributes import (
    AttrFloat32,
    AttrInt64,
    AttrInt64s,
    AttrString,
)
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._node import OpType
from spox._standard import StandardNode
from spox._var import Var
from spox.opset.ai.onnx.v17 import (
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
    _And,
    _ArgMax,
    _ArgMin,
    _Asin,
    _Asinh,
    _Atan,
    _Atanh,
    _AveragePool,
    _BatchNormalization,
    _Bernoulli,
    _BitShift,
    _BlackmanWindow,
    _Cast,
    _CastLike,
    _Ceil,
    _Celu,
    _Clip,
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
    _Gemm,
    _GlobalAveragePool,
    _GlobalLpPool,
    _GlobalMaxPool,
    _Greater,
    _GreaterOrEqual,
    _GridSample,
    _HammingWindow,
    _HannWindow,
    _Hardmax,
    _HardSigmoid,
    _HardSwish,
    _Identity,
    _If,
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
    _Or,
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
    _ReduceSum,
    _Relu,
    _Reshape,
    _ReverseSequence,
    _RoiAlign,
    _Round,
    _Scan,
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
    _SplitToSequence,
    _Sqrt,
    _Squeeze,
    _StringNormalizer,
    _Sub,
    _Sum,
    _Tan,
    _Tanh,
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
    and_,
    arg_max,
    arg_min,
    asin,
    asinh,
    atan,
    atanh,
    average_pool,
    batch_normalization,
    bernoulli,
    bit_shift,
    blackman_window,
    cast,
    cast_like,
    ceil,
    celu,
    clip,
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
    gemm,
    global_average_pool,
    global_lp_pool,
    global_max_pool,
    greater,
    greater_or_equal,
    grid_sample,
    gru,
    hamming_window,
    hann_window,
    hard_sigmoid,
    hard_swish,
    hardmax,
    identity,
    if_,
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
    or_,
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
    reduce_sum,
    relu,
    reshape,
    reverse_sequence,
    rnn,
    roi_align,
    round,
    scan,
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
    split_to_sequence,
    sqrt,
    squeeze,
    stft,
    string_normalizer,
    sub,
    sum,
    tan,
    tanh,
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


class _BitwiseAnd(StandardNode):
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

    op_type = OpType("BitwiseAnd", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _BitwiseNot(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("BitwiseNot", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _BitwiseOr(StandardNode):
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

    op_type = OpType("BitwiseOr", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _BitwiseXor(StandardNode):
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

    op_type = OpType("BitwiseXor", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _CenterCropPad(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axes: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        input_data: Var
        shape: Var

    @dataclass
    class Outputs(BaseOutputs):
        output_data: Var

    op_type = OpType("CenterCropPad", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Col2Im(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dilations: Optional[AttrInt64s]
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        image_shape: Var
        block_shape: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Col2Im", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _GroupNormalization(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        epsilon: AttrFloat32
        num_groups: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        scale: Var
        bias: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("GroupNormalization", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LpPool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        ceil_mode: AttrInt64
        dilations: Optional[AttrInt64s]
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

    op_type = OpType("LpPool", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Mish(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("Mish", "", 18)

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

    op_type = OpType("OptionalGetElement", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _OptionalHasElement(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        pass

    @dataclass
    class Inputs(BaseInputs):
        input: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("OptionalHasElement", "", 18)

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
        axes: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Pad", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceL1(StandardNode):
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

    op_type = OpType("ReduceL1", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceL2(StandardNode):
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

    op_type = OpType("ReduceL2", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceLogSum(StandardNode):
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

    op_type = OpType("ReduceLogSum", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceLogSumExp(StandardNode):
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

    op_type = OpType("ReduceLogSumExp", "", 18)

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
        data: Var
        axes: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceMax", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceMean(StandardNode):
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

    op_type = OpType("ReduceMean", "", 18)

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
        data: Var
        axes: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        reduced: Var

    op_type = OpType("ReduceMin", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceProd(StandardNode):
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

    op_type = OpType("ReduceProd", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ReduceSumSquare(StandardNode):
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

    op_type = OpType("ReduceSumSquare", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Resize(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        antialias: AttrInt64
        axes: Optional[AttrInt64s]
        coordinate_transformation_mode: AttrString
        cubic_coeff_a: AttrFloat32
        exclude_outside: AttrInt64
        extrapolation_value: AttrFloat32
        keep_aspect_ratio_policy: AttrString
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

    op_type = OpType("Resize", "", 18)

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

    op_type = OpType("ScatterElements", "", 18)

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

    op_type = OpType("ScatterND", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Split(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        num_outputs: Optional[AttrInt64]

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        split: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        outputs: Sequence[Var]

    op_type = OpType("Split", "", 18)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def bitwise_and(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulting from performing the bitwise ``and``
    operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the bitwise operator.
    B
        Type T.
        Second input operand for the bitwise operator.

    Returns
    =======
    C : Var
        Type T.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@18::BitwiseAnd``.

    Type constraints:
     - T: `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _BitwiseAnd(
        _BitwiseAnd.Attributes(),
        _BitwiseAnd.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def bitwise_not(
    X: Var,
) -> Var:
    r"""
    Returns the bitwise not of the input tensor element-wise.

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
    Signature: ``ai.onnx@18::BitwiseNot``.

    Type constraints:
     - T: `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _BitwiseNot(
        _BitwiseNot.Attributes(),
        _BitwiseNot.Inputs(
            X=X,
        ),
    ).outputs.Y


def bitwise_or(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulting from performing the bitwise ``or``
    operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the bitwise operator.
    B
        Type T.
        Second input operand for the bitwise operator.

    Returns
    =======
    C : Var
        Type T.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@18::BitwiseOr``.

    Type constraints:
     - T: `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _BitwiseOr(
        _BitwiseOr.Attributes(),
        _BitwiseOr.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def bitwise_xor(
    A: Var,
    B: Var,
) -> Var:
    r"""
    Returns the tensor resulting from performing the bitwise ``xor``
    operation elementwise on the input tensors ``A`` and ``B`` (with
    Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style)
    broadcasting**; for more details please check `the
    doc <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>`__.

    Parameters
    ==========
    A
        Type T.
        First input operand for the bitwise operator.
    B
        Type T.
        Second input operand for the bitwise operator.

    Returns
    =======
    C : Var
        Type T.
        Result tensor.

    Notes
    =====
    Signature: ``ai.onnx@18::BitwiseXor``.

    Type constraints:
     - T: `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _BitwiseXor(
        _BitwiseXor.Attributes(),
        _BitwiseXor.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


def center_crop_pad(
    input_data: Var,
    shape: Var,
    *,
    axes: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    Center crop or pad an input to given dimensions.

    The crop/pad dimensions can be specified for a subset of the ``axes``.
    Non-specified dimensions will not be cropped or padded.

    If the input dimensions are bigger than the crop shape, a centered
    cropping window is extracted from the input. If the input dimensions are
    smaller than the crop shape, the input is padded on each side equally,
    so that the input is centered in the output.

    Parameters
    ==========
    input_data
        Type T.
        Input to extract the centered crop from.
    shape
        Type Tind.
        1-D tensor representing the cropping window dimensions.
    axes
        Attribute.
        If provided, it specifies a subset of axes that 'shape' refer to. If not
        provided, all axes are assumed [0, 1, ..., r-1], where r = rank(data).
        Negative value means counting dimensions from the back. Accepted range
        is [-r, r-1], where r = rank(data). Behavior is undefined if an axis is
        repeated.

    Returns
    =======
    output_data : Var
        Type T.
        Output data.

    Notes
    =====
    Signature: ``ai.onnx@18::CenterCropPad``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - Tind: `tensor(int32)`, `tensor(int64)`
    """
    return _CenterCropPad(
        _CenterCropPad.Attributes(
            axes=AttrInt64s.maybe(axes, name="axes"),
        ),
        _CenterCropPad.Inputs(
            input_data=input_data,
            shape=shape,
        ),
    ).outputs.output_data


def col2_im(
    input: Var,
    image_shape: Var,
    block_shape: Var,
    *,
    dilations: Optional[Iterable[int]] = None,
    pads: Optional[Iterable[int]] = None,
    strides: Optional[Iterable[int]] = None,
) -> Var:
    r"""
    The operator rearranges column blocks back into a multidimensional image

    Col2Im behaves similarly to PyTorch's fold
    https://pytorch.org/docs/stable/generated/torch.nn.Fold.html, but it
    only supports *batched* multi-dimensional image tensors. Another
    implementation in Python with N-dimension support can be found at
    https://github.com/f-dangel/unfoldNd/.

    NOTE: Although specifying image_shape looks redundant because it could
    be calculated from convolution formulas, it is required as input for
    more advanced scenarios as explained at PyTorch's implementation
    (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Col2Im.cpp#L10)

    Parameters
    ==========
    input
        Type T.
        Input data tensor to be rearranged from column blocks back into an
        image. This is a 3-dimensional tensor containing [N, C \*
        n-ary-product(block_shape), L], where N is batch dimension, C is image
        channel dimension and L is number of blocks.The blocks are enumerated in
        increasing lexicographic-order of their indices.For example, with an
        image-size 10\ *20 and block-size 9*\ 18, there would be 2*3 blocks,
        enumerated in the order block(0, 0), block(0, 1), block(0, 2), block(1,
        0), block(1, 1), block(1, 2).
    image_shape
        Type tensor(int64).
        The shape of the spatial dimensions of the image after rearranging the
        column blocks.This is a 1-dimensional tensor with size of at least 2,
        containing the value [H_img, W_img] for a 2-D image or [dim_i1, dim_i2,
        ..., dim_iN] for a N-D image.
    block_shape
        Type tensor(int64).
        The shape of the block to apply on the input.This is a 1-dimensional
        tensor of size of at least 2, containing the value [H_block, W_block]
        for a 2-D image or [dim_b1, dim_b2, ..., dim_bN] for a N-D block.This is
        the block-shape before dilation is applied to it.
    dilations
        Attribute.
        1-dimensional tensor with dilation value along each spatial axis of the
        image. If not present, the dilation defaults to 1 along each spatial
        axis of the image.
    pads
        Attribute.
        1-dimensional tensor with padding value for the beginning and ending
        along each spatial axis, it can take any value greater than or equal to
        0. The value represent the number of pixels added to the beginning and
        end part of the corresponding axis. ``pads`` format should be as follow
        [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin is the number
        of pixels added at the beginning of axis ``i`` and xi_end is the number
        of pixels added at the end of axis ``i``. If not present, the padding
        defaults to 0 along start and end of each spatial axis.
    strides
        Attribute.
        1-dimensional tensor with stride value along each spatial axis. If not
        present, the stride defaults to 1 along each spatial axis.

    Returns
    =======
    output : Var
        Type T.
        Output tensor produced by rearranging blocks into an image.

    Notes
    =====
    Signature: ``ai.onnx@18::Col2Im``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Col2Im(
        _Col2Im.Attributes(
            dilations=AttrInt64s.maybe(dilations, name="dilations"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _Col2Im.Inputs(
            input=input,
            image_shape=image_shape,
            block_shape=block_shape,
        ),
    ).outputs.output


def group_normalization(
    X: Var,
    scale: Var,
    bias: Var,
    *,
    epsilon: float = 9.999999747378752e-06,
    num_groups: int,
) -> Var:
    r"""
    A GroupNormalization function. Carries out group normalization as
    described in the paper https://arxiv.org/abs/1803.08494

    This operator transforms input according to

    ::

       y = scale * (x - mean) / sqrt(variance + epsilon) + bias,

    where the mean and variance are computed per instance per group of
    channels, and ``scale`` and ``bias`` should be specified for each group
    of channels. The number of groups ``num_groups`` should be divisible by
    the number of channels so that there are an equal number of channels per
    group.

    When the number of groups is the same as the number of channels, this
    operator is equivalent to InstanceNormalization. When there is only one
    group, this operator is equivalent to LayerNormalization.

    Parameters
    ==========
    X
        Type T.
        Input data tensor. Dimensions for image cases are ``(N x C x H x W)``,
        where ``N`` is the batch size, ``C`` is the number of channels, and
        ``H`` and ``W`` are the height and width of the data. Statistics are
        computed for every group of channels over ``C``, ``H``, and ``W``. For
        non-image cases, the dimensions are in the form of
        ``(N x C x D1 x D2 ... Dn)``.
    scale
        Type T.
        Scale tensor of shape ``(num_groups)``.
    bias
        Type T.
        Bias tensor of shape ``(num_groups)``.
    epsilon
        Attribute.
        The epsilon value to use to avoid division by zero.
    num_groups
        Attribute.
        The number of groups of channels. It should be a divisor of the number
        of channels ``C``.

    Returns
    =======
    Y : Var
        Type T.
        The output tensor of the same shape as ``X``.

    Notes
    =====
    Signature: ``ai.onnx@18::GroupNormalization``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _GroupNormalization(
        _GroupNormalization.Attributes(
            epsilon=AttrFloat32(epsilon, name="epsilon"),
            num_groups=AttrInt64(num_groups, name="num_groups"),
        ),
        _GroupNormalization.Inputs(
            X=X,
            scale=scale,
            bias=bias,
        ),
    ).outputs.Y


def lp_pool(
    X: Var,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    dilations: Optional[Iterable[int]] = None,
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
    Signature: ``ai.onnx@18::LpPool``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _LpPool(
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
            X=X,
        ),
    ).outputs.Y


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
    Signature: ``ai.onnx@18::Mish``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Mish(
        _Mish.Attributes(),
        _Mish.Inputs(
            X=X,
        ),
    ).outputs.Y


def optional_get_element(
    input: Var,
) -> Var:
    r"""
    If the input is a tensor or sequence type, it returns the input. If the
    input is an optional type, it outputs the element in the input. It is an
    error if the input is an empty optional-type (i.e. does not have an
    element) and the behavior is undefined in this case.

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
    Signature: ``ai.onnx@18::OptionalGetElement``.

    Type constraints:
     - O: `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`, `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - V: `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _OptionalGetElement(
        _OptionalGetElement.Attributes(),
        _OptionalGetElement.Inputs(
            input=input,
        ),
    ).outputs.output


def optional_has_element(
    input: Optional[Var] = None,
) -> Var:
    r"""
    Returns true if (1) the input is an optional-type and contains an
    element, or, (2) the input is a tensor or sequence type. If the input is
    not provided or is an empty optional-type, this op returns false.

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
    Signature: ``ai.onnx@18::OptionalHasElement``.

    Type constraints:
     - O: `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`, `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - B: `tensor(bool)`
    """
    return _OptionalHasElement(
        _OptionalHasElement.Attributes(),
        _OptionalHasElement.Inputs(
            input=input,
        ),
    ).outputs.output


def pad(
    data: Var,
    pads: Var,
    constant_value: Optional[Var] = None,
    axes: Optional[Var] = None,
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

    Example 1 (``constant`` mode):

    Insert 0 pads to the beginning of the second dimension.

    ::

       data = [
           [1.0, 1.2],
           [2.3, 3.4],
           [4.5, 5.7],
       ]

       pads = [0, 2, 0, 0]

       mode = 'constant'

       constant_value = 0.0

       output = [
           [0.0, 0.0, 1.0, 1.2],
           [0.0, 0.0, 2.3, 3.4],
           [0.0, 0.0, 4.5, 5.7],
       ]

    Example 2 (``reflect`` mode):

    ::

       data = [
           [1.0, 1.2],
           [2.3, 3.4],
           [4.5, 5.7],
       ]

       pads = [0, 2, 0, 0]

       mode = 'reflect'

       output = [
           [1.0, 1.2, 1.0, 1.2],
           [2.3, 3.4, 2.3, 3.4],
           [4.5, 5.7, 4.5, 5.7],
       ]

    Example 3 (``edge`` mode):

    ::

       data = [
           [1.0, 1.2],
           [2.3, 3.4],
           [4.5, 5.7],
       ]

       pads = [0, 2, 0, 0]

       mode = 'edge'

       output = [
           [1.0, 1.0, 1.0, 1.2],
           [2.3, 2.3, 2.3, 3.4],
           [4.5, 4.5, 4.5, 5.7],
       ]

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
        shape [2 \* num_axes] where ``num_axes`` refers to the number of
        elements in the ``axes`` input or the input rank if ``axes`` are not
        provided explicitly. ``pads`` format should be: [x1_begin, x2_begin,
        ..., x1_end, x2_end,...], where xi_begin is the number of pad values
        added at the beginning of axis ``axes[i]`` and xi_end, the number of pad
        values added at the end of axis ``axes[i]``.
    constant_value
        Type T.
        (Optional) A scalar value to be used if the mode chosen is ``constant``
        (by default it is 0, empty string or False).
    axes
        Type Tind.
        1-D tensor of axes that ``pads`` apply to. Negative value means counting
        dimensions from the back. Accepted range is [-r, r-1] where r =
        rank(data). Behavior is undefined if an axis is repeated. If not
        provided, all axes are assumed (``[0, 1, ..., input_rank-1]``).
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
    Signature: ``ai.onnx@18::Pad``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - Tind: `tensor(int32)`, `tensor(int64)`
    """
    return _Pad(
        _Pad.Attributes(
            mode=AttrString(mode, name="mode"),
        ),
        _Pad.Inputs(
            data=data,
            pads=pads,
            constant_value=constant_value,
            axes=axes,
        ),
    ).outputs.output


def reduce_l1(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
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
    Signature: ``ai.onnx@18::ReduceL1``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceL1(
        _ReduceL1.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceL1.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def reduce_l2(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
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
    Signature: ``ai.onnx@18::ReduceL2``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceL2(
        _ReduceL2.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceL2.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def reduce_log_sum(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
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
    Signature: ``ai.onnx@18::ReduceLogSum``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceLogSum(
        _ReduceLogSum.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceLogSum.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def reduce_log_sum_exp(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
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
    Signature: ``ai.onnx@18::ReduceLogSumExp``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceLogSumExp(
        _ReduceLogSumExp.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceLogSumExp.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def reduce_max(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
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
    Signature: ``ai.onnx@18::ReduceMax``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ReduceMax(
        _ReduceMax.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceMax.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def reduce_mean(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
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
    Signature: ``ai.onnx@18::ReduceMean``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceMean(
        _ReduceMean.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceMean.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def reduce_min(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
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
    Signature: ``ai.onnx@18::ReduceMin``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _ReduceMin(
        _ReduceMin.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceMin.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def reduce_prod(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
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
    Signature: ``ai.onnx@18::ReduceProd``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceProd(
        _ReduceProd.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceProd.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def reduce_sum_square(
    data: Var,
    axes: Optional[Var] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
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
    Signature: ``ai.onnx@18::ReduceSumSquare``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)`
    """
    return _ReduceSumSquare(
        _ReduceSumSquare.Attributes(
            keepdims=AttrInt64(keepdims, name="keepdims"),
            noop_with_empty_axes=AttrInt64(
                noop_with_empty_axes, name="noop_with_empty_axes"
            ),
        ),
        _ReduceSumSquare.Inputs(
            data=data,
            axes=axes,
        ),
    ).outputs.reduced


def resize(
    X: Var,
    roi: Optional[Var] = None,
    scales: Optional[Var] = None,
    sizes: Optional[Var] = None,
    *,
    antialias: int = 0,
    axes: Optional[Iterable[int]] = None,
    coordinate_transformation_mode: str = "half_pixel",
    cubic_coeff_a: float = -0.75,
    exclude_outside: int = 0,
    extrapolation_value: float = 0.0,
    keep_aspect_ratio_policy: str = "stretch",
    mode: str = "nearest",
    nearest_mode: str = "round_prefer_floor",
) -> Var:
    r"""
    Resize the input tensor. In general, it calculates every value in the
    output tensor as a weighted average of neighborhood (a.k.a. sampling
    locations) in the input tensor. Each dimension value of the output
    tensor is:
    ``output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)``
    if input "sizes" is not specified.

    Parameters
    ==========
    X
        Type T1.
        N-D tensor
    roi
        Type T2.
        1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is
        the rank of X or the length of axes, if provided. The RoIs' coordinates
        are normalized in the coordinate system of the input image. It only
        takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
    scales
        Type tensor(float).
        The scale array along each dimension. It takes value greater than 0. If
        it's less than 1, it's sampling down, otherwise, it's upsampling. The
        number of elements of 'scales' should be the same as the rank of input
        'X' or the length of 'axes', if provided. One of 'scales' and 'sizes'
        MUST be specified and it is an error if both are specified. If 'sizes'
        is needed, the user can use an empty string as the name of 'scales' in
        this operator's input list.
    sizes
        Type tensor(int64).
        Target size of the output tensor. Its interpretation depends on the
        'keep_aspect_ratio_policy' value.The number of elements of 'sizes'
        should be the same as the rank of input 'X', or the length of 'axes', if
        provided. Only one of 'scales' and 'sizes' can be specified.
    antialias
        Attribute.
        If set to 1, "linear" and "cubic" interpolation modes will use an
        antialiasing filter when downscaling. Antialiasing is achieved by
        stretching the resampling filter by a factor max(1, 1 / scale), which
        means that when downsampling, more input pixels contribute to an output
        pixel.
    axes
        Attribute.
        If provided, it specifies a subset of axes that 'roi', 'scales' and
        'sizes' refer to. If not provided, all axes are assumed [0, 1, ...,
        r-1], where r = rank(data). Non-specified dimensions are interpreted as
        non-resizable. Negative value means counting dimensions from the back.
        Accepted range is [-r, r-1], where r = rank(data). Behavior is undefined
        if an axis is repeated.
    coordinate_transformation_mode
        Attribute.
        This attribute describes how to transform the coordinate in the resized
        tensor to the coordinate in the original tensor.

        The coordinate of each dimension is transformed individually. Let's
        describe a case using axis x as an example. Denote x_resized as the
        coordinate of axis x in the resized tensor, x_original as the coordinate
        of axis x in the original tensor, ``length_original`` as the length of
        the original tensor in axis x, length_resized as the length of the
        resized tensor in axis x, roi_x = (start_x, end_x) of the axis x in
        input "roi", ``scale = length_resized / length_original``,

        if coordinate_transformation_mode is ``"half_pixel"``,
        ``x_original = (x_resized + 0.5) / scale - 0.5``

        if coordinate_transformation_mode is ``"pytorch_half_pixel"``,
        ``x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0``

        if coordinate_transformation_mode is ``"align_corners"``,
        ``x_original = x_resized * (length_original - 1) / (length_resized - 1)``

        if coordinate_transformation_mode is ``"asymmetric"``,
        ``x_original = x_resized / scale``

        if coordinate_transformation_mode is ``"tf_crop_and_resize"``,
        ``x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1)``
        .
    cubic_coeff_a
        Attribute.
        The coefficient 'a' used in cubic interpolation. Two common choice are
        -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check out
        Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the
        details. This attribute is valid only if mode is "cubic".
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
    keep_aspect_ratio_policy
        Attribute.
        This attribute describes how to interpret the ``sizes`` input with
        regard to keeping the original aspect ratio of the input, and it is not
        applicable when the ``scales`` input is used.

        Given a set of ``sizes``, associated with a subset of ``axes``
        (explicitly provided or default), and assuming ``d = axes[i]``, with
        ``i`` being the index of the provided ``sizes``.

        If ``keep_aspect_ratio_policy`` is ``"stretch"``, the original aspect
        ratio is disregarded, and the input is resized to the specified size:
        ``out_size[d] = sizes[i]``

        If ``keep_aspect_ratio_policy`` is ``"not_larger"``, the sizes are
        adjusted so that no extent of the output is larger than the specified
        size, while keeping the original aspect ratio:
        ``scale = Min(sizes[i] / in_size[d])``
        ``out_size[d] = round_int(scale * in_size[i])``

        If ``keep_aspect_ratio_policy`` is ``"not_smaller"``, the sizes are
        adjusted so that no extent of the output is smaller than the specified
        size, while keeping the original aspect ratio:
        ``scale = Max(sizes[i] / in_size[d])``
        ``out_size[d] = round_int(scale * in_size[i])``

        For non-resizable axes (those not specified in ``axes``), the output
        size will be equal to the input size.

        Note: ``round_int`` stands for computing the nearest integer value,
        rounding halfway cases up.
    mode
        Attribute.
        Three interpolation modes: "nearest" (default), "linear" and "cubic".
        The "linear" mode includes linear interpolation for 1D tensor and
        N-linear interpolation for N-D tensor (for example, bilinear
        interpolation for 2D tensor). The "cubic" mode includes cubic
        interpolation for 1D tensor and N-cubic interpolation for N-D tensor
        (for example, bicubic interpolation for 2D tensor).
    nearest_mode
        Attribute.
        Four modes: "round_prefer_floor" (default, as known as round half down),
        "round_prefer_ceil" (as known as round half up), "floor", "ceil". Only
        used by nearest interpolation. It indicates how to get "nearest" pixel
        in input tensor from x_original, so this attribute is valid only if
        "mode" is "nearest".

    Returns
    =======
    Y : Var
        Type T1.
        N-D tensor after resizing

    Notes
    =====
    Signature: ``ai.onnx@18::Resize``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _Resize(
        _Resize.Attributes(
            antialias=AttrInt64(antialias, name="antialias"),
            axes=AttrInt64s.maybe(axes, name="axes"),
            coordinate_transformation_mode=AttrString(
                coordinate_transformation_mode, name="coordinate_transformation_mode"
            ),
            cubic_coeff_a=AttrFloat32(cubic_coeff_a, name="cubic_coeff_a"),
            exclude_outside=AttrInt64(exclude_outside, name="exclude_outside"),
            extrapolation_value=AttrFloat32(
                extrapolation_value, name="extrapolation_value"
            ),
            keep_aspect_ratio_policy=AttrString(
                keep_aspect_ratio_policy, name="keep_aspect_ratio_policy"
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
    output shape is the same as the shape of ``data``.

    For each entry in ``updates``, the target index in ``data`` is obtained
    by combining the corresponding entry in ``indices`` with the index of
    the entry itself: the index-value for dimension = axis is obtained from
    the value of the corresponding entry in ``indices`` and the index-value
    for dimension != axis is obtained from the index of the entry itself.

    ``reduction`` allows specification of an optional reduction operation,
    which is applied to all values in ``updates`` tensor into ``output`` at
    the specified ``indices``. In cases where ``reduction`` is set to
    "none", indices should not have duplicate entries: that is, if idx1 !=
    idx2, then indices[idx1] != indices[idx2]. For instance, in a 2-D tensor
    case, the update corresponding to the [i][j] entry is performed as
    below:

    ::

       output[indices[i][j]][j] = updates[i][j] if axis = 0,
       output[i][indices[i][j]] = updates[i][j] if axis = 1,

    When ``reduction`` is set to some reduction function ``f``, the update
    corresponding to the [i][j] entry is performed as below:

    ::

       output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0,
       output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1,

    where the ``f`` is ``+``, ``*``, ``max`` or ``min`` as specified.

    This operator is the inverse of GatherElements. It is similar to Torch's
    Scatter operation.

    (Opset 18 change): Adds max/min to the set of allowed reduction ops.

    Example 1:

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
        Type of reduction to apply: none (default), add, mul, max, min. 'none':
        no reduction applied. 'add': reduction using the addition operation.
        'mul': reduction using the multiplication operation.'max': reduction
        using the maximum operation.'min': reduction using the minimum
        operation.

    Returns
    =======
    output : Var
        Type T.
        Tensor of rank r >= 1 (same rank as input).

    Notes
    =====
    Signature: ``ai.onnx@18::ScatterElements``.

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

    The ``output`` is calculated via the following equation:

    ::

       output = np.copy(data)
       update_indices = indices.shape[:-1]
       for idx in np.ndindex(update_indices):
           output[indices[idx]] = updates[idx]

    The order of iteration in the above loop is not specified. In
    particular, indices should not have duplicate entries: that is, if idx1
    != idx2, then indices[idx1] != indices[idx2]. This ensures that the
    output value does not depend on the iteration order.

    ``reduction`` allows specification of an optional reduction operation,
    which is applied to all values in ``updates`` tensor into ``output`` at
    the specified ``indices``. In cases where ``reduction`` is set to
    "none", indices should not have duplicate entries: that is, if idx1 !=
    idx2, then indices[idx1] != indices[idx2]. This ensures that the output
    value does not depend on the iteration order. When ``reduction`` is set
    to some reduction function ``f``, ``output`` is calculated as follows:

    ::

       output = np.copy(data)
       update_indices = indices.shape[:-1]
       for idx in np.ndindex(update_indices):
           output[indices[idx]] = f(output[indices[idx]], updates[idx])

    where the ``f`` is ``+``, ``*``, ``max`` or ``min`` as specified.

    This operator is the inverse of GatherND.

    (Opset 18 change): Adds max/min to the set of allowed reduction ops.

    Example 1:

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
        Type of reduction to apply: none (default), add, mul, max, min. 'none':
        no reduction applied. 'add': reduction using the addition operation.
        'mul': reduction using the addition operation. 'max': reduction using
        the maximum operation.'min': reduction using the minimum operation.

    Returns
    =======
    output : Var
        Type T.
        Tensor of rank r >= 1.

    Notes
    =====
    Signature: ``ai.onnx@18::ScatterND``.

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


def split(
    input: Var,
    split: Optional[Var] = None,
    *,
    axis: int = 0,
    num_outputs: Optional[int] = None,
) -> Sequence[Var]:
    r"""
    Split a tensor into a list of tensors, along the specified 'axis'.
    Either input 'split' or the attribute 'num_outputs' should be specified,
    but not both. If the attribute 'num_outputs' is specified, then the
    tensor is split into equal sized parts. If the tensor is not evenly
    splittable into ``num_outputs``, the last chunk will be smaller. If the
    input 'split' is specified, it indicates the sizes of each output in the
    split.

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
    num_outputs
        Attribute.
        Number of outputs to split parts of the tensor into. If the tensor is
        not evenly splittable the last chunk will be smaller.

    Returns
    =======
    outputs : Sequence[Var]
        Type T.
        One or more outputs forming list of tensors after splitting

    Notes
    =====
    Signature: ``ai.onnx@18::Split``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Split(
        _Split.Attributes(
            axis=AttrInt64(axis, name="axis"),
            num_outputs=AttrInt64.maybe(num_outputs, name="num_outputs"),
        ),
        _Split.Inputs(
            input=input,
            split=split,
        ),
        out_variadic=num_outputs,
    ).outputs.outputs


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
    "GroupNormalization": _GroupNormalization,
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
    "GroupNormalization": group_normalization,
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
