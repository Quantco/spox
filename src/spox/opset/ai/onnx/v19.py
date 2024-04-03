# ruff: noqa: E741 -- Allow ambiguous variable name
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
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
)
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._graph import Graph, subgraph
from spox._node import OpType
from spox._standard import StandardNode
from spox._type_system import Tensor, Type
from spox._value_prop import PropValueType
from spox._var import Var
from spox.opset.ai.onnx.v18 import (
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
    _BatchNormalization,
    _Bernoulli,
    _BitShift,
    _BitwiseAnd,
    _BitwiseNot,
    _BitwiseOr,
    _BitwiseXor,
    _BlackmanWindow,
    _Ceil,
    _Celu,
    _CenterCropPad,
    _Clip,
    _Col2Im,
    _Compress,
    _Concat,
    _ConcatFromSequence,
    _ConstantOfShape,
    _Conv,
    _ConvInteger,
    _ConvTranspose,
    _Cos,
    _Cosh,
    _CumSum,
    _DepthToSpace,
    _Det,
    _Div,
    _Dropout,
    _DynamicQuantizeLinear,
    _Einsum,
    _Elu,
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
    _GroupNormalization,
    _HammingWindow,
    _HannWindow,
    _Hardmax,
    _HardSigmoid,
    _HardSwish,
    _InstanceNormalization,
    _IsInf,
    _IsNaN,
    _LayerNormalization,
    _LeakyRelu,
    _Less,
    _LessOrEqual,
    _Log,
    _LogSoftmax,
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
    _Pow,
    _PRelu,
    _QLinearConv,
    _QLinearMatMul,
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
    _Relu,
    _ReverseSequence,
    _RoiAlign,
    _Round,
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
    _Shrink,
    _Sigmoid,
    _Sign,
    _Sin,
    _Sinh,
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
    batch_normalization,
    bernoulli,
    bit_shift,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    blackman_window,
    ceil,
    celu,
    center_crop_pad,
    clip,
    col2_im,
    compress,
    concat,
    concat_from_sequence,
    constant_of_shape,
    conv,
    conv_integer,
    conv_transpose,
    cos,
    cosh,
    cumsum,
    depth_to_space,
    det,
    dft,
    div,
    dropout,
    dynamic_quantize_linear,
    einsum,
    elu,
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
    group_normalization,
    gru,
    hamming_window,
    hann_window,
    hard_sigmoid,
    hard_swish,
    hardmax,
    instance_normalization,
    isinf,
    isnan,
    layer_normalization,
    leaky_relu,
    less,
    less_or_equal,
    log,
    log_softmax,
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
    pow,
    prelu,
    qlinear_conv,
    qlinear_matmul,
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
    relu,
    reverse_sequence,
    rnn,
    roi_align,
    round,
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
    shrink,
    sigmoid,
    sign,
    sin,
    sinh,
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


class _AveragePool(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        auto_pad: AttrString
        ceil_mode: AttrInt64
        count_include_pad: AttrInt64
        dilations: Optional[AttrInt64s]
        kernel_shape: AttrInt64s
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("AveragePool", "", 19)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Cast(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        saturate: AttrInt64
        to: AttrDtype

    @dataclass
    class Inputs(BaseInputs):
        input: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("Cast", "", 19)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _CastLike(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        saturate: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        input: Var
        target_type: Var

    @dataclass
    class Outputs(BaseOutputs):
        output: Var

    op_type = OpType("CastLike", "", 19)

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

    op_type = OpType("Constant", "", 19)

    attrs: Attributes
    inputs: BaseInputs
    outputs: Outputs


class _DeformConv(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        dilations: Optional[AttrInt64s]
        group: AttrInt64
        kernel_shape: Optional[AttrInt64s]
        offset_group: AttrInt64
        pads: Optional[AttrInt64s]
        strides: Optional[AttrInt64s]

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        W: Var
        offset: Var
        B: Optional[Var]
        mask: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("DeformConv", "", 19)

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

    op_type = OpType("DequantizeLinear", "", 19)

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

    op_type = OpType("Equal", "", 19)

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

    op_type = OpType("Identity", "", 19)

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

    op_type = OpType("If", "", 19)

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

    op_type = OpType("Loop", "", 19)

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

    op_type = OpType("Pad", "", 19)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _QuantizeLinear(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        axis: AttrInt64
        saturate: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        x: Var
        y_scale: Var
        y_zero_point: Optional[Var]

    @dataclass
    class Outputs(BaseOutputs):
        y: Var

    op_type = OpType("QuantizeLinear", "", 19)

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

    op_type = OpType("Reshape", "", 19)

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

    op_type = OpType("Resize", "", 19)

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

    op_type = OpType("Scan", "", 19)

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

    op_type = OpType("Shape", "", 19)

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

    op_type = OpType("Size", "", 19)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def average_pool(
    X: Var,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    count_include_pad: int = 0,
    dilations: Optional[Iterable[int]] = None,
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
    Signature: ``ai.onnx@19::AveragePool``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _AveragePool(
        _AveragePool.Attributes(
            auto_pad=AttrString(auto_pad, name="auto_pad"),
            ceil_mode=AttrInt64(ceil_mode, name="ceil_mode"),
            count_include_pad=AttrInt64(count_include_pad, name="count_include_pad"),
            dilations=AttrInt64s.maybe(dilations, name="dilations"),
            kernel_shape=AttrInt64s(kernel_shape, name="kernel_shape"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _AveragePool.Inputs(
            X=X,
        ),
    ).outputs.Y


def cast(
    input: Var,
    *,
    saturate: int = 1,
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
    rules if the destination type is not a float 8 type.

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

    Float 8 type were introduced to speed up the training of deep models. By
    default the conversion of a float *x* obeys to the following rules.
    ``[x]`` means the value rounded to the target mantissa width.

    ============== =========== ======== ======== ========
    x              E4M3FN      E4M3FNUZ E5M2     E5M2FNUZ
    ============== =========== ======== ======== ========
    0              0           0        0        0
    -0             -0          0        -0       0
    NaN            NaN         NaN      NaN      NaN
    +/- Inf        +/- FLT_MAX NaN      FLT_MAX  NaN
    [x] > FLT_MAX  FLT_MAX     FLT_MAX  FLT_MAX  FLT_MAX
    [x] < -FLT_MAX -FLT_MAX    -FLT_MAX -FLT_MAX -FLT_MAX
    else           RNE         RNE      RNE      RNE
    ============== =========== ======== ======== ========

    The behavior changes if the parameter 'saturate' is set to False. The
    rules then become:

    ============== ====== ======== ======= ========
    x              E4M3FN E4M3FNUZ E5M2    E5M2FNUZ
    ============== ====== ======== ======= ========
    0              0      0        0       0
    -0             -0     0        -0      0
    NaN            NaN    NaN      NaN     NaN
    +/- Inf        NaN    NaN      +/- Inf NaN
    [x] > FLT_MAX  NaN    NaN      Inf     NaN
    [x] < -FLT_MAX NaN    NaN      -Inf    NaN
    else           RNE    RNE      RNE     RNE
    ============== ====== ======== ======= ========

    Parameters
    ==========
    input
        Type T1.
        Input tensor to be cast.
    saturate
        Attribute.
        The parameter defines how the conversion behaves if an input value is
        out of range of the destination type. It only applies for float 8
        conversion (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz).
        It is true by default. All cases are fully described in two tables
        inserted in the operator description.
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
    Signature: ``ai.onnx@19::Cast``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _Cast(
        _Cast.Attributes(
            saturate=AttrInt64(saturate, name="saturate"),
            to=AttrDtype(to, name="to"),
        ),
        _Cast.Inputs(
            input=input,
        ),
    ).outputs.output


def cast_like(
    input: Var,
    target_type: Var,
    *,
    saturate: int = 1,
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
    saturate
        Attribute.
        The parameter defines how the conversion behaves if an input value is
        out of range of the destination type. It only applies for float 8
        conversion (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz).
        It is true by default. Please refer to operator Cast description for
        further details.

    Returns
    =======
    output : Var
        Type T2.
        Output tensor produced by casting the first input tensor to have the
        same type as the second input tensor.

    Notes
    =====
    Signature: ``ai.onnx@19::CastLike``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
    """
    return _CastLike(
        _CastLike.Attributes(
            saturate=AttrInt64(saturate, name="saturate"),
        ),
        _CastLike.Inputs(
            input=input,
            target_type=target_type,
        ),
    ).outputs.output


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
    Signature: ``ai.onnx@19::Constant``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
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


def deform_conv(
    X: Var,
    W: Var,
    offset: Var,
    B: Optional[Var] = None,
    mask: Optional[Var] = None,
    *,
    dilations: Optional[Iterable[int]] = None,
    group: int = 1,
    kernel_shape: Optional[Iterable[int]] = None,
    offset_group: int = 1,
    pads: Optional[Iterable[int]] = None,
    strides: Optional[Iterable[int]] = None,
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
    Signature: ``ai.onnx@19::DeformConv``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _DeformConv(
        _DeformConv.Attributes(
            dilations=AttrInt64s.maybe(dilations, name="dilations"),
            group=AttrInt64(group, name="group"),
            kernel_shape=AttrInt64s.maybe(kernel_shape, name="kernel_shape"),
            offset_group=AttrInt64(offset_group, name="offset_group"),
            pads=AttrInt64s.maybe(pads, name="pads"),
            strides=AttrInt64s.maybe(strides, name="strides"),
        ),
        _DeformConv.Inputs(
            X=X,
            W=W,
            offset=offset,
            B=B,
            mask=mask,
        ),
    ).outputs.Y


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
    there's no zero point (zero point is supposed to be 0). ``zero-point``
    is usually not used in the case of float8e4m3fn, float8e4m3fnuz,
    float8e5m2, float8e5m2fnuz quantization, but the dequantization formula
    remains the same for consistency and 'x_scale' still determines the
    output type.

    Parameters
    ==========
    x
        Type T1.
        N-D quantized input tensor to be de-quantized.
    x_scale
        Type T2.
        Scale for input 'x'. It can be a scalar, which means a per-tensor/layer
        dequantization, or a 1-D tensor for per-axis dequantization.
    x_zero_point
        Type T1.
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
        Type T2.
        N-D full precision output tensor. It has same shape as input 'x'.

    Notes
    =====
    Signature: ``ai.onnx@19::DequantizeLinear``.

    Type constraints:
     - T1: `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int32)`, `tensor(int8)`, `tensor(uint8)`
     - T2: `tensor(bfloat16)`, `tensor(float)`, `tensor(float16)`
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
    Signature: ``ai.onnx@19::Equal``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(bool)`
    """
    return _Equal(
        _Equal.Attributes(),
        _Equal.Inputs(
            A=A,
            B=B,
        ),
    ).outputs.C


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
    Signature: ``ai.onnx@19::Identity``.

    Type constraints:
     - V: `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`, `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
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
    Signature: ``ai.onnx@19::If``.

    Type constraints:
     - B: `tensor(bool)`
     - V: `optional(seq(tensor(bfloat16)))`, `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bfloat16))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(float8e4m3fn))`, `optional(tensor(float8e4m3fnuz))`, `optional(tensor(float8e5m2))`, `optional(tensor(float8e5m2fnuz))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`, `seq(tensor(bfloat16))`, `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(float8e4m3fn))`, `seq(tensor(float8e4m3fnuz))`, `seq(tensor(float8e5m2))`, `seq(tensor(float8e5m2fnuz))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
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
    Signature: ``ai.onnx@19::Loop``.

    Type constraints:
     - I: `tensor(int64)`
     - B: `tensor(bool)`
     - V: `optional(seq(tensor(bfloat16)))`, `optional(seq(tensor(bool)))`, `optional(seq(tensor(complex128)))`, `optional(seq(tensor(complex64)))`, `optional(seq(tensor(double)))`, `optional(seq(tensor(float)))`, `optional(seq(tensor(float16)))`, `optional(seq(tensor(int16)))`, `optional(seq(tensor(int32)))`, `optional(seq(tensor(int64)))`, `optional(seq(tensor(int8)))`, `optional(seq(tensor(string)))`, `optional(seq(tensor(uint16)))`, `optional(seq(tensor(uint32)))`, `optional(seq(tensor(uint64)))`, `optional(seq(tensor(uint8)))`, `optional(tensor(bfloat16))`, `optional(tensor(bool))`, `optional(tensor(complex128))`, `optional(tensor(complex64))`, `optional(tensor(double))`, `optional(tensor(float))`, `optional(tensor(float16))`, `optional(tensor(float8e4m3fn))`, `optional(tensor(float8e4m3fnuz))`, `optional(tensor(float8e5m2))`, `optional(tensor(float8e5m2fnuz))`, `optional(tensor(int16))`, `optional(tensor(int32))`, `optional(tensor(int64))`, `optional(tensor(int8))`, `optional(tensor(string))`, `optional(tensor(uint16))`, `optional(tensor(uint32))`, `optional(tensor(uint64))`, `optional(tensor(uint8))`, `seq(tensor(bfloat16))`, `seq(tensor(bool))`, `seq(tensor(complex128))`, `seq(tensor(complex64))`, `seq(tensor(double))`, `seq(tensor(float))`, `seq(tensor(float16))`, `seq(tensor(float8e4m3fn))`, `seq(tensor(float8e4m3fnuz))`, `seq(tensor(float8e5m2))`, `seq(tensor(float8e5m2fnuz))`, `seq(tensor(int16))`, `seq(tensor(int32))`, `seq(tensor(int64))`, `seq(tensor(int8))`, `seq(tensor(string))`, `seq(tensor(uint16))`, `seq(tensor(uint32))`, `seq(tensor(uint64))`, `seq(tensor(uint8))`, `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
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

    4) ``wrap`` - wrap-around padding as if the data tensor forms a torus

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

    Example 4 (``wrap`` mode):

    ::

       data = [
           [1.0, 1.2],
           [2.3, 3.4],
           [4.5, 5.7],
       ]

       pads = [2, 1, 1, 1]

       mode = 'wrap'

       output = [
           [3.4, 2.3, 3.4, 2.3],
           [5.7, 4.5, 5.7, 4.5],
           [1.2, 1.0, 1.2, 1.0],
           [3.4, 2.3, 3.4, 2.3],
           [5.7, 4.5, 5.7, 4.5],
           [1.2, 1.0, 1.2, 1.0],
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
        Supported modes: ``constant``\ (default), ``reflect``, ``edge``,
        ``wrap``

    Returns
    =======
    output : Var
        Type T.
        Tensor after padding.

    Notes
    =====
    Signature: ``ai.onnx@19::Pad``.

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


def quantize_linear(
    x: Var,
    y_scale: Var,
    y_zero_point: Optional[Var] = None,
    *,
    axis: int = 1,
    saturate: int = 1,
) -> Var:
    r"""
    The linear quantization operator. It consumes a high precision tensor, a
    scale, and a zero point to compute the low precision / quantized tensor.
    The scale factor and zero point must have same shape, and can be either
    a scalar for per-tensor / per layer quantization, or a 1-D tensor for
    per-axis quantization. The quantization formula is
    ``y = saturate ((x / y_scale) + y_zero_point)``. For saturation, it
    saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8. For (x
    / y_scale), it's rounding to the nearest even. Refer to
    https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and
    'y' must have same type. 'y_zero_point' is usually not used for
    quantization to float8e4m3fn, float8e4m3fnuz, float8e5m2,
    float8e5m2fnuz, but the quantization formula remains the same for
    consistency and the type of the attribute 'y_zero_point' still
    determines the quantization type.

    Parameters
    ==========
    x
        Type T1.
        N-D full precision Input tensor to be quantized.
    y_scale
        Type T1.
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
    saturate
        Attribute.
        The parameter defines how the conversion behaves if an input value is
        out of range of the destination type. It only applies for float 8
        quantization (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz).
        It is true by default. All cases are fully described in two tables
        inserted in the operator description.

    Returns
    =======
    y : Var
        Type T2.
        N-D quantized output tensor. It has same shape as input 'x'.

    Notes
    =====
    Signature: ``ai.onnx@19::QuantizeLinear``.

    Type constraints:
     - T1: `tensor(bfloat16)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`
     - T2: `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int8)`, `tensor(uint8)`
    """
    return _QuantizeLinear(
        _QuantizeLinear.Attributes(
            axis=AttrInt64(axis, name="axis"),
            saturate=AttrInt64(saturate, name="saturate"),
        ),
        _QuantizeLinear.Inputs(
            x=x,
            y_scale=y_scale,
            y_zero_point=y_zero_point,
        ),
    ).outputs.y


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
    Signature: ``ai.onnx@19::Reshape``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
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

    ::

       output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)

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
        describe a case using axis x as an example. Denote ``x_resized`` as the
        coordinate of axis x in the resized tensor, ``x_original`` as the
        coordinate of axis x in the original tensor, ``length_original`` as the
        length of the original tensor in axis x, ``length_resized`` as the
        length of the resized tensor in axis x,
        ``scale = length_resized / length_original``, ``output_width`` the
        target length on the axis x which can be a fractional number when it is
        calculated out of a scale factor, and ``output_width_int`` the effective
        output width as an integer.

        if coordinate_transformation_mode is ``"half_pixel"``,

        ::

           x_original = (x_resized + 0.5) / scale - 0.5

        if coordinate_transformation_mode is ``"half_pixel_symmetric"``,

        ::

           adjustment = output_width_int / output_width
           center = input_width / 2
           offset = center * (1 - adjustment)
           x_ori = offset + (x + 0.5) / scale - 0.5

        if coordinate_transformation_mode is ``"pytorch_half_pixel"``,

        ::

           x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0

        if coordinate_transformation_mode is ``"align_corners"``,

        ::

           x_original = x_resized * (length_original - 1) / (length_resized - 1)

        if coordinate_transformation_mode is ``"asymmetric"``,

        ::

           x_original = x_resized / scale

        if coordinate_transformation_mode is ``"tf_crop_and_resize"``,

        ::

           x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1)

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

        ::

           scale = Min(sizes[i] / in_size[d])
           out_size[d] = round_int(scale * in_size[i])

        If ``keep_aspect_ratio_policy`` is ``"not_smaller"``, the sizes are
        adjusted so that no extent of the output is smaller than the specified
        size, while keeping the original aspect ratio:

        ::

           scale = Max(sizes[i] / in_size[d])
           out_size[d] = round_int(scale * in_size[i])

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
    Signature: ``ai.onnx@19::Resize``.

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
    Signature: ``ai.onnx@19::Scan``.

    Type constraints:
     - V: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
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
    Signature: ``ai.onnx@19::Shape``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
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
    Signature: ``ai.onnx@19::Size``.

    Type constraints:
     - T: `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(float8e4m3fn)`, `tensor(float8e4m3fnuz)`, `tensor(float8e5m2)`, `tensor(float8e5m2fnuz)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)`
     - T1: `tensor(int64)`
    """
    return _Size(
        _Size.Attributes(),
        _Size.Inputs(
            data=data,
        ),
    ).outputs.size


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
