# flake8: noqa
import typing  # noqa: F401
from typing import (  # noqa: F401
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from typing import cast as typing_cast  # noqa: F401

import numpy  # noqa: F401
from numpy import ndarray  # noqa: F401

from steelix.arrow import Arrow, _nil, result_type  # noqa: F401
from steelix.arrowfields import ArrowFields, NoArrows  # noqa: F401
from steelix.attr import Attr  # noqa: F401
from steelix.attrfields import AttrFields, NoAttrs  # noqa: F401
from steelix.fields import of  # noqa: F401
from steelix.graph import Graph, subgraph  # noqa: F401
from steelix.internal_op import intro  # noqa: F401
from steelix.node import OpType  # noqa: F401
from steelix.standard import StandardNode  # noqa: F401
from steelix.type_system import Tensor, Type, type_match  # noqa: F401


class _ArrayFeatureExtractor(StandardNode):
    Attributes = NoAttrs

    class Inputs(ArrowFields):
        X: Arrow
        Y: Arrow

    class Outputs(ArrowFields):
        Z: Arrow

    op_type = OpType("ArrayFeatureExtractor", "ai.onnx.ml", 1)

    attrs: NoAttrs
    inputs: Inputs
    outputs: Outputs


class _Binarizer(StandardNode):
    class Attributes(AttrFields):
        threshold: Attr[float]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("Binarizer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _CastMap(StandardNode):
    class Attributes(AttrFields):
        cast_to: Attr[str]
        map_form: Attr[str]
        max_map: Attr[int]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("CastMap", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _CategoryMapper(StandardNode):
    class Attributes(AttrFields):
        cats_int64s: Attr[Sequence[int]]
        cats_strings: Attr[Sequence[str]]
        default_int64: Attr[int]
        default_string: Attr[str]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("CategoryMapper", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _DictVectorizer(StandardNode):
    class Attributes(AttrFields):
        int64_vocabulary: Attr[Sequence[int]]
        string_vocabulary: Attr[Sequence[str]]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("DictVectorizer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _FeatureVectorizer(StandardNode):
    class Attributes(AttrFields):
        inputdimensions: Attr[Sequence[int]]

    class Inputs(ArrowFields):
        X: Sequence[Arrow]

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("FeatureVectorizer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Imputer(StandardNode):
    class Attributes(AttrFields):
        imputed_value_floats: Attr[Sequence[float]]
        imputed_value_int64s: Attr[Sequence[int]]
        replaced_value_float: Attr[float]
        replaced_value_int64: Attr[int]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("Imputer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LabelEncoder(StandardNode):
    class Attributes(AttrFields):
        default_float: Attr[float]
        default_int64: Attr[int]
        default_string: Attr[str]
        keys_floats: Attr[Sequence[float]]
        keys_int64s: Attr[Sequence[int]]
        keys_strings: Attr[Sequence[str]]
        values_floats: Attr[Sequence[float]]
        values_int64s: Attr[Sequence[int]]
        values_strings: Attr[Sequence[str]]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("LabelEncoder", "ai.onnx.ml", 2)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LinearClassifier(StandardNode):
    class Attributes(AttrFields):
        classlabels_ints: Attr[Sequence[int]]
        classlabels_strings: Attr[Sequence[str]]
        coefficients: Attr[Sequence[float]]
        intercepts: Attr[Sequence[float]]
        multi_class: Attr[int]
        post_transform: Attr[str]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow
        Z: Arrow

    op_type = OpType("LinearClassifier", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LinearRegressor(StandardNode):
    class Attributes(AttrFields):
        coefficients: Attr[Sequence[float]]
        intercepts: Attr[Sequence[float]]
        post_transform: Attr[str]
        targets: Attr[int]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("LinearRegressor", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Normalizer(StandardNode):
    class Attributes(AttrFields):
        norm: Attr[str]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("Normalizer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _OneHotEncoder(StandardNode):
    class Attributes(AttrFields):
        cats_int64s: Attr[Sequence[int]]
        cats_strings: Attr[Sequence[str]]
        zeros: Attr[int]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        try:
            n_encodings = len(self.attrs.cats_int64s._value)
        except TypeError:
            n_encodings = len(self.attrs.cats_strings._value)
        shape = (*self.inputs.X.unwrap_tensor().shape.to_simple(), n_encodings)  # type: ignore
        return {"Y": Tensor(elem_type=numpy.float32, shape=shape)}

    op_type = OpType("OneHotEncoder", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SVMClassifier(StandardNode):
    class Attributes(AttrFields):
        classlabels_ints: Attr[Sequence[int]]
        classlabels_strings: Attr[Sequence[str]]
        coefficients: Attr[Sequence[float]]
        kernel_params: Attr[Sequence[float]]
        kernel_type: Attr[str]
        post_transform: Attr[str]
        prob_a: Attr[Sequence[float]]
        prob_b: Attr[Sequence[float]]
        rho: Attr[Sequence[float]]
        support_vectors: Attr[Sequence[float]]
        vectors_per_class: Attr[Sequence[int]]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow
        Z: Arrow

    op_type = OpType("SVMClassifier", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SVMRegressor(StandardNode):
    class Attributes(AttrFields):
        coefficients: Attr[Sequence[float]]
        kernel_params: Attr[Sequence[float]]
        kernel_type: Attr[str]
        n_supports: Attr[int]
        one_class: Attr[int]
        post_transform: Attr[str]
        rho: Attr[Sequence[float]]
        support_vectors: Attr[Sequence[float]]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("SVMRegressor", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Scaler(StandardNode):
    class Attributes(AttrFields):
        offset: Attr[Sequence[float]]
        scale: Attr[Sequence[float]]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("Scaler", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _TreeEnsembleClassifier(StandardNode):
    class Attributes(AttrFields):
        base_values: Attr[Sequence[float]]
        base_values_as_tensor: Attr[ndarray]
        class_ids: Attr[Sequence[int]]
        class_nodeids: Attr[Sequence[int]]
        class_treeids: Attr[Sequence[int]]
        class_weights: Attr[Sequence[float]]
        class_weights_as_tensor: Attr[ndarray]
        classlabels_int64s: Attr[Sequence[int]]
        classlabels_strings: Attr[Sequence[str]]
        nodes_falsenodeids: Attr[Sequence[int]]
        nodes_featureids: Attr[Sequence[int]]
        nodes_hitrates: Attr[Sequence[float]]
        nodes_hitrates_as_tensor: Attr[ndarray]
        nodes_missing_value_tracks_true: Attr[Sequence[int]]
        nodes_modes: Attr[Sequence[str]]
        nodes_nodeids: Attr[Sequence[int]]
        nodes_treeids: Attr[Sequence[int]]
        nodes_truenodeids: Attr[Sequence[int]]
        nodes_values: Attr[Sequence[float]]
        nodes_values_as_tensor: Attr[ndarray]
        post_transform: Attr[str]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow
        Z: Arrow

    op_type = OpType("TreeEnsembleClassifier", "ai.onnx.ml", 3)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _TreeEnsembleRegressor(StandardNode):
    class Attributes(AttrFields):
        aggregate_function: Attr[str]
        base_values: Attr[Sequence[float]]
        base_values_as_tensor: Attr[ndarray]
        n_targets: Attr[int]
        nodes_falsenodeids: Attr[Sequence[int]]
        nodes_featureids: Attr[Sequence[int]]
        nodes_hitrates: Attr[Sequence[float]]
        nodes_hitrates_as_tensor: Attr[ndarray]
        nodes_missing_value_tracks_true: Attr[Sequence[int]]
        nodes_modes: Attr[Sequence[str]]
        nodes_nodeids: Attr[Sequence[int]]
        nodes_treeids: Attr[Sequence[int]]
        nodes_truenodeids: Attr[Sequence[int]]
        nodes_values: Attr[Sequence[float]]
        nodes_values_as_tensor: Attr[ndarray]
        post_transform: Attr[str]
        target_ids: Attr[Sequence[int]]
        target_nodeids: Attr[Sequence[int]]
        target_treeids: Attr[Sequence[int]]
        target_weights: Attr[Sequence[float]]
        target_weights_as_tensor: Attr[ndarray]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("TreeEnsembleRegressor", "ai.onnx.ml", 3)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ZipMap(StandardNode):
    class Attributes(AttrFields):
        classlabels_int64s: Attr[Sequence[int]]
        classlabels_strings: Attr[Sequence[str]]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Z: Arrow

    op_type = OpType("ZipMap", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def array_feature_extractor(
    X: Arrow,
    Y: Arrow,
) -> Arrow:
    r"""
    Select elements of the input tensor based on the indices passed.


        The indices are applied to the last axes of the tensor.

    Parameters
    ==========
    X
        Type T.
        Data to be selected
    Y
        Type tensor(int64).
        The indices, based on 0 as the first index of any dimension.

    Returns
    =======
    Z : Arrow
        Type T.
        Selected output data as an array

    Notes
    =====
    Signature: ``ai.onnx.ml@1::ArrayFeatureExtractor``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`, `tensor(string)`
    """
    return _ArrayFeatureExtractor(
        _ArrayFeatureExtractor.Attributes(),
        _ArrayFeatureExtractor.Inputs(
            X=X,
            Y=Y,
        ),
    ).outputs.Z


def binarizer(
    X: Arrow,
    *,
    threshold: float = 0.0,
) -> Arrow:
    r"""
    Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.

    Parameters
    ==========
    X
        Type T.
        Data to be binarized
    threshold
        Attribute.
        Values greater than this are mapped to 1, others to 0.

    Returns
    =======
    Y : Arrow
        Type T.
        Binarized output data

    Notes
    =====
    Signature: ``ai.onnx.ml@1::Binarizer``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _Binarizer(
        _Binarizer.Attributes(
            threshold=threshold,
        ),
        _Binarizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def cast_map(
    X: Arrow,
    *,
    cast_to: str = "TO_FLOAT",
    map_form: str = "DENSE",
    max_map: int = 1,
) -> Arrow:
    r"""
    Converts a map to a tensor.

    The map key must be an int64 and the values will be ordered
        in ascending order based on this key.

    The operator supports dense packing or sparse packing.
        If using sparse packing, the key cannot exceed the max_map-1 value.

    Parameters
    ==========
    X
        Type T1.
        The input map that is to be cast to a tensor
    cast_to
        Attribute.
        A string indicating the desired element type of the output tensor, one of 'TO_FLOAT', 'TO_STRING', 'TO_INT64'.
    map_form
        Attribute.
        Indicates whether to only output as many values as are in the input (dense), or position the input based on using the key of the map as the index of the output (sparse).

        One of 'DENSE', 'SPARSE'.
    max_map
        Attribute.
        If the value of map_form is 'SPARSE,' this attribute indicates the total length of the output tensor.

    Returns
    =======
    Y : Arrow
        Type T2.
        A tensor representing the same data as the input map, ordered by their keys

    Notes
    =====
    Signature: ``ai.onnx.ml@1::CastMap``.

    Type constraints:
     - T1: `map(int64,tensor(float))`, `map(int64,tensor(string))`
     - T2: `tensor(float)`, `tensor(int64)`, `tensor(string)`
    """
    return _CastMap(
        _CastMap.Attributes(
            cast_to=cast_to,
            map_form=map_form,
            max_map=max_map,
        ),
        _CastMap.Inputs(
            X=X,
        ),
    ).outputs.Y


def category_mapper(
    X: Arrow,
    *,
    cats_int64s: Optional[Iterable[int]] = None,
    cats_strings: Optional[Iterable[str]] = None,
    default_int64: int = -1,
    default_string: str = "_Unused",
) -> Arrow:
    r"""
    Converts strings to integers and vice versa.


        Two sequences of equal length are used to map between integers and strings,
        with strings and integers at the same index detailing the mapping.


        Each operator converts either integers to strings or strings to integers, depending
        on which default value attribute is provided. Only one default value attribute
        should be defined.


        If the string default value is set, it will convert integers to strings.
        If the int default value is set, it will convert strings to integers.

    Parameters
    ==========
    X
        Type T1.
        Input data
    cats_int64s
        Attribute.
        The integers of the map. This sequence must be the same length as the 'cats_strings' sequence.
    cats_strings
        Attribute.
        The strings of the map. This sequence must be the same length as the 'cats_int64s' sequence
    default_int64
        Attribute.
        An integer to use when an input string value is not found in the map.

        One and only one of the 'default_*' attributes must be defined.
    default_string
        Attribute.
        A string to use when an input integer value is not found in the map.

        One and only one of the 'default_*' attributes must be defined.

    Returns
    =======
    Y : Arrow
        Type T2.
        Output data. If strings are input, the output values are integers, and vice versa.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::CategoryMapper``.

    Type constraints:
     - T1: `tensor(int64)`, `tensor(string)`
     - T2: `tensor(int64)`, `tensor(string)`
    """
    return _CategoryMapper(
        _CategoryMapper.Attributes(
            cats_int64s=cats_int64s,
            cats_strings=cats_strings,
            default_int64=default_int64,
            default_string=default_string,
        ),
        _CategoryMapper.Inputs(
            X=X,
        ),
    ).outputs.Y


def dict_vectorizer(
    X: Arrow,
    *,
    int64_vocabulary: Optional[Iterable[int]] = None,
    string_vocabulary: Optional[Iterable[str]] = None,
) -> Arrow:
    r"""
    Uses an index mapping to convert a dictionary to an array.


        Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
        the key type. The index into the vocabulary array at which the key is found is then
        used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.


        The key type of the input map must correspond to the element type of the defined vocabulary attribute.
        Therefore, the output array will be equal in length to the index mapping vector parameter.
        All keys in the input dictionary must be present in the index mapping vector.
        For each item in the input dictionary, insert its value in the output array.
        Any keys not present in the input dictionary, will be zero in the output array.


        For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
        then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.

    Parameters
    ==========
    X
        Type T1.
        A dictionary.
    int64_vocabulary
        Attribute.
        An integer vocabulary array.

        One and only one of the vocabularies must be defined.
    string_vocabulary
        Attribute.
        A string vocabulary array.

        One and only one of the vocabularies must be defined.

    Returns
    =======
    Y : Arrow
        Type T2.
        A 1-D tensor holding values from the input dictionary.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::DictVectorizer``.

    Type constraints:
     - T1: `map(int64,tensor(double))`, `map(int64,tensor(float))`, `map(int64,tensor(string))`, `map(string,tensor(double))`, `map(string,tensor(float))`, `map(string,tensor(int64))`
     - T2: `tensor(double)`, `tensor(float)`, `tensor(int64)`, `tensor(string)`
    """
    return _DictVectorizer(
        _DictVectorizer.Attributes(
            int64_vocabulary=int64_vocabulary,
            string_vocabulary=string_vocabulary,
        ),
        _DictVectorizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def feature_vectorizer(
    X: Sequence[Arrow],
    *,
    inputdimensions: Optional[Iterable[int]] = None,
) -> Arrow:
    r"""
    Concatenates input tensors into one continuous output.


        All input shapes are 2-D and are concatenated along the second dimention. 1-D tensors are treated as [1,C].
        Inputs are copied to the output maintaining the order of the input arguments.


        All inputs must be integers or floats, while the output will be all floating point values.

    Parameters
    ==========
    X
        Type T1.
        An ordered collection of tensors, all with the same element type.
    inputdimensions
        Attribute.
        The size of each input in the input list

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        The output array, elements ordered as the inputs.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::FeatureVectorizer``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _FeatureVectorizer(
        _FeatureVectorizer.Attributes(
            inputdimensions=inputdimensions,
        ),
        _FeatureVectorizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def imputer(
    X: Arrow,
    *,
    imputed_value_floats: Optional[Iterable[float]] = None,
    imputed_value_int64s: Optional[Iterable[int]] = None,
    replaced_value_float: float = 0.0,
    replaced_value_int64: int = 0,
) -> Arrow:
    r"""
    Replaces inputs that equal one value with another, leaving all other elements alone.


        This operator is typically used to replace missing values in situations where they have a canonical
        representation, such as -1, 0, NaN, or some extreme value.


        One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
        holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
        width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
        which one depends on whether floats or integers are being processed.


        The imputed_value attribute length can be 1 element, or it can have one element per input feature.

    In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.

    Parameters
    ==========
    X
        Type T.
        Data to be processed.
    imputed_value_floats
        Attribute.
        Value(s) to change to
    imputed_value_int64s
        Attribute.
        Value(s) to change to.
    replaced_value_float
        Attribute.
        A value that needs replacing.
    replaced_value_int64
        Attribute.
        A value that needs replacing.

    Returns
    =======
    Y : Arrow
        Type T.
        Imputed output data

    Notes
    =====
    Signature: ``ai.onnx.ml@1::Imputer``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _Imputer(
        _Imputer.Attributes(
            imputed_value_floats=imputed_value_floats,
            imputed_value_int64s=imputed_value_int64s,
            replaced_value_float=replaced_value_float,
            replaced_value_int64=replaced_value_int64,
        ),
        _Imputer.Inputs(
            X=X,
        ),
    ).outputs.Y


def label_encoder(
    X: Arrow,
    *,
    default_float: float = -0.0,
    default_int64: int = -1,
    default_string: str = "_Unused",
    keys_floats: Optional[Iterable[float]] = None,
    keys_int64s: Optional[Iterable[int]] = None,
    keys_strings: Optional[Iterable[str]] = None,
    values_floats: Optional[Iterable[float]] = None,
    values_int64s: Optional[Iterable[int]] = None,
    values_strings: Optional[Iterable[str]] = None,
) -> Arrow:
    r"""
    Maps each element in the input tensor to another value.


        The mapping is determined by the two parallel attributes, 'keys_*' and
        'values_*' attribute. The i-th value in the specified 'keys_*' attribute
        would be mapped to the i-th value in the specified 'values_*' attribute. It
        implies that input's element type and the element type of the specified
        'keys_*' should be identical while the output type is identical to the
        specified 'values_*' attribute. If an input element can not be found in the
        specified 'keys_*' attribute, the 'default_*' that matches the specified
        'values_*' attribute may be used as its output value.


        Let's consider an example which maps a string tensor to an integer tensor.
        Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
        and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
        "Sally"] would be mapped to [-1, 5, 5, 6, 6].


        Since this operator is an one-to-one mapping, its input and output shapes
        are the same. Notice that only one of 'keys_*'/'values_*' can be set.


        For key look-up, bit-wise comparison is used so even a float NaN can be
        mapped to a value in 'values_*' attribute.



    Parameters
    ==========
    X
        Type T1.
        Input data. It can be either tensor or scalar.
    default_float
        Attribute.
        A float.
    default_int64
        Attribute.
        An integer.
    default_string
        Attribute.
        A string.
    keys_floats
        Attribute.
        A list of floats.
    keys_int64s
        Attribute.
        A list of ints.
    keys_strings
        Attribute.
        A list of strings. One and only one of 'keys_*'s should be set.
    values_floats
        Attribute.
        A list of floats.
    values_int64s
        Attribute.
        A list of ints.
    values_strings
        Attribute.
        A list of strings. One and only one of 'value_*'s should be set.

    Returns
    =======
    Y : Arrow
        Type T2.
        Output data.

    Notes
    =====
    Signature: ``ai.onnx.ml@2::LabelEncoder``.

    Type constraints:
     - T1: `tensor(float)`, `tensor(int64)`, `tensor(string)`
     - T2: `tensor(float)`, `tensor(int64)`, `tensor(string)`
    """
    return _LabelEncoder(
        _LabelEncoder.Attributes(
            default_float=default_float,
            default_int64=default_int64,
            default_string=default_string,
            keys_floats=keys_floats,
            keys_int64s=keys_int64s,
            keys_strings=keys_strings,
            values_floats=values_floats,
            values_int64s=values_int64s,
            values_strings=values_strings,
        ),
        _LabelEncoder.Inputs(
            X=X,
        ),
    ).outputs.Y


def linear_classifier(
    X: Arrow,
    *,
    classlabels_ints: Optional[Iterable[int]] = None,
    classlabels_strings: Optional[Iterable[str]] = None,
    coefficients: Iterable[float],
    intercepts: Optional[Iterable[float]] = None,
    multi_class: int = 0,
    post_transform: str = "NONE",
) -> _LinearClassifier.Outputs:
    r"""
    Linear classifier

    Parameters
    ==========
    X
        Type T1.
        Data to be classified.
    classlabels_ints
        Attribute.
        Class labels when using integer labels. One and only one 'classlabels' attribute must be defined.
    classlabels_strings
        Attribute.
        Class labels when using string labels. One and only one 'classlabels' attribute must be defined.
    coefficients
        Attribute.
        A collection of weights of the model(s).
    intercepts
        Attribute.
        A collection of intercepts.
    multi_class
        Attribute.
        Indicates whether to do OvR or multinomial (0=OvR is the default).
    post_transform
        Attribute.
        Indicates the transform to apply to the scores vector.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'

    Returns
    =======
    Y : Arrow
        Type T2.
        Classification outputs (one class per example).
    Z : Arrow
        Type tensor(float).
        Classification scores ([N,E] - one score for each class and example

    Notes
    =====
    Signature: ``ai.onnx.ml@1::LinearClassifier``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(int64)`, `tensor(string)`
    """
    return _LinearClassifier(
        _LinearClassifier.Attributes(
            classlabels_ints=classlabels_ints,
            classlabels_strings=classlabels_strings,
            coefficients=coefficients,
            intercepts=intercepts,
            multi_class=multi_class,
            post_transform=post_transform,
        ),
        _LinearClassifier.Inputs(
            X=X,
        ),
    ).outputs


def linear_regressor(
    X: Arrow,
    *,
    coefficients: Optional[Iterable[float]] = None,
    intercepts: Optional[Iterable[float]] = None,
    post_transform: str = "NONE",
    targets: int = 1,
) -> Arrow:
    r"""
    Generalized linear regression evaluation.


        If targets is set to 1 (default) then univariate regression is performed.


        If targets is set to M then M sets of coefficients must be passed in as a sequence
        and M results will be output for each input n in N.


        The coefficients array is of length n, and the coefficients for each target are contiguous.
        Intercepts are optional but if provided must match the number of targets.

    Parameters
    ==========
    X
        Type T.
        Data to be regressed.
    coefficients
        Attribute.
        Weights of the model(s).
    intercepts
        Attribute.
        Weights of the intercepts, if used.
    post_transform
        Attribute.
        Indicates the transform to apply to the regression output vector.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
    targets
        Attribute.
        The total number of regression targets, 1 if not defined.

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Regression outputs (one per target, per example).

    Notes
    =====
    Signature: ``ai.onnx.ml@1::LinearRegressor``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _LinearRegressor(
        _LinearRegressor.Attributes(
            coefficients=coefficients,
            intercepts=intercepts,
            post_transform=post_transform,
            targets=targets,
        ),
        _LinearRegressor.Inputs(
            X=X,
        ),
    ).outputs.Y


def normalizer(
    X: Arrow,
    *,
    norm: str = "MAX",
) -> Arrow:
    r"""
    Normalize the input.  There are three normalization modes, which have the corresponding formulas,
        defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':





        Max: Y = X / max(X)


        L1:  Y = X / sum(X)


        L2:  Y = sqrt(X^2 / sum(X^2)}


        In all modes, if the divisor is zero, Y == X.



        For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
        of the batch is normalized independently.

    Parameters
    ==========
    X
        Type T.
        Data to be encoded, a tensor of shape [N,C] or [C]
    norm
        Attribute.
        One of 'MAX,' 'L1,' 'L2'

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Encoded output data

    Notes
    =====
    Signature: ``ai.onnx.ml@1::Normalizer``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _Normalizer(
        _Normalizer.Attributes(
            norm=norm,
        ),
        _Normalizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def one_hot_encoder(
    X: Arrow,
    *,
    cats_int64s: Optional[Iterable[int]] = None,
    cats_strings: Optional[Iterable[str]] = None,
    zeros: int = 1,
) -> Arrow:
    r"""
    Replace each input element with an array of ones and zeros, where a single
        one is placed at the index of the category that was passed in. The total category count
        will determine the size of the extra dimension of the output array Y.


        For example, if we pass a tensor with a single value of 4, and a category count of 8,
        the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.


        This operator assumes every input feature is from the same set of categories.


        If the input is a tensor of float, int32, or double, the data will be cast
        to integers and the cats_int64s category list will be used for the lookups.

    Parameters
    ==========
    X
        Type T.
        Data to be encoded.
    cats_int64s
        Attribute.
        List of categories, ints.

        One and only one of the 'cats_*' attributes must be defined.
    cats_strings
        Attribute.
        List of categories, strings.

        One and only one of the 'cats_*' attributes must be defined.
    zeros
        Attribute.
        If true and category is not present, will return all zeros; if false and a category if not found, the operator will fail.

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Encoded output data, having one more dimension than X.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::OneHotEncoder``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`, `tensor(string)`
    """
    return _OneHotEncoder(
        _OneHotEncoder.Attributes(
            cats_int64s=cats_int64s,
            cats_strings=cats_strings,
            zeros=zeros,
        ),
        _OneHotEncoder.Inputs(
            X=X,
        ),
    ).outputs.Y


def svmclassifier(
    X: Arrow,
    *,
    classlabels_ints: Optional[Iterable[int]] = None,
    classlabels_strings: Optional[Iterable[str]] = None,
    coefficients: Optional[Iterable[float]] = None,
    kernel_params: Optional[Iterable[float]] = None,
    kernel_type: str = "LINEAR",
    post_transform: str = "NONE",
    prob_a: Optional[Iterable[float]] = None,
    prob_b: Optional[Iterable[float]] = None,
    rho: Optional[Iterable[float]] = None,
    support_vectors: Optional[Iterable[float]] = None,
    vectors_per_class: Optional[Iterable[int]] = None,
) -> _SVMClassifier.Outputs:
    r"""
    Support Vector Machine classifier

    Parameters
    ==========
    X
        Type T1.
        Data to be classified.
    classlabels_ints
        Attribute.
        Class labels if using integer labels.

        One and only one of the 'classlabels_*' attributes must be defined.
    classlabels_strings
        Attribute.
        Class labels if using string labels.

        One and only one of the 'classlabels_*' attributes must be defined.
    coefficients
        Attribute.

    kernel_params
        Attribute.
        List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.
    kernel_type
        Attribute.
        The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.
    post_transform
        Attribute.
        Indicates the transform to apply to the score.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
    prob_a
        Attribute.
        First set of probability coefficients.
    prob_b
        Attribute.
        Second set of probability coefficients. This array must be same size as prob_a.

        If these are provided then output Z are probability estimates, otherwise they are raw scores.
    rho
        Attribute.

    support_vectors
        Attribute.

    vectors_per_class
        Attribute.


    Returns
    =======
    Y : Arrow
        Type T2.
        Classification outputs (one class per example).
    Z : Arrow
        Type tensor(float).
        Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::SVMClassifier``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(int64)`, `tensor(string)`
    """
    return _SVMClassifier(
        _SVMClassifier.Attributes(
            classlabels_ints=classlabels_ints,
            classlabels_strings=classlabels_strings,
            coefficients=coefficients,
            kernel_params=kernel_params,
            kernel_type=kernel_type,
            post_transform=post_transform,
            prob_a=prob_a,
            prob_b=prob_b,
            rho=rho,
            support_vectors=support_vectors,
            vectors_per_class=vectors_per_class,
        ),
        _SVMClassifier.Inputs(
            X=X,
        ),
    ).outputs


def svmregressor(
    X: Arrow,
    *,
    coefficients: Optional[Iterable[float]] = None,
    kernel_params: Optional[Iterable[float]] = None,
    kernel_type: str = "LINEAR",
    n_supports: int = 0,
    one_class: int = 0,
    post_transform: str = "NONE",
    rho: Optional[Iterable[float]] = None,
    support_vectors: Optional[Iterable[float]] = None,
) -> Arrow:
    r"""
    Support Vector Machine regression prediction and one-class SVM anomaly detection.

    Parameters
    ==========
    X
        Type T.
        Data to be regressed.
    coefficients
        Attribute.
        Support vector coefficients.
    kernel_params
        Attribute.
        List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.
    kernel_type
        Attribute.
        The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.
    n_supports
        Attribute.
        The number of support vectors.
    one_class
        Attribute.
        Flag indicating whether the regression is a one-class SVM or not.
    post_transform
        Attribute.
        Indicates the transform to apply to the score.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'
    rho
        Attribute.

    support_vectors
        Attribute.
        Chosen support vectors

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Regression outputs (one score per target per example).

    Notes
    =====
    Signature: ``ai.onnx.ml@1::SVMRegressor``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _SVMRegressor(
        _SVMRegressor.Attributes(
            coefficients=coefficients,
            kernel_params=kernel_params,
            kernel_type=kernel_type,
            n_supports=n_supports,
            one_class=one_class,
            post_transform=post_transform,
            rho=rho,
            support_vectors=support_vectors,
        ),
        _SVMRegressor.Inputs(
            X=X,
        ),
    ).outputs.Y


def scaler(
    X: Arrow,
    *,
    offset: Optional[Iterable[float]] = None,
    scale: Optional[Iterable[float]] = None,
) -> Arrow:
    r"""
    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

    Parameters
    ==========
    X
        Type T.
        Data to be scaled.
    offset
        Attribute.
        First, offset by this.

        Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count.
    scale
        Attribute.
        Second, multiply by this.

        Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count.

        Must be same length as 'offset'

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Scaled output data.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::Scaler``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _Scaler(
        _Scaler.Attributes(
            offset=offset,
            scale=scale,
        ),
        _Scaler.Inputs(
            X=X,
        ),
    ).outputs.Y


def tree_ensemble_classifier(
    X: Arrow,
    *,
    base_values: Optional[Iterable[float]] = None,
    base_values_as_tensor: Optional[ndarray] = None,
    class_ids: Optional[Iterable[int]] = None,
    class_nodeids: Optional[Iterable[int]] = None,
    class_treeids: Optional[Iterable[int]] = None,
    class_weights: Optional[Iterable[float]] = None,
    class_weights_as_tensor: Optional[ndarray] = None,
    classlabels_int64s: Optional[Iterable[int]] = None,
    classlabels_strings: Optional[Iterable[str]] = None,
    nodes_falsenodeids: Optional[Iterable[int]] = None,
    nodes_featureids: Optional[Iterable[int]] = None,
    nodes_hitrates: Optional[Iterable[float]] = None,
    nodes_hitrates_as_tensor: Optional[ndarray] = None,
    nodes_missing_value_tracks_true: Optional[Iterable[int]] = None,
    nodes_modes: Optional[Iterable[str]] = None,
    nodes_nodeids: Optional[Iterable[int]] = None,
    nodes_treeids: Optional[Iterable[int]] = None,
    nodes_truenodeids: Optional[Iterable[int]] = None,
    nodes_values: Optional[Iterable[float]] = None,
    nodes_values_as_tensor: Optional[ndarray] = None,
    post_transform: str = "NONE",
) -> _TreeEnsembleClassifier.Outputs:
    r"""
    Tree Ensemble classifier. Returns the top class for each of N inputs.


        The attributes named 'nodes_X' form a sequence of tuples, associated by
        index into the sequences, which must all be of equal length. These tuples
        define the nodes.


        Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
        A leaf may have multiple votes, where each vote is weighted by
        the associated class_weights index.


        One and only one of classlabels_strings or classlabels_int64s
        will be defined. The class_ids are indices into this list.
        All fields ending with `_as_tensor` can be used instead of the
        same parameter without the suffix if the element type is double and not float.

    Parameters
    ==========
    X
        Type T1.
        Input of shape [N,F]
    base_values
        Attribute.
        Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
    base_values_as_tensor
        Attribute.
        Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
    class_ids
        Attribute.
        The index of the class list that each weight is for.
    class_nodeids
        Attribute.
        node id that this weight is for.
    class_treeids
        Attribute.
        The id of the tree that this node is in.
    class_weights
        Attribute.
        The weight for the class in class_id.
    class_weights_as_tensor
        Attribute.
        The weight for the class in class_id.
    classlabels_int64s
        Attribute.
        Class labels if using integer labels.

        One and only one of the 'classlabels_*' attributes must be defined.
    classlabels_strings
        Attribute.
        Class labels if using string labels.

        One and only one of the 'classlabels_*' attributes must be defined.
    nodes_falsenodeids
        Attribute.
        Child node if expression is false.
    nodes_featureids
        Attribute.
        Feature id for each node.
    nodes_hitrates
        Attribute.
        Popularity of each node, used for performance and may be omitted.
    nodes_hitrates_as_tensor
        Attribute.
        Popularity of each node, used for performance and may be omitted.
    nodes_missing_value_tracks_true
        Attribute.
        For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the 'true' or 'false' branch based on the value in this array.

        This attribute may be left undefined, and the defalt value is false (0) for all nodes.
    nodes_modes
        Attribute.
        The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.

        One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'
    nodes_nodeids
        Attribute.
        Node id for each node. Ids may restart at zero for each tree, but it not required to.
    nodes_treeids
        Attribute.
        Tree id for each node.
    nodes_truenodeids
        Attribute.
        Child node if expression is true.
    nodes_values
        Attribute.
        Thresholds to do the splitting on for each node.
    nodes_values_as_tensor
        Attribute.
        Thresholds to do the splitting on for each node.
    post_transform
        Attribute.
        Indicates the transform to apply to the score.

         One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'

    Returns
    =======
    Y : Arrow
        Type T2.
        N, Top class for each point
    Z : Arrow
        Type tensor(float).
        The class score for each class, for each point, a tensor of shape [N,E].

    Notes
    =====
    Signature: ``ai.onnx.ml@3::TreeEnsembleClassifier``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(int64)`, `tensor(string)`
    """
    return _TreeEnsembleClassifier(
        _TreeEnsembleClassifier.Attributes(
            base_values=base_values,
            base_values_as_tensor=base_values_as_tensor,
            class_ids=class_ids,
            class_nodeids=class_nodeids,
            class_treeids=class_treeids,
            class_weights=class_weights,
            class_weights_as_tensor=class_weights_as_tensor,
            classlabels_int64s=classlabels_int64s,
            classlabels_strings=classlabels_strings,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_featureids=nodes_featureids,
            nodes_hitrates=nodes_hitrates,
            nodes_hitrates_as_tensor=nodes_hitrates_as_tensor,
            nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
            nodes_modes=nodes_modes,
            nodes_nodeids=nodes_nodeids,
            nodes_treeids=nodes_treeids,
            nodes_truenodeids=nodes_truenodeids,
            nodes_values=nodes_values,
            nodes_values_as_tensor=nodes_values_as_tensor,
            post_transform=post_transform,
        ),
        _TreeEnsembleClassifier.Inputs(
            X=X,
        ),
    ).outputs


def tree_ensemble_regressor(
    X: Arrow,
    *,
    aggregate_function: str = "SUM",
    base_values: Optional[Iterable[float]] = None,
    base_values_as_tensor: Optional[ndarray] = None,
    n_targets: Optional[int] = None,
    nodes_falsenodeids: Optional[Iterable[int]] = None,
    nodes_featureids: Optional[Iterable[int]] = None,
    nodes_hitrates: Optional[Iterable[float]] = None,
    nodes_hitrates_as_tensor: Optional[ndarray] = None,
    nodes_missing_value_tracks_true: Optional[Iterable[int]] = None,
    nodes_modes: Optional[Iterable[str]] = None,
    nodes_nodeids: Optional[Iterable[int]] = None,
    nodes_treeids: Optional[Iterable[int]] = None,
    nodes_truenodeids: Optional[Iterable[int]] = None,
    nodes_values: Optional[Iterable[float]] = None,
    nodes_values_as_tensor: Optional[ndarray] = None,
    post_transform: str = "NONE",
    target_ids: Optional[Iterable[int]] = None,
    target_nodeids: Optional[Iterable[int]] = None,
    target_treeids: Optional[Iterable[int]] = None,
    target_weights: Optional[Iterable[float]] = None,
    target_weights_as_tensor: Optional[ndarray] = None,
) -> Arrow:
    r"""
    Tree Ensemble regressor.  Returns the regressed values for each input in N.


        All args with nodes_ are fields of a tuple of tree nodes, and
        it is assumed they are the same length, and an index i will decode the
        tuple across these inputs.  Each node id can appear only once
        for each tree id.


        All fields prefixed with target_ are tuples of votes at the leaves.


        A leaf may have multiple votes, where each vote is weighted by
        the associated target_weights index.


        All fields ending with `_as_tensor` can be used instead of the
        same parameter without the suffix if the element type is double and not float.
        All trees must have their node ids start at 0 and increment by 1.


        Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF

    Parameters
    ==========
    X
        Type T.
        Input of shape [N,F]
    aggregate_function
        Attribute.
        Defines how to aggregate leaf values within a target.

        One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'
    base_values
        Attribute.
        Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
    base_values_as_tensor
        Attribute.
        Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
    n_targets
        Attribute.
        The total number of targets.
    nodes_falsenodeids
        Attribute.
        Child node if expression is false
    nodes_featureids
        Attribute.
        Feature id for each node.
    nodes_hitrates
        Attribute.
        Popularity of each node, used for performance and may be omitted.
    nodes_hitrates_as_tensor
        Attribute.
        Popularity of each node, used for performance and may be omitted.
    nodes_missing_value_tracks_true
        Attribute.
        For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) or 'false' (if the attribute value is 0) branch based on the value in this array.

        This attribute may be left undefined and the defalt value is false (0) for all nodes.
    nodes_modes
        Attribute.
        The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.

        One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'
    nodes_nodeids
        Attribute.
        Node id for each node. Node ids must restart at zero for each tree and increase sequentially.
    nodes_treeids
        Attribute.
        Tree id for each node.
    nodes_truenodeids
        Attribute.
        Child node if expression is true
    nodes_values
        Attribute.
        Thresholds to do the splitting on for each node.
    nodes_values_as_tensor
        Attribute.
        Thresholds to do the splitting on for each node.
    post_transform
        Attribute.
        Indicates the transform to apply to the score.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
    target_ids
        Attribute.
        The index of the target that each weight is for
    target_nodeids
        Attribute.
        The node id of each weight
    target_treeids
        Attribute.
        The id of the tree that each node is in.
    target_weights
        Attribute.
        The weight for each target
    target_weights_as_tensor
        Attribute.
        The weight for each target

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        N classes

    Notes
    =====
    Signature: ``ai.onnx.ml@3::TreeEnsembleRegressor``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _TreeEnsembleRegressor(
        _TreeEnsembleRegressor.Attributes(
            aggregate_function=aggregate_function,
            base_values=base_values,
            base_values_as_tensor=base_values_as_tensor,
            n_targets=n_targets,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_featureids=nodes_featureids,
            nodes_hitrates=nodes_hitrates,
            nodes_hitrates_as_tensor=nodes_hitrates_as_tensor,
            nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
            nodes_modes=nodes_modes,
            nodes_nodeids=nodes_nodeids,
            nodes_treeids=nodes_treeids,
            nodes_truenodeids=nodes_truenodeids,
            nodes_values=nodes_values,
            nodes_values_as_tensor=nodes_values_as_tensor,
            post_transform=post_transform,
            target_ids=target_ids,
            target_nodeids=target_nodeids,
            target_treeids=target_treeids,
            target_weights=target_weights,
            target_weights_as_tensor=target_weights_as_tensor,
        ),
        _TreeEnsembleRegressor.Inputs(
            X=X,
        ),
    ).outputs.Y


def zip_map(
    X: Arrow,
    *,
    classlabels_int64s: Optional[Iterable[int]] = None,
    classlabels_strings: Optional[Iterable[str]] = None,
) -> Arrow:
    r"""
    Creates a map from the input and the attributes.


        The values are provided by the input tensor, while the keys are specified by the attributes.
        Must provide keys in either classlabels_strings or classlabels_int64s (but not both).


        The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.



    Parameters
    ==========
    X
        Type tensor(float).
        The input values
    classlabels_int64s
        Attribute.
        The keys when using int keys.

        One and only one of the 'classlabels_*' attributes must be defined.
    classlabels_strings
        Attribute.
        The keys when using string keys.

        One and only one of the 'classlabels_*' attributes must be defined.

    Returns
    =======
    Z : Arrow
        Type T.
        The output map

    Notes
    =====
    Signature: ``ai.onnx.ml@1::ZipMap``.

    Type constraints:
     - T: `seq(map(int64,tensor(float)))`, `seq(map(string,tensor(float)))`
    """
    return _ZipMap(
        _ZipMap.Attributes(
            classlabels_int64s=classlabels_int64s,
            classlabels_strings=classlabels_strings,
        ),
        _ZipMap.Inputs(
            X=X,
        ),
    ).outputs.Z


_OPERATORS = {
    "ArrayFeatureExtractor": _ArrayFeatureExtractor,
    "Binarizer": _Binarizer,
    "CastMap": _CastMap,
    "CategoryMapper": _CategoryMapper,
    "DictVectorizer": _DictVectorizer,
    "FeatureVectorizer": _FeatureVectorizer,
    "Imputer": _Imputer,
    "LabelEncoder": _LabelEncoder,
    "LinearClassifier": _LinearClassifier,
    "LinearRegressor": _LinearRegressor,
    "Normalizer": _Normalizer,
    "OneHotEncoder": _OneHotEncoder,
    "SVMClassifier": _SVMClassifier,
    "SVMRegressor": _SVMRegressor,
    "Scaler": _Scaler,
    "TreeEnsembleClassifier": _TreeEnsembleClassifier,
    "TreeEnsembleRegressor": _TreeEnsembleRegressor,
    "ZipMap": _ZipMap,
}

CONSTRUCTORS = {
    "ArrayFeatureExtractor": array_feature_extractor,
    "Binarizer": binarizer,
    "CastMap": cast_map,
    "CategoryMapper": category_mapper,
    "DictVectorizer": dict_vectorizer,
    "FeatureVectorizer": feature_vectorizer,
    "Imputer": imputer,
    "LabelEncoder": label_encoder,
    "LinearClassifier": linear_classifier,
    "LinearRegressor": linear_regressor,
    "Normalizer": normalizer,
    "OneHotEncoder": one_hot_encoder,
    "SVMClassifier": svmclassifier,
    "SVMRegressor": svmregressor,
    "Scaler": scaler,
    "TreeEnsembleClassifier": tree_ensemble_classifier,
    "TreeEnsembleRegressor": tree_ensemble_regressor,
    "ZipMap": zip_map,
}
