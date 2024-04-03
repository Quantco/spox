# ruff: noqa: E741 -- Allow ambiguous variable name
from dataclasses import dataclass
from typing import (
    Iterable,
    Optional,
)

import numpy as np

from spox._attributes import (
    AttrFloat32,
    AttrFloat32s,
    AttrInt64,
    AttrInt64s,
    AttrString,
    AttrStrings,
    AttrTensor,
)
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._node import OpType
from spox._standard import StandardNode
from spox._var import Var
from spox.opset.ai.onnx.ml.v3 import (
    _ArrayFeatureExtractor,
    _Binarizer,
    _CastMap,
    _CategoryMapper,
    _DictVectorizer,
    _FeatureVectorizer,
    _Imputer,
    _LinearClassifier,
    _LinearRegressor,
    _Normalizer,
    _OneHotEncoder,
    _Scaler,
    _SVMClassifier,
    _SVMRegressor,
    _TreeEnsembleClassifier,
    _TreeEnsembleRegressor,
    _ZipMap,
    array_feature_extractor,
    binarizer,
    cast_map,
    category_mapper,
    dict_vectorizer,
    feature_vectorizer,
    imputer,
    linear_classifier,
    linear_regressor,
    normalizer,
    one_hot_encoder,
    scaler,
    svmclassifier,
    svmregressor,
    tree_ensemble_classifier,
    tree_ensemble_regressor,
    zip_map,
)


class _LabelEncoder(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        default_float: AttrFloat32
        default_int64: AttrInt64
        default_string: AttrString
        default_tensor: Optional[AttrTensor]
        keys_floats: Optional[AttrFloat32s]
        keys_int64s: Optional[AttrInt64s]
        keys_strings: Optional[AttrStrings]
        keys_tensor: Optional[AttrTensor]
        values_floats: Optional[AttrFloat32s]
        values_int64s: Optional[AttrInt64s]
        values_strings: Optional[AttrStrings]
        values_tensor: Optional[AttrTensor]

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("LabelEncoder", "ai.onnx.ml", 4)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def label_encoder(
    X: Var,
    *,
    default_float: float = -0.0,
    default_int64: int = -1,
    default_string: str = "_Unused",
    default_tensor: Optional[np.ndarray] = None,
    keys_floats: Optional[Iterable[float]] = None,
    keys_int64s: Optional[Iterable[int]] = None,
    keys_strings: Optional[Iterable[str]] = None,
    keys_tensor: Optional[np.ndarray] = None,
    values_floats: Optional[Iterable[float]] = None,
    values_int64s: Optional[Iterable[int]] = None,
    values_strings: Optional[Iterable[str]] = None,
    values_tensor: Optional[np.ndarray] = None,
) -> Var:
    r"""
    Maps each element in the input tensor to another value. The mapping is
    determined by the two parallel attributes, 'keys\_\ *' and 'values\_*'
    attribute. The i-th value in the specified 'keys\_\ *' attribute would
    be mapped to the i-th value in the specified 'values\_*' attribute. It
    implies that input's element type and the element type of the specified
    'keys\_\ *' should be identical while the output type is identical to
    the specified 'values\_*' attribute. Note that the 'keys\_\ *' and
    'values\_*' attributes must have the same length. If an input element
    can not be found in the specified 'keys\_\ *' attribute, the
    'default\_*' that matches the specified 'values\_\ *' attribute may be
    used as its output value. The type of the 'default\_*' attribute must
    match the 'values\_\ *' attribute chosen. Let's consider an example
    which maps a string tensor to an integer tensor. Assume and
    'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6], and
    'default_int64' is '-1'. The input ["Dori", "Amy", "Amy", "Sally",
    "Sally"] would be mapped to [-1, 5, 5, 6, 6]. Since this operator is an
    one-to-one mapping, its input and output shapes are the same. Notice
    that only one of 'keys\_*'/'values\_\*' can be set. Float keys with
    value 'NaN' match any input 'NaN' value regardless of bit value. If a
    key is repeated, the last key takes precedence.

    Parameters
    ==========
    X
        Type T1.
        Input data. It must have the same element type as the keys\_\* attribute
        set.
    default_float
        Attribute.
        A float.
    default_int64
        Attribute.
        An integer.
    default_string
        Attribute.
        A string.
    default_tensor
        Attribute.
        A default tensor. {"*Unused"} if values*\ \* has string type, {-1} if
        values\_\* has integral type, and {-0.f} if values\_\* has float type.
    keys_floats
        Attribute.
        A list of floats.
    keys_int64s
        Attribute.
        A list of ints.
    keys_strings
        Attribute.
        A list of strings.
    keys_tensor
        Attribute.
        Keys encoded as a 1D tensor. One and only one of 'keys\_\*'s should be
        set.
    values_floats
        Attribute.
        A list of floats.
    values_int64s
        Attribute.
        A list of ints.
    values_strings
        Attribute.
        A list of strings.
    values_tensor
        Attribute.
        Values encoded as a 1D tensor. One and only one of 'values\_\*'s should
        be set.

    Returns
    =======
    Y : Var
        Type T2.
        Output data. This tensor's element type is based on the values\_\*
        attribute set.

    Notes
    =====
    Signature: ``ai.onnx.ml@4::LabelEncoder``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(string)`
     - T2: `tensor(double)`, `tensor(float)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(string)`
    """
    return _LabelEncoder(
        _LabelEncoder.Attributes(
            default_float=AttrFloat32(default_float, name="default_float"),
            default_int64=AttrInt64(default_int64, name="default_int64"),
            default_string=AttrString(default_string, name="default_string"),
            default_tensor=AttrTensor.maybe(default_tensor, name="default_tensor"),
            keys_floats=AttrFloat32s.maybe(keys_floats, name="keys_floats"),
            keys_int64s=AttrInt64s.maybe(keys_int64s, name="keys_int64s"),
            keys_strings=AttrStrings.maybe(keys_strings, name="keys_strings"),
            keys_tensor=AttrTensor.maybe(keys_tensor, name="keys_tensor"),
            values_floats=AttrFloat32s.maybe(values_floats, name="values_floats"),
            values_int64s=AttrInt64s.maybe(values_int64s, name="values_int64s"),
            values_strings=AttrStrings.maybe(values_strings, name="values_strings"),
            values_tensor=AttrTensor.maybe(values_tensor, name="values_tensor"),
        ),
        _LabelEncoder.Inputs(
            X=X,
        ),
    ).outputs.Y


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

_CONSTRUCTORS = {
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

__all__ = [fun.__name__ for fun in _CONSTRUCTORS.values()]
