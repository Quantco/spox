# ruff: noqa: E741 -- Allow ambiguous variable name
from dataclasses import dataclass
from typing import (
    Iterable,
    Optional,
)

import numpy as np

from spox._attributes import (
    AttrInt64,
    AttrInt64s,
    AttrTensor,
)
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs
from spox._node import OpType
from spox._standard import StandardNode
from spox._var import Var
from spox.opset.ai.onnx.ml.v4 import (
    _ArrayFeatureExtractor,
    _Binarizer,
    _CastMap,
    _CategoryMapper,
    _DictVectorizer,
    _FeatureVectorizer,
    _Imputer,
    _LabelEncoder,
    _LinearClassifier,
    _LinearRegressor,
    _Normalizer,
    _OneHotEncoder,
    _Scaler,
    _SVMClassifier,
    _SVMRegressor,
    _ZipMap,
    array_feature_extractor,
    binarizer,
    cast_map,
    category_mapper,
    dict_vectorizer,
    feature_vectorizer,
    imputer,
    label_encoder,
    linear_classifier,
    linear_regressor,
    normalizer,
    one_hot_encoder,
    scaler,
    svmclassifier,
    svmregressor,
    zip_map,
)


class _TreeEnsemble(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        aggregate_function: AttrInt64
        leaf_targetids: AttrInt64s
        leaf_weights: AttrTensor
        membership_values: Optional[AttrTensor]
        n_targets: Optional[AttrInt64]
        nodes_falseleafs: AttrInt64s
        nodes_falsenodeids: AttrInt64s
        nodes_featureids: AttrInt64s
        nodes_hitrates: Optional[AttrTensor]
        nodes_missing_value_tracks_true: Optional[AttrInt64s]
        nodes_modes: AttrTensor
        nodes_splits: AttrTensor
        nodes_trueleafs: AttrInt64s
        nodes_truenodeids: AttrInt64s
        post_transform: AttrInt64
        tree_roots: AttrInt64s

    @dataclass
    class Inputs(BaseInputs):
        X: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("TreeEnsemble", "ai.onnx.ml", 5)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def tree_ensemble(
    X: Var,
    *,
    aggregate_function: int = 1,
    leaf_targetids: Iterable[int],
    leaf_weights: np.ndarray,
    membership_values: Optional[np.ndarray] = None,
    n_targets: Optional[int] = None,
    nodes_falseleafs: Iterable[int],
    nodes_falsenodeids: Iterable[int],
    nodes_featureids: Iterable[int],
    nodes_hitrates: Optional[np.ndarray] = None,
    nodes_missing_value_tracks_true: Optional[Iterable[int]] = None,
    nodes_modes: np.ndarray,
    nodes_splits: np.ndarray,
    nodes_trueleafs: Iterable[int],
    nodes_truenodeids: Iterable[int],
    post_transform: int = 0,
    tree_roots: Iterable[int],
) -> Var:
    r"""
    Tree Ensemble operator. Returns the regressed values for each input in a
    batch. Inputs have dimensions ``[N, F]`` where ``N`` is the input batch
    size and ``F`` is the number of input features. Outputs have dimensions
    ``[N, num_targets]`` where ``N`` is the batch size and ``num_targets``
    is the number of targets, which is a configurable attribute.

    ::

       The encoding of this attribute is split along interior nodes and the leaves of the trees. Notably, attributes with the prefix `nodes_*` are associated with interior nodes, and attributes with the prefix `leaf_*` are associated with leaves.
       The attributes `nodes_*` must all have the same length and encode a sequence of tuples, as defined by taking all the `nodes_*` fields at a given position.

       All fields prefixed with `leaf_*` represent tree leaves, and similarly define tuples of leaves and must have identical length.

       This operator can be used to implement both the previous `TreeEnsembleRegressor` and `TreeEnsembleClassifier` nodes.
       The `TreeEnsembleRegressor` node maps directly to this node and requires changing how the nodes are represented.
       The `TreeEnsembleClassifier` node can be implemented by adding a `ArgMax` node after this node to determine the top class.
       To encode class labels, a `LabelEncoder` or `GatherND` operator may be used.

    Parameters
    ==========
    X
        Type T.
        Input of shape [Batch Size, Number of Features]
    aggregate_function
        Attribute.
        Defines how to aggregate leaf values within a target. One of 'AVERAGE'
        (0) 'SUM' (1) 'MIN' (2) 'MAX (3) defaults to 'SUM' (1)
    leaf_targetids
        Attribute.
        The index of the target that this leaf contributes to (this must be in
        range ``[0, n_targets)``).
    leaf_weights
        Attribute.
        The weight for each leaf.
    membership_values
        Attribute.
        Members to test membership of for each set membership node. List all of
        the members to test again in the order that the 'BRANCH_MEMBER' mode
        appears in ``node_modes``, delimited by ``NaN``\ s. Will have the same
        number of sets of values as nodes with mode 'BRANCH_MEMBER'. This may be
        omitted if the node doesn't contain any 'BRANCH_MEMBER' nodes.
    n_targets
        Attribute.
        The total number of targets.
    nodes_falseleafs
        Attribute.
        1 if false branch is leaf for each node and 0 if an interior node. To
        represent a tree that is a leaf (only has one node), one can do so by
        having a single ``nodes_*`` entry with true and false branches
        referencing the same ``leaf_*`` entry
    nodes_falsenodeids
        Attribute.
        If ``nodes_falseleafs`` is false at an entry, this represents the
        position of the false branch node. This position can be used to index
        into a ``nodes_*`` entry. If ``nodes_falseleafs`` is false, it is an
        index into the leaf\_\* attributes.
    nodes_featureids
        Attribute.
        Feature id for each node.
    nodes_hitrates
        Attribute.
        Popularity of each node, used for performance and may be omitted.
    nodes_missing_value_tracks_true
        Attribute.
        For each node, define whether to follow the true branch (if attribute
        value is 1) or false branch (if attribute value is 0) in the presence of
        a NaN input feature. This attribute may be left undefined and the
        default value is false (0) for all nodes.
    nodes_modes
        Attribute.
        The comparison operation performed by the node. This is encoded as an
        enumeration of 0 ('BRANCH_LEQ'), 1 ('BRANCH_LT'), 2 ('BRANCH_GTE'), 3
        ('BRANCH_GT'), 4 ('BRANCH_EQ'), 5 ('BRANCH_NEQ'), and 6
        ('BRANCH_MEMBER'). Note this is a tensor of type uint8.
    nodes_splits
        Attribute.
        Thresholds to do the splitting on for each node with mode that is not
        'BRANCH_MEMBER'.
    nodes_trueleafs
        Attribute.
        1 if true branch is leaf for each node and 0 an interior node. To
        represent a tree that is a leaf (only has one node), one can do so by
        having a single ``nodes_*`` entry with true and false branches
        referencing the same ``leaf_*`` entry
    nodes_truenodeids
        Attribute.
        If ``nodes_trueleafs`` is false at an entry, this represents the
        position of the true branch node. This position can be used to index
        into a ``nodes_*`` entry. If ``nodes_trueleafs`` is false, it is an
        index into the leaf\_\* attributes.
    post_transform
        Attribute.
        Indicates the transform to apply to the score. One of 'NONE' (0),
        'SOFTMAX' (1), 'LOGISTIC' (2), 'SOFTMAX_ZERO' (3) or 'PROBIT' (4),
        defaults to 'NONE' (0)
    tree_roots
        Attribute.
        Index into ``nodes_*`` for the root of each tree. The tree structure is
        derived from the branching of each node.

    Returns
    =======
    Y : Var
        Type T.
        Output of shape [Batch Size, Number of targets]

    Notes
    =====
    Signature: ``ai.onnx.ml@5::TreeEnsemble``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(float16)`
    """
    return _TreeEnsemble(
        _TreeEnsemble.Attributes(
            aggregate_function=AttrInt64(aggregate_function, name="aggregate_function"),
            leaf_targetids=AttrInt64s(leaf_targetids, name="leaf_targetids"),
            leaf_weights=AttrTensor(leaf_weights, name="leaf_weights"),
            membership_values=AttrTensor.maybe(
                membership_values, name="membership_values"
            ),
            n_targets=AttrInt64.maybe(n_targets, name="n_targets"),
            nodes_falseleafs=AttrInt64s(nodes_falseleafs, name="nodes_falseleafs"),
            nodes_falsenodeids=AttrInt64s(
                nodes_falsenodeids, name="nodes_falsenodeids"
            ),
            nodes_featureids=AttrInt64s(nodes_featureids, name="nodes_featureids"),
            nodes_hitrates=AttrTensor.maybe(nodes_hitrates, name="nodes_hitrates"),
            nodes_missing_value_tracks_true=AttrInt64s.maybe(
                nodes_missing_value_tracks_true, name="nodes_missing_value_tracks_true"
            ),
            nodes_modes=AttrTensor(nodes_modes, name="nodes_modes"),
            nodes_splits=AttrTensor(nodes_splits, name="nodes_splits"),
            nodes_trueleafs=AttrInt64s(nodes_trueleafs, name="nodes_trueleafs"),
            nodes_truenodeids=AttrInt64s(nodes_truenodeids, name="nodes_truenodeids"),
            post_transform=AttrInt64(post_transform, name="post_transform"),
            tree_roots=AttrInt64s(tree_roots, name="tree_roots"),
        ),
        _TreeEnsemble.Inputs(
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
    "TreeEnsemble": _TreeEnsemble,
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
    "TreeEnsemble": tree_ensemble,
    "ZipMap": zip_map,
}

__all__ = [fun.__name__ for fun in _CONSTRUCTORS.values()]
