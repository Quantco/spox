from abc import ABC
from typing import Any, ClassVar, Generic, Iterable, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from onnx import AttributeProto
from onnx.helper import (
    make_attribute,
    make_optional_type_proto,
    make_sequence_type_proto,
    make_tensor_type_proto,
)
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from steelix import graph, type_system
from steelix._utils import from_array

S = TypeVar("S")
T = TypeVar("T")


class Attr(ABC, Generic[T]):
    _value: Union[T, "_Ref[T]"]
    _attribute_proto_type_int: ClassVar[int]

    def __init__(self, value: Union[T, "_Ref[T]"]):
        self._value = value
        self._validate()

    @property
    def value(self) -> T:
        # implicitly "dereference" `_Ref`
        if isinstance(self._value, _Ref):
            return _deref(self._value)
        return self._value

    def _validate(self):
        if not self._to_onnx("dummy").type == self._attribute_proto_type_int:
            raise TypeError(
                f"Unable to instantiate `{type(self).__name__}` with value of type `{type(self.value).__name__}`."
            )

    def _to_onnx(self, key: str) -> AttributeProto:
        if isinstance(self._value, _Ref):
            return self._value._to_onnx(key)
        return self._to_onnx_deref(key)

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        """Conversion method for the dereferenced case."""
        raise NotImplementedError


class _Ref(Generic[T]):
    """
    Special attribute value used in function bodies.

    An ``AttrRef`` is a reference to an attribute defined
    elsewhere. May be used as ``_value`` in ``Attr*`` classes.
    """

    _concrete: Attr[T]

    def __init__(self, concrete: Attr[T], outer_name: str):
        self._concrete = concrete
        self._outer_name = outer_name

    def _to_onnx(self, key: str) -> AttributeProto:
        parent_type = self._concrete._to_onnx(key).type
        return AttributeProto(
            name=key, ref_attr_name=self._outer_name, type=parent_type
        )


class AttrFloat32(Attr[float]):
    _attribute_proto_type_int = AttributeProto.FLOAT

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        if isinstance(self.value, int):
            return make_attribute(key, float(self.value))
        return make_attribute(key, self.value)


class AttrInt64(Attr[int]):
    _attribute_proto_type_int = AttributeProto.INT

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        return make_attribute(key, self.value)


class AttrString(Attr[str]):
    _attribute_proto_type_int = AttributeProto.STRING

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        # Strings are bytes on the onnx side
        return make_attribute(key, self.value.encode())


class AttrTensor(Attr[np.ndarray]):
    _attribute_proto_type_int = AttributeProto.TENSOR

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        return make_attribute(key, from_array(self.value))


class AttrType(Attr[type_system.Type]):
    _attribute_proto_type_int = AttributeProto.TYPE_PROTO

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        value = self.value  # for type-checkers with limited property support
        if isinstance(value, type_system.Tensor):
            type_proto = make_tensor_type_proto(
                value.elem_type_to_onnx(value.elem_type),
                value.shape.to_simple(),
            )
        elif isinstance(value, type_system.Sequence):
            type_proto = make_sequence_type_proto(value.elem_type.to_onnx())
        elif isinstance(value, type_system.Optional):
            type_proto = make_optional_type_proto(value.elem_type.to_onnx())
        else:
            raise NotImplementedError
        return make_attribute(key, type_proto)


class AttrDtype(Attr[npt.DTypeLike]):
    """Special attribute for specifying data types as ``numpy.dtype``s, for example in ``Cast``."""

    _attribute_proto_type_int = AttributeProto.INT

    def _validate(self):
        if self.value is None:
            raise TypeError("")
        self._get_tensor_type()

    def _get_tensor_type(self) -> int:
        dtype = np.dtype(self.value)
        # There are various different dtypes denoting strings
        if dtype.type == np.str_:
            dtype = np.dtype("O")

        # `None` is included in DTypeLike and defaults to
        # `float64`. We don't want to allow that implicit behavior
        # (for now).
        if self.value is None or dtype not in NP_TYPE_TO_TENSOR_TYPE:
            raise TypeError(
                f"`{self.value}` does not have a corresponding tensor type."
            )
        return NP_TYPE_TO_TENSOR_TYPE[dtype]

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        return make_attribute(key, self._get_tensor_type())


class AttrGraph(Attr[Any]):
    _attribute_proto_type_int = AttributeProto.GRAPH

    def _validate(self):
        if not isinstance(self.value, graph.Graph):
            raise TypeError(
                f"Expected value of type `steelix.graph.Graph found `{type(self.value)}`"
            )

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        raise TypeError(
            "Graph attributes must be built using the `build_subgraph` callback in `Node.to_onnx`."
        )


class _AttrIterable(Attr[Tuple[S, ...]], ABC):
    def __init__(self, value: Union[Iterable[S], _Ref[Tuple[S, ...]]]):
        super().__init__(value if isinstance(value, _Ref) else tuple(value))


class AttrFloat32s(_AttrIterable[float]):
    _attribute_proto_type_int = AttributeProto.FLOATS

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        # ensure values are all floats
        values = [float(v) for v in self.value]
        return make_attribute(key, values)


class AttrInt64s(_AttrIterable[int]):
    _attribute_proto_type_int = AttributeProto.INTS

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        return make_attribute(key, self.value)


class AttrStrings(_AttrIterable[str]):
    _attribute_proto_type_int = AttributeProto.STRINGS

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        return make_attribute(key, [v.encode() for v in self.value])


class AttrTensors(_AttrIterable[np.ndarray]):
    _attribute_proto_type_int = AttributeProto.TENSORS

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        tensors = [from_array(t) for t in self.value]
        return make_attribute(key, tensors)


def _deref(ref: _Ref[T]) -> T:
    if isinstance(ref._concrete._value, _Ref):
        return _deref(ref._concrete._value)
    return ref._concrete._value