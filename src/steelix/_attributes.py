from abc import ABC
from typing import Any, Generic, Iterable, Tuple, TypeVar, Union

import numpy as np
from onnx import AttributeProto
from onnx.helper import make_attribute, make_sequence_type_proto, make_tensor_type_proto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from steelix import type_system
from steelix._utils import from_array

T = TypeVar("T")


class Attr(ABC, Generic[T]):
    _value: Union[T, "_Ref[T]"]

    def __init__(self, value: Union[T, "_Ref[T]"]):
        self._value = value

    @property
    def value(self) -> T:
        # implicitly "dereference" `_Ref`
        if isinstance(self._value, _Ref):
            return _deref(self._value)
        return self._value

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
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        if isinstance(self.value, int):
            return make_attribute(key, float(self.value))
        return make_attribute(key, self.value)


class AttrInt64(Attr[int]):
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        return make_attribute(key, self.value)


class AttrString(Attr[str]):
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        # Strings are bytes on the onnx side
        return make_attribute(key, self.value.encode())


class AttrTensor(Attr[np.ndarray]):
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        return make_attribute(key, from_array(self.value))


class AttrType(Attr[type_system.Type]):
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        if isinstance(self.value, type_system.Tensor):
            type_proto = make_tensor_type_proto(
                self.value.elem_type_to_onnx(self.value.elem_type),
                self.value.shape.to_simple(),
            )
        elif isinstance(self.value, type_system.Sequence):
            type_proto = make_sequence_type_proto(self.value.elem_type.to_onnx())
        else:
            raise NotImplementedError
        return make_attribute(key, type_proto)


class AttrDtype(Attr[Union[np.dtype, np.generic]]):
    """Special attribute for specifying data types as ``numpy.dtype``s, for example in ``Cast``."""

    def _to_onnx_deref(self, key: str) -> AttributeProto:
        dtype = np.dtype(self.value)
        # There are various different dtypes denoting strings
        if dtype.type == np.str_:
            return make_attribute(key, NP_TYPE_TO_TENSOR_TYPE[np.dtype("O")])
        return make_attribute(key, NP_TYPE_TO_TENSOR_TYPE[dtype])


class AttrGraph(Attr[Any]):
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        raise TypeError(
            "Graph attributes must be built using the `build_subgraph` callback in `Node.to_onnx`."
        )


class _AttrIterable(Attr[Tuple[T, ...]], ABC):
    def __init__(self, value: Union[Iterable[T], _Ref]):
        super().__init__(value if isinstance(value, _Ref) else tuple(value))


class AttrFloat32s(_AttrIterable[float]):
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        # ensure values are all floats
        values = [float(v) for v in self.value]
        return make_attribute(key, values)


class AttrInt64s(_AttrIterable[int]):
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        return make_attribute(key, self.value)


class AttrStrings(_AttrIterable[str]):
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        return make_attribute(key, [v.encode() for v in self.value])


class AttrTensors(_AttrIterable[np.ndarray]):
    def _to_onnx_deref(self, key: str) -> AttributeProto:
        tensors = [from_array(t) for t in self.value]
        return make_attribute(key, tensors)


def _deref(ref: _Ref[T]) -> T:
    if isinstance(ref._concrete._value, _Ref):
        return _deref(ref._concrete._value)
    return ref._concrete._value
