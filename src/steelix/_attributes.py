from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Union

import numpy as np
from onnx import AttributeProto
from onnx.helper import make_attribute, make_sequence_type_proto, make_tensor_type_proto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from steelix import type_system
from steelix._utils import from_array


class Attr(ABC):
    value: Any

    def _to_onnx(self, key: str) -> AttributeProto:
        raise NotImplementedError

    def _deref(self) -> "Attr":
        """Dereference ``self`` if its ``value`` is an ``AttrRef``.

        This operation is idempotent if ``value`` is not an ``AttrRef``.
        """
        if isinstance(self.value, _Ref):
            return self.value._concrete
        return self


class _Ref:
    """Special attribute value used in function bodies.

    An ``AttrRef`` is a reference to an attribute defined
    elsewhere. May be used as ``value`` in ``Attr*`` classes.
    """

    def __init__(self, concrete: Attr, outer_name: str):
        self._concrete = concrete
        self._outer_name = outer_name

    def _to_onnx(self, key: str) -> AttributeProto:
        parent_type = self._concrete._to_onnx(key).type
        return AttributeProto(
            name=key, ref_attr_name=self._outer_name, type=parent_type
        )


@dataclass
class AttrFloat32(Attr):
    value: Union[float, _Ref]

    def _to_onnx(self, key: str) -> AttributeProto:
        return _make_attribute_maybe_ref(key, self.value)


@dataclass
class AttrInt64(Attr):
    value: Union[int, _Ref]

    def _to_onnx(self, key: str) -> AttributeProto:
        return _make_attribute_maybe_ref(key, self.value)


@dataclass
class AttrString(Attr):
    value: Union[str, _Ref]

    def _to_onnx(self, key: str) -> AttributeProto:
        # Strings are bytes on the onnx side
        if isinstance(self.value, _Ref):
            return self.value._to_onnx(key)
        return make_attribute(key, self.value.encode())


@dataclass
class AttrFloat32s(Attr):
    value: Union[Tuple[float, ...], _Ref]

    def __init__(self, value: Union[Iterable[float], _Ref]):
        if isinstance(value, Iterable):
            self.value = tuple(value)
        else:
            self.value = value

    def _to_onnx(self, key: str) -> AttributeProto:
        return _make_attribute_maybe_ref(key, self.value)


@dataclass
class AttrInt64s(Attr):
    value: Union[Tuple[int, ...], _Ref]

    def __init__(self, value: Union[Iterable[int], _Ref]):
        if isinstance(value, Iterable):
            self.value = tuple(value)
        else:
            self.value = value

    def _to_onnx(self, key: str) -> AttributeProto:
        return _make_attribute_maybe_ref(key, self.value)


@dataclass
class AttrStrings(Attr):
    value: Union[Tuple[str, ...], _Ref]

    def __init__(self, value: Union[Iterable[str], _Ref]):
        if isinstance(value, Iterable):
            self.value = tuple(value)
        else:
            self.value = value

    def _to_onnx(self, key: str) -> AttributeProto:
        if isinstance(self.value, _Ref):
            return self.value._to_onnx(key)
        return make_attribute(key, [v.encode() for v in self.value])


@dataclass
class AttrTensors(Attr):
    value: Union[Tuple[np.ndarray, ...], _Ref]

    def __init__(self, value: Union[Iterable[np.ndarray], _Ref]):
        if isinstance(value, Iterable):
            self.value = tuple(value)
        else:
            self.value = value

    def _to_onnx(self, key: str) -> AttributeProto:
        if isinstance(self.value, _Ref):
            return self.value._to_onnx(key)
        tensors = [from_array(t) for t in self.value]
        return make_attribute(key, tensors)


@dataclass
class AttrTensor(Attr):
    value: Union[np.ndarray, _Ref]

    def _to_onnx(self, key: str) -> AttributeProto:
        if isinstance(self.value, _Ref):
            return self.value._to_onnx(key)

        return make_attribute(key, from_array(self.value))


@dataclass
class AttrType(Attr):
    value: Union[type_system.Type, _Ref]

    def _to_onnx(self, key: str) -> AttributeProto:
        if isinstance(self.value, _Ref):
            return self.value._to_onnx(key)

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


@dataclass
class AttrDtype(Attr):
    """Special attribute for sepecifying data types as `numpy.dtype`s."""

    value: Union[np.dtype, np.generic]

    def _to_onnx(self, key: str) -> AttributeProto:
        if isinstance(self.value, _Ref):
            return self.value._to_onnx(key)

        dtype = np.dtype(self.value)
        # There are various different dtypes denoting strings
        if dtype.type == np.str_:
            return make_attribute(key, NP_TYPE_TO_TENSOR_TYPE[np.dtype("O")])
        return make_attribute(key, NP_TYPE_TO_TENSOR_TYPE[dtype])


@dataclass
class AttrGraph(Attr):
    value: Any

    def _to_onnx(self, key: str) -> AttributeProto:
        # Build with build_subgraph in Node, currently
        raise NotImplementedError


def _make_attribute_maybe_ref(key, value: Union[Any, _Ref]) -> AttributeProto:
    if isinstance(value, _Ref):
        return value._to_onnx(key)
    return make_attribute(key, value)
