import typing
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, Optional, Sequence, TypeVar

import numpy
import onnx
import onnx.numpy_helper
from typing_extensions import Annotated

from .type_system import Tensor, Type

T = TypeVar("T")
S = TypeVar("S")


def _is_list_attribute(hint):
    origin = typing.get_origin(hint)
    return isinstance(origin, type) and issubclass(origin, Sequence)


@dataclass(frozen=True)
class Ref:
    """
    Represents a reference to a parent function's attribute.

    May be passed down to operators and other functions.

    Function attributes only have basic support in ONNX Runtime and are somewhat buggy. Hence, they are not
    well-tested in Steelix.
    """

    value_type: type
    name: str
    parent: Any


class Attr(Generic[T]):
    """
    Represents a single (unnamed) strongly typed attribute for an ONNX operator.

    Has a known Python type - ``value_type``, which must be declared on construction.
    Fields types may use type annotations from the class definition to extract it.

    This ``value_type`` is used to guide some type promotion/fixing behaviour, like making sure a
    float attribute value is converted to float if it is an int.

    Additionally, the held value may be a ``Ref`` for any type. In that case it is an attribute reference,
    and the ``.value`` getter will get its exact value from the parent of the reference.

    Refs are used for representing attributes of the function in a function body. When defining functions,
    to access the Ref object (so that it can be passed to the body of other functions) use ``._value`` directly.

    However, note that function attributes are not fully supported by ONNX Runtime.

    The generic annotation is used for introspection when constructing a Node and its Attr,
    marking the expected type of the attribute.
    """

    value_type: type
    _value: Any

    def __init__(self, value_type: type, value, *, _traceback_name: str = "?"):
        self.value_type = value_type
        self._value = self._cast_attr_value(value_type, value, _traceback_name)

    @staticmethod
    def _cast_attr_value(value_type: type, value, _traceback_name: str = "?"):
        if value is None:  # Cannot fix unset value
            return value
        if isinstance(value, Ref):  # Assume that Refs already had a fixed value
            if value.value_type != value_type:
                raise TypeError(
                    f"Ref expected to have value_type {value_type}, but had {value.value_type}."
                )
            return value
        assert (
            value_type in Attr.py_type_to_attr_type()
        ), f"Declared Attr type {value_type} is not supported."
        # Force-cast (fix) to expected type to avoid bad inference
        # by make_attribute (like int vs. float)
        if _is_list_attribute(value_type):
            inner_type = typing.get_args(value_type)[0]
            return tuple(
                _cast_attr_value(
                    inner_type, value, f"<element of list {_traceback_name}>"
                )
                for value in typing.cast(Iterable, value)
            )
        return _cast_attr_value(value_type, value, _traceback_name)

    @staticmethod
    def attr_type_to_py_type() -> Dict[
        Annotated[int, onnx.AttributeProto.AttributeType], type
    ]:
        """Dictionary from AttributeProto.AttributeType enum into Steelix-representation types."""
        from .graph import Graph

        return {
            onnx.AttributeProto.FLOAT: float,
            onnx.AttributeProto.INT: int,
            onnx.AttributeProto.STRING: str,
            onnx.AttributeProto.TENSOR: numpy.ndarray,
            onnx.AttributeProto.GRAPH: Graph,
            onnx.AttributeProto.TYPE_PROTO: Type,
            onnx.AttributeProto.FLOATS: Sequence[float],
            onnx.AttributeProto.INTS: Sequence[int],
            onnx.AttributeProto.STRINGS: Sequence[str],
            onnx.AttributeProto.TENSORS: Sequence[numpy.ndarray],
            onnx.AttributeProto.TYPE_PROTOS: Sequence[Type],
        }

    @staticmethod
    def py_type_to_attr_type() -> Dict[
        type, Annotated[int, onnx.AttributeProto.AttributeType]
    ]:
        """
        Inverse map of ``attr_type_to_py_type``.

        However, since some Python types (like ``numpy.generic`` for ``Tensor.elem_type``) may be represented
        with the same AttributeType, there are collisions possible.
        """
        return {
            **{v: k for k, v in Attr.attr_type_to_py_type().items()},
            numpy.generic: onnx.AttributeProto.INT,
        }

    @property
    def attr_proto_type(self) -> onnx.AttributeProto.AttributeType:
        """Get an AttributeProto enum for this Attr's ``value_type``."""
        return typing.cast(
            onnx.AttributeProto.AttributeType,
            self.py_type_to_attr_type()[self.value_type],
        )

    @classmethod
    def from_onnx(
        cls, proto: onnx.AttributeProto, parent: Optional[Any] = None
    ) -> "Attr":
        """Translate an AttributeProto into an Attr."""
        value_type = cls.attr_type_to_py_type()[proto.type]
        if proto.HasField("ref_attr_name"):
            return Attr(
                value_type,
                Ref(value_type, proto.ref_attr_name, parent),
                _traceback_name=proto.name,
            )
        value = onnx.helper.get_attribute_value(proto)
        return Attr(value_type, value, _traceback_name=proto.name)

    @property
    def value(self):
        """
        Access the concrete value of this attribute, in particular evaluating Ref.

        If the Ref object itself needs to be accessed, use the protected attribute ``._value``.
        """
        if isinstance(self._value, Ref):
            if self._value.parent is None:
                raise AttributeError(
                    f"{self} has no parent fields set."
                    "Is this the right context to request the value of a RefAttr object, or was it created wrong?"
                )
            return getattr(self._value.parent.attrs, self._value.name).value
        return self._value

    def to_onnx(
        self, name: str, doc_string: Optional[str] = None
    ) -> onnx.AttributeProto:
        """Translate self into an AttributeProto."""
        if isinstance(self._value, Ref):
            return onnx.AttributeProto(
                name=name, ref_attr_name=self._value.name, type=self.attr_proto_type
            )
        if _is_list_attribute(self.value_type):
            assert isinstance(self._value, Iterable)
            inner_type = typing.get_args(self.value_type)[0]
            onnx_value = [_onnx_attr_value(inner_type, value) for value in self._value]
        else:
            onnx_value = _onnx_attr_value(self.value_type, self._value)
        return onnx.helper.make_attribute(name, onnx_value, doc_string)


def from_array(array: numpy.ndarray, name: Optional[str] = None) -> onnx.TensorProto:
    """
    Helper function for converting numpy arrays into TensorProto.
    As it may be useful to name the TensorProto (e.g. in initializers), there is a ``name`` parameter.

    Uses ``numpy.str_`` instead of ``numpy.object_`` for strings, calling ``onnx.numpy_helper.from_array`` internally.
    """
    if array.dtype.type is numpy.str_:
        array = array.astype(numpy.object_)
    return onnx.numpy_helper.from_array(array, name=name)


def to_array(proto: onnx.TensorProto) -> numpy.ndarray:
    """
    Helper function for converting TensorProto values into numpy arrays.
    Note that the name of the TensorProto is lost and should be accessed with ``proto.name`` directly if needed.

    Uses ``numpy.str_`` instead of ``numpy.object_`` for strings, calling ``onnx.numpy_helper.to_array`` internally.
    """
    array = onnx.numpy_helper.to_array(proto)
    if array.dtype.type is numpy.object_:
        array = array.astype(numpy.str_)
    return array


def _cast_attr_value(value_type, value, _traceback_name: str = "?"):
    """Helper function for strongly typing a ``value`` into a given ``value_type``, which may fail."""
    if not isinstance(value_type, type):
        raise TypeError(
            f"Expected {value_type} to be a type, not '{type(value_type).__name__}'. "
            f"-- in '{_traceback_name}'."
        )
    if value_type is str and isinstance(value, bytes):
        return value.decode("utf-8")
    elif value_type is float and isinstance(value, int):
        return float(value)
    elif value_type is numpy.ndarray and isinstance(value, onnx.TensorProto):
        return to_array(value)
    elif value_type is Type and isinstance(value, onnx.TypeProto):
        return Type.from_onnx(value)
    elif value_type is str and isinstance(value, numpy.str_):
        return str(value)
    elif value_type is float and isinstance(value, numpy.floating):
        return float(value)
    elif value_type is int and isinstance(value, numpy.integer):
        return int(value)
    elif value_type is numpy.generic and type(value) is type:
        np_type = numpy.result_type(value).type  # type: ignore
        try:
            Tensor(np_type)
        except TypeError as e:
            raise TypeError(f"{str(e)} -- in '{_traceback_name}'.") from e
        return np_type
    elif (
        value_type is numpy.generic
        and type(value) is type
        and issubclass(value, numpy.generic)
    ):
        return value
    elif value_type is Type and isinstance(value, Type):
        return value
    if type(value) is not value_type:
        value_type_name = value_type.__name__
        if value_type is numpy.generic:
            value_type_name = "numpy.generic"
        raise TypeError(
            f"Could not cast {value!r} of type '{type(value).__name__}' into expected type '{value_type_name}' "
            f"-- in '{_traceback_name}'."
        )
    return value


def _onnx_attr_value(value_type, value):
    """Helper function from converting to an ONNX value from the Python representation."""
    from .graph import Graph

    if value_type is numpy.ndarray:
        return from_array(value)
    elif value_type is Graph:
        # TODO: Move out subgraph build in the general case.
        raise TypeError(
            "Cannot build subgraphs via Attr - use the build_subgraph callback in Node."
        )
    elif value_type is Type:
        return typing.cast(Type, value).to_onnx()
    elif value_type == numpy.generic:
        return Tensor.elem_type_to_onnx(value)
    return value
