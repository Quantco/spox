import typing
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import onnx

from ._shape import Shape, SimpleShape
from ._utils import dtype_to_tensor_type, tensor_type_to_dtype

T = TypeVar("T")
S = TypeVar("S")


@dataclass(frozen=True)
class Type:
    """Base class for classes representing ONNX types in Spox."""

    @classmethod
    def _from_onnx(cls, proto: onnx.TypeProto) -> "Type":
        """
        Parameters
        ----------
        proto
            Protobuf object to translate from.

        Returns
        -------
        Type
            Respective subtype of Type representing the ONNX type in
            the protobuf object.

        Raises
        ------
        ValueError
            If the passed protobuf does not contain any of the
            expected fields (tensor, sequence, optional).
        """
        if proto.HasField("tensor_type"):
            return Tensor(
                tensor_type_to_dtype(proto.tensor_type.elem_type),
                Shape.from_onnx(proto.tensor_type.shape).to_simple()
                if proto.tensor_type.HasField("shape")
                else None,
            )
        elif proto.HasField("sequence_type"):
            return Sequence(Type._from_onnx(proto.sequence_type.elem_type))
        elif proto.HasField("optional_type"):
            return Optional(Type._from_onnx(proto.optional_type.elem_type))
        raise ValueError(
            f"Cannot get Type from invalid protobuf (not tensor, sequence or optional): {proto}"
        )

    def _assert_concrete(self, *, _traceback_name: str = "?"):
        """Function used by the build process to check if a type is
        well-specified (e.g. Tensor shape is defined).

        Inheritors of ``Type`` should throw if they do not specify enough
        information to be accepted as Model input/outputs.
        """
        return self

    @property
    def _is_concrete(self) -> bool:
        try:
            self._assert_concrete()
        except Exception:
            return False
        else:
            return True

    def unwrap_tensor(self) -> "Tensor":
        """
        Return ``self``, unless this Type is not a Tensor.

        Raises
        ------
        TypeError
            If the type isn't a Tensor.

        """
        if not isinstance(self, Tensor):
            raise TypeError(f"Cannot unwrap requested Tensor type from {self}")
        return self

    def unwrap_sequence(self) -> "Sequence":
        """
        Return ``self``, unless this Type is not a Sequence.

        Raises
        ------
        TypeError
            If the type isn't a Sequence.
        """
        if not isinstance(self, Sequence):
            raise TypeError(f"Cannot unwrap requested Sequence type from {self}")
        return self

    def unwrap_optional(self) -> "Optional":
        """
        Return ``self``, unless this Type is not an Optional.

        Raises
        ------
        TypeError
            If the type isn't an Optional.

        """
        if not isinstance(self, Optional):
            raise TypeError(f"Cannot unwrap requested Optional type from {self}")
        return self

    def _to_onnx(self) -> onnx.TypeProto:
        """Translate ``self`` into an ONNX TypeProto."""
        raise TypeError(
            f"Cannot generate ONNX TypeProto for {self} (not implemented or bad type)."
        )

    def _to_onnx_value_info(
        self,
        name: str,
        doc_string: str = "",
        *,
        concrete: bool = False,
        _traceback_name: str = "?",
    ) -> onnx.ValueInfoProto:
        """Translation of ``self`` into an ONNX ValueInfoProto"""
        if concrete:
            self._assert_concrete(_traceback_name=_traceback_name)
        return onnx.helper.make_value_info(
            name,
            self._to_onnx(),
            doc_string,
        )

    def _subtype(self, other: "Type") -> bool:
        """
        Compare Types for membership.
        An Unknown field (like an unspecified Tensor shape) is treated as "any" in this comparison.
        """
        if not isinstance(other, Type):
            return NotImplemented
        return other == Type() or self == other


@dataclass(frozen=True)
class Tensor(Type):
    """
    Represents a ``Tensor`` of given ``dtype`` and ``shape``.

    The ``dtype`` describes the element type of the ``Tensor``.
    It must correspond to an allowed ONNX tensor element type.

    A shape is denoted with a tuple (simplified) format, where each
    element describes the respective axis.  The types used may be:

    - An ``int`` denoting a statically known length
    - A ``str`` denoting a named runtime-determined length
    - ``None`` representing any length.

    The ``shape`` may also be ``None`` if the rank of the ``Tensor``
    is unknown.

    If you want to specify that dimensions will be equal, you can use
    the same parameter strings.  However, this is not very strictly
    enforced.
    """

    _elem_type: typing.Type[np.generic]
    _shape: Shape

    def __init__(
        self,
        dtype: npt.DTypeLike,
        shape: SimpleShape = None,
    ):
        """Create a ``Tensor``.

        Raises
        ------
        TypeError
            If the passed ``elem_type`` does not correspond to one of
            the following numpy scalar types: ``numpy.bool_``,
            ``numpy.complex128``, ``numpy.complex64``,
            ``numpy.float16``, ``numpy.float32``, ``numpy.float64``,
            ``numpy.int16``, ``numpy.int32``, ``numpy.int64``,
            ``numpy.int8``, ``numpy.str_``, ``numpy.uint16``,
            ``numpy.uint32``, ``numpy.uint64``, ``numpy.uint8``.

        """
        # Try converting to a tensor type. If it fails, we allow the
        # exception to bubble up.
        dtype_to_tensor_type(dtype)
        rich_shape = Shape.from_simple(shape)
        object.__setattr__(self, "_elem_type", np.dtype(dtype).type)
        object.__setattr__(self, "_shape", rich_shape)

    @property
    def dtype(self) -> np.dtype:
        """Data type of this tensor."""
        return np.dtype(self._elem_type)

    @property
    def shape(self) -> SimpleShape:
        """
        Return the shape of this tensor in a simplified/tuple format
        (as used by the ``onnx`` module).  Each element of the
        ``SimpleShape`` tuple denotes information about the respective
        axis.

        Returns
        -------
        SimpleShape
            The shape of this Tensor. If it is unknown, ``None`` is returned,
            otherwise it is a tuple describing each dimension.

        """
        return self._shape.to_simple()

    def _to_onnx(self) -> onnx.TypeProto:
        return onnx.helper.make_tensor_type_proto(
            dtype_to_tensor_type(self._elem_type), self.shape
        )

    def _assert_concrete(self, *, _traceback_name: str = "?"):
        if self.shape is None:
            raise ValueError(
                f"Tensor {self} does not specify the shape -- in {_traceback_name}."
            )
        return self

    def __repr__(self):
        return f"{type(self).__name__}(dtype={self._elem_type.__name__}, shape={self.shape})"

    def __str__(self):
        dims = self.shape
        dims_repr = (
            "".join(f"[{dim if dim is not None else '?'}]" for dim in dims)
            if dims is not None
            else "[...]"
        )
        return f"{self._elem_type.__name__.rstrip('_')}" + dims_repr

    def _subtype(self, other: Type) -> bool:
        if not isinstance(other, Type):
            return NotImplemented
        if other == Type() or self == other:
            return True
        if not isinstance(other, Tensor):
            return False
        return (
            issubclass(self._elem_type, other._elem_type)
            and self._shape <= other._shape
        )


@dataclass(frozen=True)
class Sequence(Type):
    """Ordered collection of elements that are of homogeneous types."""

    elem_type: Type

    def _to_onnx(self) -> onnx.TypeProto:
        return onnx.helper.make_sequence_type_proto(self.elem_type._to_onnx())

    def __repr__(self):
        return f"{type(self).__name__}(elem_type={self.elem_type!r}"

    def __str__(self):
        return f"[{self.elem_type}]"

    def _subtype(self, other: Type) -> bool:
        if not isinstance(other, Type):
            return NotImplemented
        if other == Type() or self == other:
            return True
        if not isinstance(other, Sequence):
            return False
        return self.elem_type._subtype(other.elem_type)


@dataclass(frozen=True)
class Optional(Type):
    """
    Wrapper that may contain an element of :class:`~spox.Tensor` or
    :class:`~spox.Sequence` type, or may be empty (containing none).

    """

    elem_type: Type

    def _to_onnx(self) -> onnx.TypeProto:
        return onnx.helper.make_optional_type_proto(self.elem_type._to_onnx())

    def __repr__(self):
        return f"{type(self).__name__}(elem_type={self.elem_type!r}"

    def __str__(self):
        return f"{self.elem_type}?"

    def _subtype(self, other: Type) -> bool:
        if not isinstance(other, Type):
            return NotImplemented
        if other == Type() or self == other:
            return True
        if not isinstance(other, Optional):
            return False
        return self.elem_type._subtype(other.elem_type)
