"""
Data representation for Tensor shapes. Largely attempts to mimic semantics of ONNX shapes.
Shapes have 3 representations:

- ONNX, composed of the member of TensorProto: TensorShapeProto with members TensorShapeProto.Dimension
- Simplified, also used by ONNX helper and convenient for printing and using in Python. This is SimpleShape in the code,
  and it's a tuple of mixed types (int, str, None).
- Class, implemented by Natural and Shape, which also exposes some useful methods for conversion.

Shapes may be unknown (when the rank is unknown, unknown dimensions are OK), but this cannot be used in many contexts.
Unknown shapes may cause warnings or errors to be raised.
"""

import abc
import typing
from dataclasses import dataclass
from typing import Optional, Tuple, TypeVar, Union

import onnx

SimpleShapeElem = Union[str, int, None]
SimpleShape = Optional[Tuple[SimpleShapeElem, ...]]


class ShapeError(TypeError):
    pass


@dataclass(frozen=True)
class Natural(abc.ABC):
    """Family of types used for storing known or unknown natural numbers, primarily for tensor shapes."""

    @classmethod
    def from_simple(cls, value: SimpleShapeElem) -> "Natural":
        """Translate into Natural from simplified representation."""
        if isinstance(value, int):
            return Constant(value)
        elif isinstance(value, str):
            return Unknown(value)
        elif value is None:
            return Unknown()
        raise TypeError(
            f"Cannot translate to Natural from simple shape element {value}."
        )

    @classmethod
    def simple_from_onnx(
        cls, proto: onnx.TensorShapeProto.Dimension
    ) -> SimpleShapeElem:
        """Translate an ONNX dimension element into the simplified representation."""
        if proto.HasField("dim_value"):
            return int(proto.dim_value)
        elif proto.HasField("dim_param"):
            return str(proto.dim_param)
        else:
            return None

    @classmethod
    def simple_to_onnx(cls, value: SimpleShapeElem) -> onnx.TensorShapeProto.Dimension:
        """Translate simplified representation element into an ONNX one."""
        if isinstance(value, int):
            return onnx.TensorShapeProto.Dimension(dim_value=value)
        if isinstance(value, str):
            return onnx.TensorShapeProto.Dimension(dim_param=value)
        elif value is None:
            return onnx.TensorShapeProto.Dimension()
        raise TypeError(f"Cannot translate to ONNX from simple shape element {value}.")

    def to_onnx(self) -> onnx.TensorShapeProto.Dimension:
        """Translate self into ONNX representation."""
        raise self.simple_to_onnx(self.to_simple())

    @classmethod
    def from_onnx(cls, proto: onnx.TensorShapeProto.Dimension) -> "Natural":
        """Translate Natural from ONNX dimension element."""
        return cls.from_simple(cls.simple_from_onnx(proto))

    def to_simple(self) -> SimpleShapeElem:
        """Translate self into simplified representation."""
        raise NotImplementedError(f"Cannot translate {self} to simple shape element.")

    def __le__(self, other: "Natural") -> bool:
        """Shape dimension membership comparison, with Unknown serving as an "any" quantifier."""
        if not isinstance(other, Natural):
            return NotImplemented
        return other == Unknown()


@dataclass(frozen=True)
class Unknown(Natural):
    """Represents an unknown natural number. Unknown passes any dimension constraint."""

    label: str = ""

    def to_simple(self) -> Union[str, None]:
        return None if not self.label else self.label

    def __le__(self, other: Natural) -> bool:
        if not isinstance(other, Natural):
            return NotImplemented
        return True


@dataclass(frozen=True)
class Constant(Natural):
    """Represents a given constant natural number."""

    n: int

    def to_simple(self) -> int:
        return self.n

    def __le__(self, other: Natural) -> bool:
        if not isinstance(other, Natural):
            return NotImplemented
        return isinstance(other, Unknown) or self == other


ShapeT = TypeVar("ShapeT", bound="Shape")


@dataclass(frozen=True)
class Shape:
    """Type representing a static Tensor shape."""

    dims: Optional[Tuple[Natural, ...]]

    def __bool__(self):
        return self.dims is not None

    @classmethod
    def from_simple(cls: typing.Type[ShapeT], shape: SimpleShape) -> ShapeT:
        """Translate into a Shape from the simplified representation."""
        return cls(
            tuple(Natural.from_simple(v) for v in shape) if shape is not None else None
        )

    @classmethod
    def from_onnx(
        cls: typing.Type[ShapeT], proto: Optional[onnx.TensorShapeProto]
    ) -> ShapeT:
        """Translate into a Shape from ONNX shape."""
        return (
            cls(tuple(Natural.from_onnx(dim) for dim in proto.dim))
            if proto is not None
            else cls(None)
        )

    def to_simple(self) -> SimpleShape:
        """Translate into the simplified representation."""
        return (
            tuple(v.to_simple() for v in self.dims) if self.dims is not None else None
        )

    def to_onnx(self) -> Optional[onnx.TensorShapeProto]:
        """Translate into the ONNX representation."""
        if self.dims is None:
            return None
        proto = onnx.TensorShapeProto(dim=(v.to_onnx() for v in self.dims))
        proto.dim.extend(
            []
        )  # Make sure that the field is set at least with an empty list.
        return proto

    @property
    def maybe_rank(self) -> Optional[int]:
        """Get the rank of this Shape, or None if it is unknown."""
        return len(self.dims) if self.dims is not None else None

    @property
    def rank(self) -> int:
        r = self.maybe_rank
        if r is None:
            raise ShapeError(f"Rank of {self} is unknown.")
        return r

    def __getitem__(self, item) -> Union["Shape", Natural]:
        """Indexing the dimensions, also provides iteration."""
        if self.dims is None:
            raise ShapeError(f"Cannot index unknown {self}.")
        elif isinstance(item, slice):
            return Shape(self.dims[item])
        return self.dims[item]

    def can_broadcast(self, other: "Shape") -> bool:
        """Check if this shape can be broadcast with ``other``."""
        try:
            self.broadcast(other)
        except ShapeError:
            return False
        else:
            return True

    def broadcast(self, other: Union["Shape", SimpleShape]) -> "Shape":
        """Return the result of shape broadcasting on ``self`` and ``other``."""
        if not isinstance(other, Shape):
            other = Shape.from_simple(other)
        a, b = self.to_simple(), other.to_simple()
        if a is None or b is None:
            return Shape(None)
        if len(a) > len(b):  # w.l.o.g. rank a < rank b
            a, b = b, a
        a = (1,) * (len(b) - len(a)) + a  # prepend with 1
        try:
            return Shape.from_simple(tuple(_broadcast_elem(x, y) for x, y in zip(a, b)))
        except ShapeError as e:
            raise ShapeError(
                f"Could not broadcast shapes: {self.to_simple()}, {other.to_simple()}."
            ) from e

    def __le__(self, other: "Shape") -> bool:
        """Shape membership comparison. Unknown shapes are treated as "any" qualifiers."""
        if not isinstance(other, Shape):
            return NotImplemented
        elif self.dims is None or other.dims is None:
            return True
        elif self.rank != other.rank:
            return False
        return all(x <= y for x, y in zip(self.dims, other.dims))


def _broadcast_elem(x: SimpleShapeElem, y: SimpleShapeElem) -> SimpleShapeElem:
    """
    Utility function for getting the broadcasting result over one dimension.
    Note that this is a very relaxed check, as unknowns may equal any constant or unknown (or they may be 1).
    Only constant (integer) dimensions are compared strictly.
    """
    if x == y:
        return x
    if x == 1:
        return y
    if y == 1:
        return x
    if isinstance(x, int) and isinstance(y, int):
        if x != y:
            raise ShapeError(
                f"Could not broadcast different constant dimensions: {x}, {y}."
            )
        return x
    if isinstance(x, int):
        return x
    if isinstance(y, int):
        return y
    return None
