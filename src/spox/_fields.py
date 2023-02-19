import dataclasses
import enum
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

from ._attributes import Attr
from ._var import Var


@dataclass
class BaseFields:
    pass


@dataclass
class BaseAttributes(BaseFields):
    def get_fields(self) -> Dict[str, Union[None, Attr]]:
        """Return a mapping of all fields stored in this object by name."""
        return self.__dict__.copy()


class VarFieldKind(enum.Enum):
    """An enumeration of possible kinds of ONNX input/outputs fields."""

    SINGLE = 0
    OPTIONAL = 1
    VARIADIC = 2


@dataclass
class BaseVars(BaseFields):
    def __post_init__(self):
        # Check if passed fields are of the appropriate types based on field kinds
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            field_type = self._get_field_type(field)
            if field_type == VarFieldKind.SINGLE:
                if not isinstance(value, Var):
                    raise TypeError(f"Field expected Var, got: {type(value)}.")
            elif field_type == VarFieldKind.OPTIONAL:
                if value is not None and not isinstance(value, Var):
                    raise TypeError(
                        f"Optional must be Var or None, got: {type(value)}."
                    )
            elif field_type == VarFieldKind.VARIADIC:
                if not isinstance(value, Iterable):
                    raise TypeError(
                        f"Variadic field must be iterable, got '{type(value)}'."
                    )
                # Cast to tuple to avoid accidental mutation
                setattr(self, field.name, tuple(value))
                if bad := {type(var) for var in value} - {Var}:
                    raise TypeError(
                        f"Variadic field must only consist of Vars, got: {bad}."
                    )

    @classmethod
    def _get_field_type(cls, field) -> VarFieldKind:
        """Access the kind of the field (single, optional, variadic) based on its type annotation."""
        if field.type == Var:
            return VarFieldKind.SINGLE
        elif field.type == Optional[Var]:
            return VarFieldKind.OPTIONAL
        elif field.type == Sequence[Var]:
            return VarFieldKind.VARIADIC
        raise ValueError(f"Bad field type: '{field.type}'.")

    def _flatten(self) -> Iterable[Tuple[str, Optional[Var]]]:
        """Iterate over the pairs of names and values of fields in this object."""
        for key, value in self.__dict__.items():
            if value is None or isinstance(value, Var):
                yield key, value
            else:
                yield from ((f"{key}_{i}", v) for i, v in enumerate(value))

    def __iter__(self) -> Iterator[Optional[Var]]:
        """Iterate over the values of fields in this object."""
        yield from (v for _, v in self._flatten())

    def __len__(self) -> int:
        """Count the number of fields in this object (should be same as declared in the class)."""
        return sum(1 for _ in self)

    def get_vars(self) -> Dict[str, Var]:
        """Return a flat mapping by name of all the Vars in this object."""
        return {key: var for key, var in self._flatten() if var is not None}

    def get_fields(self) -> Dict[str, Union[None, Var, Sequence[Var]]]:
        """Return a mapping of all fields stored in this object by name."""
        return self.__dict__.copy()

    def _unpack_to_any(self) -> Any:
        """Unpack the stored fields into a tuple of appropriate length, typed as Any."""
        return tuple(self.__dict__.values())

    @property
    def fully_typed(self) -> bool:
        """Check if all stored variables have a concrete type."""
        return all(
            var.type is not None and var.type._is_concrete
            for var in self.get_vars().values()
        )


@dataclass
class BaseInputs(BaseVars):
    pass


@dataclass
class BaseOutputs(BaseVars):
    pass
