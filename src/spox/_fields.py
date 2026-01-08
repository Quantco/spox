# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import Field, dataclass
from typing import get_type_hints

from . import _type_system
from ._attributes import Attr
from ._value_prop import PropDict, PropValue
from ._var import Var, _VarInfo


@dataclass
class BaseFields:
    pass


@dataclass
class BaseAttributes(BaseFields):
    def get_fields(self) -> dict[str, None | Attr]:
        """Return a mapping of all fields stored in this object by name."""
        return self.__dict__.copy()


class VarFieldKind(enum.Enum):
    """An enumeration of possible kinds of ONNX input/outputs fields."""

    SINGLE = 0
    OPTIONAL = 1
    VARIADIC = 2


class BaseVars:
    """A collection of `Var`-s used to carry around inputs/outputs of nodes"""

    vars: dict[str, Var | None | Sequence[Var]]

    def __init__(self, vars: dict[str, Var | None | Sequence[Var]]):
        self.vars = vars

    def _unpack_to_any(self) -> tuple[Var | None | Sequence[Var], ...]:
        """Unpack the stored fields into a tuple of appropriate length, typed as Any."""
        return tuple(self.vars.values())

    def _flatten(self) -> Iterator[tuple[str, Var | None]]:
        """Iterate over the pairs of names and values of fields in this object."""
        for key, value in self.vars.items():
            if value is None or isinstance(value, Var):
                yield key, value
            else:
                yield from ((f"{key}_{i}", v) for i, v in enumerate(value))

    def flatten_vars(self) -> dict[str, Var]:
        """Return a flat mapping by name of all the VarInfos in this object."""
        return {key: var for key, var in self._flatten() if var is not None}

    def __getattr__(self, attr: str) -> Var | None | Sequence[Var]:
        """Retrieves the attribute if present in the stored variables."""
        try:
            return self.vars[attr]
        except KeyError:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {attr!r}"
            )

    def __setattr__(self, attr: str, value: Var | None | Sequence[Var]) -> None:
        """Sets the attribute to a value if the attribute is present in the stored variables."""
        if attr == "vars":
            super().__setattr__(attr, value)
        else:
            self.vars[attr] = value


@dataclass
class BaseVarInfos(BaseFields):
    def __post_init__(self) -> None:
        # Check if passed fields are of the appropriate types based on field kinds
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            field_type = self._get_field_type(field)
            if field_type == VarFieldKind.SINGLE:
                if not isinstance(value, _VarInfo):
                    raise TypeError(f"Field expected VarInfo, got: {type(value)}.")
            elif field_type == VarFieldKind.OPTIONAL:
                if value is not None and not isinstance(value, _VarInfo):
                    raise TypeError(
                        f"Optional must be VarInfo or None, got: {type(value)}."
                    )
            elif field_type == VarFieldKind.VARIADIC:
                if not isinstance(value, Iterable):
                    raise TypeError(
                        f"Variadic field must be iterable, got '{type(value)}'."
                    )
                # Cast to tuple to avoid accidental mutation
                setattr(self, field.name, tuple(value))
                if bad := {type(var) for var in value} - {_VarInfo}:
                    raise TypeError(
                        f"Variadic field must only consist of VarInfos, got: {bad}."
                    )

    @classmethod
    def _get_field_type(cls, field: Field) -> VarFieldKind:
        """Access the kind of the field (single, optional, variadic) based on its type annotation."""
        # The field.type may be unannotated as per
        # from __future__ import annotations
        field_type = get_type_hints(cls)[field.name]
        if field_type == _VarInfo:
            return VarFieldKind.SINGLE
        elif field_type == _VarInfo | None:
            return VarFieldKind.OPTIONAL
        elif field_type == Sequence[_VarInfo]:
            return VarFieldKind.VARIADIC
        raise ValueError(f"Bad field type: '{field.type}'.")

    def _flatten(self) -> Iterable[tuple[str, _VarInfo | None]]:
        """Iterate over the pairs of names and values of fields in this object."""
        for key, value in self.__dict__.items():
            if value is None or isinstance(value, _VarInfo):
                yield key, value
            else:
                yield from ((f"{key}_{i}", v) for i, v in enumerate(value))

    def __iter__(self) -> Iterator[_VarInfo | None]:
        """Iterate over the values of fields in this object."""
        yield from (v for _, v in self._flatten())

    def __len__(self) -> int:
        """Count the number of fields in this object (should be same as declared in the class)."""
        return sum(1 for _ in self)

    def get_var_infos(self) -> dict[str, _VarInfo]:
        """Return a flat mapping by name of all the VarInfos in this object."""
        return {key: var for key, var in self._flatten() if var is not None}

    def get_fields(self) -> dict[str, None | _VarInfo | Sequence[_VarInfo]]:
        """Return a mapping of all fields stored in this object by name."""
        return self.__dict__.copy()

    @property
    def fully_typed(self) -> bool:
        """Check if all stored variables have a concrete type."""
        return all(
            var.type is not None and var.type._is_concrete
            for var in self.get_var_infos().values()
        )

    def into_vars(self, prop_values: PropDict) -> BaseVars:
        """Populate a `BaseVars` object with the propagated values and this object's var_infos"""

        def _create_var(key: str, var_info: _VarInfo) -> Var:
            ret = Var(var_info, None)

            if var_info.type is None or key not in prop_values:
                return ret

            if (
                not isinstance(var_info.type, _type_system.Optional)
                and prop_values[key] is None
            ):
                return ret

            ret._value = PropValue(var_info.type, prop_values[key].value)
            return ret

        ret_dict: dict[str, Var | None | Sequence[Var]] = {}

        for key, var_info in self.__dict__.items():
            if isinstance(var_info, _VarInfo):
                ret_dict[key] = _create_var(key, var_info)
            else:
                ret_dict[key] = [
                    _create_var(f"{key}_{i}", v) for i, v in enumerate(var_info)
                ]

        return BaseVars(ret_dict)


@dataclass
class BaseInputs(BaseVarInfos):
    pass


@dataclass
class BaseOutputs(BaseVarInfos):
    pass
