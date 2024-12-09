# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import Field, dataclass
from typing import Optional, Union, cast, get_type_hints

from ._attributes import Attr
from ._exceptions import InferenceWarning
from ._value_prop import PropDict, PropValue
from ._var import Var, _VarInfo


@dataclass
class BaseFields:
    pass


@dataclass
class BaseAttributes(BaseFields):
    def get_fields(self) -> dict[str, Union[None, Attr]]:
        """Return a mapping of all fields stored in this object by name."""
        return self.__dict__.copy()


class VarFieldKind(enum.Enum):
    """An enumeration of possible kinds of ONNX input/outputs fields."""

    SINGLE = 0
    OPTIONAL = 1
    VARIADIC = 2


class BaseVars:
    """A collection of `Var`-s used to carry around inputs/outputs of nodes"""

    vars: dict[str, Union[Var, Optional[Var], Sequence[Var]]]

    def __init__(self, vars: dict[str, Union[Var, Optional[Var], Sequence[Var]]]):
        self.vars = vars

    def _unpack_to_any(self) -> tuple[Union[Var, Optional[Var], Sequence[Var]], ...]:
        """Unpack the stored fields into a tuple of appropriate length, typed as Any."""
        return tuple(self.vars.values())

    def _flatten(self) -> Iterator[tuple[str, Optional[Var]]]:
        """Iterate over the pairs of names and values of fields in this object."""
        for key, value in self.vars.items():
            if value is None or isinstance(value, Var):
                yield key, value
            else:
                yield from ((f"{key}_{i}", v) for i, v in enumerate(value))

    def flatten_vars(self) -> dict[str, Var]:
        """Return a flat mapping by name of all the VarInfos in this object."""
        return {key: var for key, var in self._flatten() if var is not None}

    def __getattr__(self, attr: str) -> Union[Var, Optional[Var], Sequence[Var]]:
        """Retrieves the attribute if present in the stored variables."""
        try:
            return self.vars[attr]
        except KeyError:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {attr!r}"
            )

    def __setattr__(
        self, attr: str, value: Union[Var, Optional[Var], Sequence[Var]]
    ) -> None:
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
        elif field_type == Optional[_VarInfo]:
            return VarFieldKind.OPTIONAL
        elif field_type == Sequence[_VarInfo]:
            return VarFieldKind.VARIADIC
        raise ValueError(f"Bad field type: '{field.type}'.")

    def _flatten(self) -> Iterable[tuple[str, Optional[_VarInfo]]]:
        """Iterate over the pairs of names and values of fields in this object."""
        for key, value in self.__dict__.items():
            if value is None or isinstance(value, _VarInfo):
                yield key, value
            else:
                yield from ((f"{key}_{i}", v) for i, v in enumerate(value))

    def __iter__(self) -> Iterator[Optional[_VarInfo]]:
        """Iterate over the values of fields in this object."""
        yield from (v for _, v in self._flatten())

    def __len__(self) -> int:
        """Count the number of fields in this object (should be same as declared in the class)."""
        return sum(1 for _ in self)

    def get_var_infos(self) -> dict[str, _VarInfo]:
        """Return a flat mapping by name of all the VarInfos in this object."""
        return {key: var for key, var in self._flatten() if var is not None}

    def get_fields(self) -> dict[str, Union[None, _VarInfo, Sequence[_VarInfo]]]:
        """Return a mapping of all fields stored in this object by name."""
        return self.__dict__.copy()

    @property
    def fully_typed(self) -> bool:
        """Check if all stored variables have a concrete type."""
        return all(
            var.type is not None and var.type._is_concrete
            for var in self.get_var_infos().values()
        )


@dataclass
class BaseInputs(BaseVarInfos):
    def vars(self, prop_values: Optional[PropDict] = None) -> BaseVars:
        if prop_values is None:
            prop_values = {}

        vars_dict: dict[str, Union[Var, Optional[Var], Sequence[Var]]] = {}

        for field in dataclasses.fields(self):
            field_type = self._get_field_type(field)
            field_value = getattr(self, field.name)

            if field_type == VarFieldKind.SINGLE:
                prop_value = cast(PropValue, prop_values.get(field.name, None))
                vars_dict[field.name] = Var(field_value, prop_value)

            elif (
                field_type == VarFieldKind.OPTIONAL
                and prop_values.get(field.name, None) is not None
            ):
                prop_value = cast(PropValue, prop_values.get(field.name, None))
                vars_dict[field.name] = Var(field_value, prop_value)

            elif field_type == VarFieldKind.VARIADIC:
                vars = []

                for i, var_info in enumerate(field_value):
                    var_value = prop_values.get(f"{field.name}_{i}", None)
                    assert isinstance(var_value, PropValue)
                    vars.append(Var(var_info, var_value))

                vars_dict[field.name] = vars

        return BaseVars(vars_dict)


@dataclass
class BaseOutputs(BaseVarInfos):
    def _propagate_vars(self, prop_values: Optional[PropDict] = None) -> BaseVars:
        if prop_values is None:
            prop_values = {}

        def _create_var(key: str, var_info: _VarInfo) -> Var:
            ret = Var(var_info, None)

            if var_info.type is None or key not in prop_values:
                return ret

            prop = PropValue(var_info.type, prop_values.get(key))
            if prop.check():
                ret._value = prop
            else:
                warnings.warn(
                    InferenceWarning(
                        f"Propagated value {prop} does not type-check, dropping. "
                        f"Hint: this indicates a bug with the current value prop backend or type inference."
                    )
                )

            return ret

        ret_dict: dict[str, Union[Var, Optional[Var], Sequence[Var]]] = {}

        for key, var_info in self.__dict__.items():
            if isinstance(var_info, _VarInfo):
                ret_dict[key] = _create_var(key, var_info)
            else:
                ret_dict[key] = [
                    _create_var(f"{key}_{i}", v) for i, v in enumerate(var_info)
                ]

        return BaseVars(ret_dict)
