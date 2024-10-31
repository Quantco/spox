# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

from ._attributes import Attr
from ._exceptions import InferenceWarning
from ._value_prop import PropValue
from ._var import Var, VarInfo


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


@dataclass
class BaseVarInfos(BaseFields):
    def __post_init__(self):
        # Check if passed fields are of the appropriate types based on field kinds
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            field_type = self._get_field_type(field)
            if field_type == VarFieldKind.SINGLE:
                if not isinstance(value, VarInfo):
                    raise TypeError(f"Field expected VarInfo, got: {type(value)}.")
            elif field_type == VarFieldKind.OPTIONAL:
                if value is not None and not isinstance(value, VarInfo):
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
                if bad := {type(var) for var in value} - {VarInfo}:
                    raise TypeError(
                        f"Variadic field must only consist of VarInfos, got: {bad}."
                    )

    @classmethod
    def _get_field_type(cls, field) -> VarFieldKind:
        """Access the kind of the field (single, optional, variadic) based on its type annotation."""
        if field.type == VarInfo:
            return VarFieldKind.SINGLE
        elif field.type == Optional[VarInfo]:
            return VarFieldKind.OPTIONAL
        elif field.type == Sequence[VarInfo]:
            return VarFieldKind.VARIADIC
        raise ValueError(f"Bad field type: '{field.type}'.")

    def _flatten(self) -> Iterable[tuple[str, Optional[VarInfo]]]:
        """Iterate over the pairs of names and values of fields in this object."""
        for key, value in self.__dict__.items():
            if value is None or isinstance(value, VarInfo):
                yield key, value
            else:
                yield from ((f"{key}_{i}", v) for i, v in enumerate(value))

    def __iter__(self) -> Iterator[Optional[VarInfo]]:
        """Iterate over the values of fields in this object."""
        yield from (v for _, v in self._flatten())

    def __len__(self) -> int:
        """Count the number of fields in this object (should be same as declared in the class)."""
        return sum(1 for _ in self)

    def get_vars(self) -> dict[str, VarInfo]:
        """Return a flat mapping by name of all the VarInfos in this object."""
        return {key: var for key, var in self._flatten() if var is not None}

    def get_fields(self) -> dict[str, Union[None, VarInfo, Sequence[VarInfo]]]:
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
class BaseInputs(BaseVarInfos):
    pass


@dataclass
class BaseOutputs(BaseVarInfos):
    def _propagate_vars(
        self,
        prop_values={},
        flatten_variadic=False,
    ):
        def _create_var(key, var_info):
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

        ret_dict = {}

        for key, var_info in self.__dict__.items():
            if var_info is None or isinstance(var_info, VarInfo):
                ret_dict[key] = _create_var(key, var_info)
            elif flatten_variadic:
                for i, v in enumerate(var_info):
                    ret_dict[f"{key}_{i}"] = _create_var(f"{key}_{i}", v)
            else:
                ret_dict[key] = [
                    _create_var(f"{key}_{i}", v) for i, v in enumerate(var_info)
                ]

        return ret_dict
