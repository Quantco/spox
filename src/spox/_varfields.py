"""
Module implementing a rather elaborate dataclass-like type for storing inputs & outputs, the VarFields.
It will most likely be replaced by something else in the future like attributes were, but for now it is a collection
of methods useful when working with ONNX.
"""

import typing
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    get_type_hints,
)

from ._var import Var, _nil

T_co = TypeVar("T_co", covariant=True)
FieldsT = TypeVar("FieldsT", bound="Fields")
TYPE_HINTS_CACHE = {}


def get_type_hints_cached(obj) -> dict:
    """
    As Fields very often use typing.get_type_hints for introspection when building
    instances, this function caches them based on ID.
    Warning: this means that modifying a Node or Fields subclass after the type hints
    were computed here implicitly requires clearing TYPE_HINTS_CACHE.
    """
    if not isinstance(obj, type):
        obj = type(obj)
    if obj not in TYPE_HINTS_CACHE:
        TYPE_HINTS_CACHE[obj] = get_type_hints(obj)
    return TYPE_HINTS_CACHE[obj]


class Fields(Generic[T_co]):
    """
    Base class for VarFields.
    Fields classes are based on the notion of conveniently defining a signature (like a dataclass),
    but in a way that is closer to ONNX.
    The type hints of only the target class is used (inheritance is not considered).
    """

    @classmethod
    def get_kwargs_types(cls) -> Dict[str, type]:
        """Get expected types for kwargs."""
        return dict(get_type_hints_cached(cls))

    @classmethod
    def get_kwargs(cls) -> List[str]:
        """Get expected names for kwargs."""
        return list(cls.get_kwargs_types())

    @classmethod
    def move_args_into_kwargs(cls, *args, **kwargs) -> Dict[str, Any]:
        """Convenience function for merging args and kwargs into a single kwargs dictionary."""
        kwargs = kwargs.copy()
        for name, value in zip(cls.get_kwargs(), args):
            if name in kwargs:
                raise TypeError(
                    f"Fields.__init__ got multiple values for argument '{name}'"
                )
            kwargs[name] = value
        return kwargs

    def get_types(self) -> Dict[str, typing.Type[T_co]]:
        """
        Get types for fields which are defined in this instance.
        For example, VarFields defines new fields (`var_0, var_1, ...`) for variadic fields,
        even though a different keyword argument is passed in (`var`).
        """
        return typing.cast(Dict[str, typing.Type[T_co]], self.get_kwargs_types())

    def as_dict(self) -> Dict[str, T_co]:
        """
        Get values for fields defined in this instance.
        In particular used for dumping all stored values in `Node.to_onnx`.
        Note that this may have different keys than get_kwargs - see docstring of get_types.
        """
        return {name: getattr(self, name) for name in self.get_kwargs()}

    def unpack(self) -> Tuple:
        """Convenience function for unpacking fields into a tuple of all the base fields."""
        return tuple(getattr(self, name) for name in self.get_kwargs())

    def _unpack_to_any(self) -> Any:
        return self.unpack()

    def __init__(self, *args, **kwargs):
        kwargs = self.move_args_into_kwargs(*args, **kwargs)
        if set(kwargs) > set(self.get_kwargs()):
            raise ValueError(
                f"Can't instantiate {type(self).__name__}, "
                f"unknown kwargs {set(kwargs) - set(self.get_kwargs())}."
            )
        if set(kwargs) < set(self.get_kwargs()):
            raise ValueError(
                f"Can't instantiate {type(self).__name__}, "
                f"missing kwargs {set(self.get_kwargs()) - set(kwargs)}."
            )
        for name, value in list(kwargs.items()):
            setattr(self, name, value)


class VarFields(Fields[Var]):
    """
    Fields subclass used for storing inputs & outputs of Nodes - which are Vars.
    Overrides parent methods to support variadics. Example definitions:

    ``
    class MyFieldsOpt(VarFields):
        A: Var
        B: Optional[Var]

    class MyFieldsVar(VarFields):
        A: Var
        B: Var
        C: Sequence[Var]
    ``

    Note that, especially for VarFields, the order has significance.
    Return values follow the order in which the fields are defined.

    For a variadic field ``var``, the keyword argument of the same name is expected to be a list.
    On creation fields ``var_0, var_1, ...`` are filled with the sequence
    and the field ``var`` stores the list used to instantiate the variadic.

    The hint for the variadic field propagates to all the fields created from it.
    Only the last field may be variadic, same as in ONNX.
    """

    def __init__(self, *_args: Union[Optional[Var], Iterable[Var]], **_kwargs):
        var = self.get_variadic_name()
        # Typecheck for Vars (variadic is a list of Vars)
        kwargs = self.move_args_into_kwargs(*_args, **_kwargs)

        for name in kwargs:
            if kwargs[name] is None:
                if not self.is_optional(name):
                    raise TypeError(
                        f"Field '{name}' is not optional, but a None was passed in for it."
                    )
            elif name == var:
                kwargs[var] = tuple(kwargs[var])
                if not all(isinstance(a, Var) for a in kwargs[var]):
                    raise TypeError(
                        f"Variadic field '{var}' must be an iterable of Vars."
                    )
            else:
                if not isinstance(kwargs[name], Var):
                    raise TypeError(
                        f"Expected field '{name}' to be a `Var`, not {kwargs[name]}."
                    )

        super().__init__(**kwargs)
        if var:
            # Store and create fields for the variadic, as documented above
            setattr(self, var, tuple(getattr(self, var)))
            for key, value in self.get_variadic_values().items():
                setattr(self, key, value)

        val = list(self.as_dict().values())
        if any(first and not second for first, second in zip(val, val[:-1])):
            raise RuntimeError(
                "Only the suffix of optional fields in sequence may be unset."
            )

    @classmethod
    def _is_hinted_with(cls, name: str, wrap) -> bool:
        hint = cls.get_kwargs_types()[name]
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)[0] if typing.get_args(hint) else None
        return bool(
            hint
            and (wrap == origin if args is None else wrap[args] == hint)
            and (args is Var or typing.get_origin(args) is Var)
        )

    @classmethod
    def is_variadic(cls, name: str) -> bool:
        return cls._is_hinted_with(name, Sequence)

    @classmethod
    def is_optional(cls, name: str) -> bool:
        return cls._is_hinted_with(name, Optional)

    @classmethod
    def get_variadic_name(cls) -> Optional[str]:
        """Get the name of the variadic field, if one exists."""
        kwargs = cls.get_kwargs()

        if any(cls.is_variadic(name) for name in kwargs[:-1]):
            raise RuntimeError("Only the last field may be marked variadic.")
        return kwargs[-1] if kwargs and cls.is_variadic(kwargs[-1]) else None

    def get_variadic_values(self) -> Dict[str, Var]:
        """Get the dictionary of field names and their values for the variadic (``var_0, var_1, ...``)."""
        field_name = self.get_variadic_name()
        return (
            {
                f"{field_name}_{i}": var if var else None
                for i, var in enumerate(getattr(self, field_name))
            }
            if field_name is not None
            else {}
        )

    def get_types(self) -> Dict[str, typing.Type[Var]]:
        """
        Overrides the default type list, replacing the variadic field with the sequenced fields.
        The field ``var`` is replaced in the dictionary with ``var_0, var_1, ...`` set to the same type.
        """
        result = super().get_types()
        if var := self.get_variadic_name():
            for key in self.get_variadic_values():
                result[key] = typing.get_args(result[var])[0]
            del result[var]
        return result

    def as_dict(self) -> Dict[str, Var]:
        """Returns all the fields and their values in flat form (variadic sequence in separate fields)."""
        var = self.get_variadic_name()
        dt = super().as_dict()
        if var:
            dt.update(self.get_variadic_values())
            del dt[var]
        return {key: (var if var is not None else _nil) for key, var in dt.items()}

    @property
    def fully_typed(self) -> bool:
        return all(
            var.type is not None and var.type._is_concrete
            for var in self.as_dict().values()
            if var
        )


@typing.final
class NoVars(VarFields):
    pass
