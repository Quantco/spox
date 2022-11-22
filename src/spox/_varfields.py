import typing
from typing import Dict, Iterable, Optional, Sequence, Union

from ._fields import Fields
from ._var import Var, _nil


class VarFields(Fields[Var]):
    """
    Fields subclass used for storing inputs & outputs of Nodes - which are Vars.
    Overrides parent methods to support variadics. Example definitions:

    ``
    class MyFieldsOpt(VarFields, Generic[T1, T2]):
        # Generics are interpreted by the type_inference system
        A: Var[T1]
        B: Optional[Var[T1]]  # marks an optional field of T1

    class MyFieldsVar(VarFields, Generic[T1, T2]):
        A: Var[T1]
        B: Var[T2]
        C: Sequence[Var[T2]]  # marks a variadic field of T2
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
