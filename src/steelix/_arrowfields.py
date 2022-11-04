import typing
from typing import Dict, Iterable, Optional, Sequence, Union

from ._arrow import Arrow, _nil
from ._fields import Fields


class ArrowFields(Fields[Arrow]):
    """
    Fields subclass used for storing inputs & outputs of Nodes - which are Arrows.
    Overrides parent methods to support variadics. Example definitions:

    ``
    class MyFieldsOpt(ArrowFields, Generic[T1, T2]):
        # Generics are interpreted by the type_inference system
        A: Arrow[T1]
        B: Optional[Arrow[T1]]  # marks an optional field of T1

    class MyFieldsVar(ArrowFields, Generic[T1, T2]):
        A: Arrow[T1]
        B: Arrow[T2]
        C: Sequence[Arrow[T2]]  # marks a variadic field of T2
    ``

    Note that, especially for ArrowFields, the order has significance.
    Return values follow the order in which the fields are defined.

    For a variadic field ``var``, the keyword argument of the same name is expected to be a list.
    On creation fields ``var_0, var_1, ...`` are filled with the sequence
    and the field ``var`` stores the list used to instantiate the variadic.

    The hint for the variadic field propagates to all the fields created from it.
    Only the last field may be variadic, same as in ONNX.
    """

    def __init__(self, *_args: Union[Optional[Arrow], Iterable[Arrow]], **_kwargs):
        var = self.get_variadic_name()
        # Typecheck for Arrows (variadic is a list of Arrows)
        kwargs = self.move_args_into_kwargs(*_args, **_kwargs)

        for name in kwargs:
            if kwargs[name] is None:
                if not self.is_optional(name):
                    raise TypeError(
                        f"Field '{name}' is not optional, but a None was passed in for it."
                    )
            elif name == var:
                kwargs[var] = tuple(kwargs[var])
                if not all(isinstance(a, Arrow) for a in kwargs[var]):
                    raise TypeError(
                        f"Variadic field '{var}' must be an iterable of Arrows."
                    )
            else:
                if not isinstance(kwargs[name], Arrow):
                    raise TypeError(
                        f"Expected field '{name}' to be an Arrow, not {kwargs[name]}."
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
            and (args is Arrow or typing.get_origin(args) is Arrow)
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

    def get_variadic_values(self) -> Dict[str, Arrow]:
        """Get the dictionary of field names and their values for the variadic (``var_0, var_1, ...``)."""
        var = self.get_variadic_name()
        return (
            {
                f"{var}_{i}": arrow if var else None
                for i, arrow in enumerate(getattr(self, var))
            }
            if var is not None
            else {}
        )

    def get_types(self) -> Dict[str, typing.Type[Arrow]]:
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

    def as_dict(self) -> Dict[str, Arrow]:
        """Returns all the fields and their values in flat form (variadic sequence in separate fields)."""
        var = self.get_variadic_name()
        dt = super().as_dict()
        if var:
            dt.update(self.get_variadic_values())
            del dt[var]
        return {
            key: (arrow if arrow is not None else _nil) for key, arrow in dt.items()
        }

    @property
    def fully_typed(self) -> bool:
        return all(
            arrow.type is not None and arrow.type.is_concrete
            for arrow in self.as_dict().values()
            if arrow
        )


@typing.final
class NoArrows(ArrowFields):
    pass
