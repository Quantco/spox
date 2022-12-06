import typing
from typing import Any, Dict, Generic, List, Tuple, TypeVar, get_type_hints

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
