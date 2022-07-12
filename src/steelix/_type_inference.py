"""
Implements functions related to Python-side type inference.
This is mostly kept for the case custom operators are implemented, as StandardOpNode uses ONNX inference.
The general gist is that input/output Arrows are marked with generics, like ``Arrow[T]``.
Then based on the InferenceFlag for a given type variable different matching behaviour is employed,
like broadcasting with ``InferenceFlag.BROADCAST`` (the default being only matching Tensor element types).
"""

import enum
import functools
import typing
import warnings
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple, TypeVar, Union

import onnx.shape_inference
from typing_extensions import Annotated

from .shape import Shape
from .type_system import Tensor, Type

InferenceError = onnx.shape_inference.InferenceError
ValidationError = onnx.checker.ValidationError


class InferenceWarning(Warning):
    pass


class InferenceFlag(enum.Enum):
    """Enumeration for flags of type inference behaviour."""

    STRICT = 1
    BROADCAST = 2


def get_hint(annotation, expect_origin: Optional[Iterable[Any]] = None):
    """
    Extract the hint from Fields annotations, for type inference purposes.
    Handles:

    - Annotated[Arrow, ConstantType] => ConstantType  # Broken on Python 3.8!
    - Arrow[T], Attr[T] => T.

    If ``expect_origin`` is not None, also checks if the ``typing.get_origin``
    of the annotation belongs to ``expect_origin``.
    """
    args = typing.get_args(annotation)
    if expect_origin is not None and typing.get_origin(annotation) not in expect_origin:
        raise TypeError(
            f"Type hint {annotation} was expected to have origin {expect_origin}, "
            f"but had {typing.get_origin(annotation)}."
        )
    if typing.get_origin(annotation) is Annotated:
        _, hint = args
    elif len(args) == 0:
        hint = None
    elif len(args) == 1:
        (hint,) = args
    else:
        hint = args
    return hint


def weak_type_eq(first: Type, second: Type) -> bool:
    """Checks if types are equivalent, but for Tensors only element types are compared."""
    if isinstance(first, Tensor) and isinstance(second, Tensor):
        return first.elem_type == second.elem_type
    return first == second


def _resolve_generic(
    types_opt: Iterable[Tuple[str, Optional[Type]]],
    flags: Set[InferenceFlag],
    op_name: str,
    in_var: TypeVar,
) -> Optional[Type]:
    """
    For a generic of given assignments and given inference flags, find a general type to assign to it.
    For example `InferenceFlag.BROADCAST` checks all element types are equal and broadcasts all the existing shapes.
    """
    types_opt = list(types_opt)
    if any(t is None for _, t in types_opt):
        return None
    types = typing.cast(List[Tuple[str, Type]], types_opt)
    if InferenceFlag.STRICT in flags:
        init_name, init_type = types[0]
        for name, in_type in types:
            if init_type != in_type:
                raise InferenceError(
                    f"Invalid types in operator {op_name}, generic {in_var} - "
                    f"mismatched types {init_name} of {init_type} != {name} of {in_type}."
                )
    elif InferenceFlag.BROADCAST in flags:
        tensors = []
        for name, in_type in types:
            if not isinstance(in_type, Tensor):
                raise InferenceError(
                    f"Invalid types in operator {op_name}, generic {in_var} - "
                    f"expected broadcast targets to be Tensors, found {name} of {in_type}."
                )
            tensors.append((name, in_type))
        init_name, init_elem_type = tensors[0][0], tensors[0][1].elem_type
        for name, tensor in tensors:
            if init_elem_type != tensor.elem_type:
                raise InferenceError(
                    f"Invalid types in operator {op_name}, generic {in_var} - "
                    f"broadcast-mismatched element types "
                    f"{init_name} of {init_elem_type} != {name} of {tensor.elem_type}."
                )
        shape = functools.reduce(
            Shape.broadcast, (tensor.shape for _, tensor in tensors)
        )
        return Tensor(init_elem_type, shape)
    init_name, init_type = types[0]
    for name, in_type in types:
        if not weak_type_eq(init_type, in_type):
            raise InferenceError(
                f"Invalid types in operator {op_name}, generic {in_var} - "
                f"weakly-mismatched types {init_name} of {init_type} != {name} of {in_type}."
            )
    return init_type


def _warn_unknown_types(
    value_type: Optional[Type], name: str, op_name: str, init_stack_level: int = 5
) -> bool:
    if value_type is None:
        warnings.warn(
            InferenceWarning(
                f"Unknown Arrow I/O type for {name} in operator {op_name}."
            ),
            stacklevel=init_stack_level,
        )
        return True
    try:
        value_type.assert_concrete()
    except Exception as e:
        warnings.warn(
            InferenceWarning(
                f"Arrow I/O type for {name} in {op_name} was not concrete, failing with - "
                f"{type(e).__name__}: {str(e)}"
            ),
            stacklevel=init_stack_level,
        )
        return True
    return False


def _run_type_checks(
    checks: Iterable[Tuple[str, Optional[Union[TypeVar, Type]], Type]],
    member_constraints: Callable[[TypeVar], Optional[Set[Type]]],
    op_name: str,
):
    """Check that values and types have been typed in a way compatible with the definition."""
    for name, hint, value_type in checks:
        if hint is None:
            pass
        elif isinstance(hint, Type):
            if not (value_type <= hint):
                raise InferenceError(
                    f"Invalid types in {op_name} - "
                    f"unsatisfied requirement {name} of {value_type} <= {hint}"
                )
        elif isinstance(hint, TypeVar):
            constraints = member_constraints(hint)
            if constraints is not None and not any(
                value_type <= cons for cons in constraints
            ):
                raise InferenceError(
                    f"Invalid types in {op_name} - "
                    f"unsatisfied requirement {name} of {value_type} <= any {constraints}"
                )
        else:
            raise RuntimeError(
                f"Unknown hint in {op_name} - {name} of {hint} ({type(hint)})."
            )
