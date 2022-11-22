import warnings
from typing import Optional

import onnx.shape_inference

from ._type_system import Type

InferenceError = onnx.shape_inference.InferenceError
ValidationError = onnx.checker.ValidationError


class InferenceWarning(Warning):
    pass


def _warn_unknown_types(
    value_type: Optional[Type], name: str, op_name: str, init_stack_level: int = 5
) -> bool:
    if value_type is None:
        warnings.warn(
            InferenceWarning(f"Unknown type for {name} in operator {op_name}."),
            stacklevel=init_stack_level,
        )
        return True
    try:
        value_type._assert_concrete()
    except Exception as e:
        warnings.warn(
            InferenceWarning(
                f"Var I/O type for {name} in {op_name} was not concrete, failing with - "
                f"{type(e).__name__}: {str(e)}"
            ),
            stacklevel=init_stack_level,
        )
        return True
    return False
