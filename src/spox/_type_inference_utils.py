# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from spox._standard import InferenceError
from spox._type_system import Optional, Sequence, Tensor, Type


def loop_erase_shape_info(typ: Type) -> Type:
    """Erases the shape information for a type, that can exists as a state variable in a Loop"""
    if isinstance(typ, Tensor):
        return Tensor(typ.dtype, None)
    elif isinstance(typ, Sequence):
        if not isinstance(typ.elem_type, Tensor):
            raise InferenceError(
                f"Type {typ} not allowed for state variables in Loop operator, sequence element can only be a tensor"
            )
        return Sequence(loop_erase_shape_info(typ.elem_type))
    elif isinstance(typ, Optional):
        if isinstance(typ.elem_type, Optional):
            raise InferenceError(
                f"Type {typ} not allowed for state variables in Loop operator, optionals of optionals are not allowed"
            )
        return Optional(loop_erase_shape_info(typ.elem_type))
    raise InferenceError(
        f"Type {typ} not allowed for state variables in Loop operator."
    )
