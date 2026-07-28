# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from spox._standard import InferenceError
from spox._type_system import Optional, Sequence, Tensor, Type

if TYPE_CHECKING:
    from spox._var import Var


def split_num_outputs(split: Var | None, num_outputs: int | None) -> int:
    """Number of variadic outputs of a ``Split`` node.

    Split accepts *either* the ``split`` input or the ``num_outputs`` attribute,
    but not both. The number of outputs is ``num_outputs`` if given, otherwise
    the (static) length of the ``split`` input.

    Raises
    ------
    InferenceError
        If both or neither of ``split`` and ``num_outputs`` are given, or if the
        number of outputs cannot be determined from a ``split`` input with an
        unknown static length.
    """
    if num_outputs is not None:
        if split is not None:
            raise InferenceError(
                "Only one of the 'split' input or the 'num_outputs' attribute "
                "may be given to Split, not both."
            )
        return num_outputs
    if split is None:
        raise InferenceError(
            "Either the 'split' input or the 'num_outputs' attribute must be "
            "given to Split."
        )
    # ``split`` is a 1-D tensor whose length equals the number of outputs.
    # Prefer the static shape and fall back to a propagated constant value.
    shape = split.type.shape if isinstance(split.type, Tensor) else None
    if shape is not None and len(shape) == 1 and isinstance(shape[0], int):
        return shape[0]
    value = split._value.value if split._value is not None else None
    if isinstance(value, np.ndarray):
        return len(value)
    raise InferenceError(
        "Could not determine the number of Split outputs: the 'split' input has "
        "neither a static length nor a constant value. Provide the 'num_outputs' "
        "attribute instead."
    )


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
