# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import importlib.metadata

from spox._public import argument, build, inline
from spox._type_system import Optional, Sequence, Tensor, Type
from spox._value_prop_backend import (
    OnnxruntimeValuePropBackend,
    ReferenceValuePropBackend,
    ValuePropBackend,
    get_value_prop_backend,
    set_value_prop_backend,
    value_prop_backend,
)
from spox._var import Var

__all__ = [
    "Var",
    "Type",
    "Tensor",
    "Sequence",
    "Optional",
    "argument",
    "build",
    "inline",
    "get_value_prop_backend",
    "set_value_prop_backend",
    "value_prop_backend",
    "OnnxruntimeValuePropBackend",
    "ValuePropBackend",
    "ReferenceValuePropBackend",
]

try:
    __version__ = importlib.metadata.distribution(__name__).version
except Exception:
    __version__ = "unknown"
