# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import enum
import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import numpy.typing as npt
import onnx
import onnx.reference

from ._exceptions import InferenceWarning
from ._shape import Shape
from ._type_system import Optional, Sequence, Tensor, Type

"""
The internal representation for runtime values.

- numpy.ndarray -> Tensor
- list[PropValue] -> Sequence
- PropValue -> Optional, Some (has value)
- None -> Optional, Nothing (no value)
"""
PropValueType = Union[np.ndarray, list["PropValue"], "PropValue", None]
PropDict = dict[str, PropValueType]
ORTValue = Union[np.ndarray, list, None]
RefValue = Union[np.ndarray, list, float, None]

VALUE_PROP_STRICT_CHECK: bool = False


class ValuePropBackend(enum.Enum):
    NONE = 0
    REFERENCE = 1
    ONNXRUNTIME = 2


_VALUE_PROP_BACKEND: ValuePropBackend = ValuePropBackend.REFERENCE


@dataclass(frozen=True)
class PropValue:
    """Propagated value given to a VarInfo, which has a run-time value known at compile-time.

    Wrapper for a few Python types which are used to represent values of ONNX types.

    Implements routines for conversion to and from:

    - ONNX Runtime (ORT)
    - Reference implementations (Ref).
    """

    type: Type
    value: PropValueType

    def __post_init__(self) -> None:
        # The underlying numpy array might have been constructed with a
        # platform-dependent dtype - such as ulonglong.
        # Though very similar, it does not compare equal to the usual sized dtype.
        # (for example ulonglong is not uint64)
        if isinstance(self.value, np.ndarray) and np.issubdtype(
            self.value.dtype, np.number
        ):
            # We normalize by reconstructing the dtype through its name
            object.__setattr__(
                self, "value", self.value.astype(np.dtype(self.value.dtype.name))
            )

        if VALUE_PROP_STRICT_CHECK and not self.check():
            raise ValueError(
                f"Attempt to construct PropValue of {self.value}, "
                f"which does not match the expected type {self.type}."
            )

    def __str__(self) -> str:
        return f"<Propagated {self.value}: {self.type}>"

    def check(self) -> bool:
        if isinstance(self.type, Tensor):
            if not (
                isinstance(self.value, np.ndarray)
                and Shape.from_simple(self.value.shape) <= self.type._shape
            ):
                return False
            # Strings need some special handling
            if self.value.dtype == object and self.type.dtype == str:
                return True
            return self.value.dtype.type is self.type.dtype.type
        elif isinstance(self.type, Sequence):
            return isinstance(self.value, list) and all(
                elem.type._subtype(self.type.elem_type) for elem in self.value
            )
        elif isinstance(self.type, Optional):
            return self.value is None or isinstance(self.value, PropValue)
        warnings.warn(
            InferenceWarning(
                f"Unknown or unspecified type for propagated value: {self.type!r}"
            )
        )
        return True

    @classmethod
    def from_ref_value(cls, typ: Type, value: RefValue) -> "PropValue":
        # Sometimes non-Sequence values are wrapped in a list.
        if (
            not isinstance(typ, Sequence)
            and isinstance(value, list)
            and len(value) == 1
        ):
            (value,) = value
        if value is None:  # Optional, Nothing
            return cls(typ, None)
        elif isinstance(typ, Optional):  # Optional, Some
            return cls(typ, cls.from_ref_value(typ.elem_type, value))
        elif isinstance(value, list):  # Sequence
            elem_type = typ.unwrap_sequence().elem_type
            return cls(typ, [cls.from_ref_value(elem_type, elem) for elem in value])
        else:  # otherwise must have Tensor (sometimes this is just a scalar)
            return cls(typ, np.array(value))
        # No fail branch because representations of Tensor are inconsistent

    @classmethod
    def from_ort_value(cls, typ: Type, value: ORTValue) -> "PropValue":
        if value is None:  # Optional, Nothing
            return cls(typ, None)
        elif isinstance(typ, Optional):  # Optional, Some
            return cls(typ, cls.from_ort_value(typ.elem_type, value))
        elif isinstance(value, list):  # Sequence
            elem_type = typ.unwrap_sequence().elem_type
            return cls(typ, [cls.from_ort_value(elem_type, elem) for elem in value])
        elif isinstance(value, np.ndarray):  # Tensor
            # Normalise the dtype in case we got an alias (like longlong)
            if value.dtype == np.dtype(object):
                value = value.astype(str)
            return cls(typ, value)
        raise TypeError(f"No handler for ORT value: {value}")

    def to_ref_value(self) -> RefValue:
        if self.value is None:  # Optional, Nothing
            return None  # Optionals are wrapped in a singleton list
        elif isinstance(self.value, PropValue):  # Optional, Some
            return [self.value.to_ref_value()]
        elif isinstance(self.value, list):  # Sequence
            return [elem.to_ref_value() for elem in self.value]
        else:  # Tensor
            return self.value

    def to_ort_value(self) -> ORTValue:
        if self.value is None:  # Optional, Nothing
            return None
        elif isinstance(self.value, PropValue):  # Optional, Some
            return self.value.to_ref_value()  # type: ignore
        elif isinstance(self.value, list):  # Sequence
            return [elem.to_ref_value() for elem in self.value]
        else:  # Tensor
            return self.value


def _run_reference_implementation(
    model: onnx.ModelProto, input_feed: dict[str, RefValue]
) -> dict[str, RefValue]:
    try:
        session = onnx.reference.ReferenceEvaluator(model)
        output_feed = dict(zip(session.output_names, session.run(None, input_feed)))
    except Exception as e:
        # Give up on value propagation if an implementation is missing.
        logging.debug(
            f"Value propagation in {model} on the ONNX reference implementation failed with - "
            f"{type(e).__name__}: {e}"
        )
        return {}
    return output_feed


def _run_onnxruntime(
    model: onnx.ModelProto, input_feed: dict[str, ORTValue]
) -> dict[str, ORTValue]:
    import onnxruntime

    # Silence possible warnings during execution (especially constant folding)
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    try:
        session = onnxruntime.InferenceSession(model.SerializeToString(), options)
        output_names = [output.name for output in session.get_outputs()]
        output_feed = dict(zip(output_names, session.run(None, input_feed)))
    except Exception as e:
        logging.debug(
            f"Value propagation in {model} on the onnxruntime failed with - "
            f"{type(e).__name__}: {e}"
        )
        return {}
    return output_feed


def get_backend_calls() -> tuple[
    Callable[..., RefValue],
    Callable[..., dict[str, npt.ArrayLike]],
    Callable[..., PropValue],
]:
    wrap_feed: Callable[..., RefValue]
    run: Callable[..., dict[str, npt.ArrayLike]]
    unwrap_feed: Callable[..., PropValue]
    if _VALUE_PROP_BACKEND == ValuePropBackend.REFERENCE:
        wrap_feed = PropValue.to_ref_value
        run = _run_reference_implementation
        unwrap_feed = PropValue.from_ref_value
    elif _VALUE_PROP_BACKEND == ValuePropBackend.ONNXRUNTIME:
        wrap_feed = PropValue.to_ort_value
        run = _run_onnxruntime
        unwrap_feed = PropValue.from_ort_value
    else:
        raise RuntimeError(
            f"Not a valid value propagation backend: {_VALUE_PROP_BACKEND}."
        )
    return wrap_feed, run, unwrap_feed
