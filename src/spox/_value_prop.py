# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import onnx
import onnx.reference

from ._type_system import Optional, Sequence, Type

"""
The internal representation for runtime values.

- numpy.ndarray -> Tensor
- list[PropValue] -> Sequence
- PropValue -> Optional, Some (has value)
- None -> Optional, Nothing (no value)
"""
PropDict: TypeAlias = "dict[str, PropValue]"

# ORT and reference currently return the same types, but we might move
# to using onnxruntime.ORTValue objects internally for ORT in the future (for
# better support for non-numpy types)
ORTValue: TypeAlias = "np.ndarray | list[ORTValue] | None"
RefValue: TypeAlias = "np.ndarray | list[RefValue] | None"

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
    value: ORTValue | RefValue

    def __str__(self) -> str:
        return f"<Propagated {self.value}: {self.type}>"

    @classmethod
    def _from_value(
        cls, typ: Type, value: ORTValue | RefValue, producer: ValuePropBackend
    ) -> PropValue:
        # Sometimes non-Sequence values are wrapped in a list.
        if (
            producer == ValuePropBackend.REFERENCE
            and not isinstance(typ, Sequence)
            and isinstance(value, list)
            and len(value) == 1
        ):
            (value,) = value
        if isinstance(typ, Optional):
            if value is None:
                return cls(typ, None)
            return cls(typ, cls.from_ref_value(typ.elem_type, value).value)
        if isinstance(typ, Sequence):
            if not isinstance(value, list):
                raise TypeError(f"expected 'list', got `{type(value)}`")
            elem_type = typ.unwrap_sequence().elem_type
            return cls(
                typ, [cls.from_ref_value(elem_type, elem).value for elem in value]
            )
        tensor = typ.unwrap_tensor()

        if not isinstance(value, np.ndarray):
            raise TypeError(f"expected 'numpy.ndarray' got, `{type(value)}`")

        # Normalise the dtype in case we got an alias (like longlong)
        if value.dtype == np.dtype(object):
            value = value.astype(str)
        # The underlying numpy array might have been constructed with a
        # platform-dependent dtype - such as ulonglong.
        # Though very similar, it does not compare equal to the usual sized dtype.
        # (for example ulonglong is not uint64)
        if np.issubdtype(value.dtype, np.number):
            # We normalize by reconstructing the dtype through its name
            value = value.astype(np.dtype(value.dtype.name))

        if tensor.dtype.kind != "U" and value.dtype != tensor.dtype:
            # Skip dtype check for string arrays
            raise TypeError(
                f"expected propagated value of dtype `{tensor.dtype}`, got `{value.dtype}`"
            )
        if tensor.shape is not None:
            expected_rank = len(tensor.shape)
            if expected_rank != value.ndim:
                raise ValueError(
                    f"expected propagated value of rank `{expected_rank}`, got `{value.ndim}`"
                )

        return cls(typ, value)

    @classmethod
    def from_ref_value(cls, typ: Type, value: RefValue) -> PropValue:
        return cls._from_value(typ, value, ValuePropBackend.REFERENCE)

    @classmethod
    def from_ort_value(cls, typ: Type, value: ORTValue) -> PropValue:
        return cls._from_value(typ, value, ValuePropBackend.ONNXRUNTIME)

    def to_ref_value(self) -> RefValue:
        return self.value

    def to_ort_value(self) -> ORTValue:
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
            "value propagation on the ONNX reference implementation failed with - "
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
            "value propagation on the onnxruntime failed with - "
            f"{type(e).__name__}: {e}"
        )
        return {}
    return output_feed


def infer(model: onnx.ModelProto, input_feed: PropDict) -> PropDict:
    if _VALUE_PROP_BACKEND == ValuePropBackend.NONE:
        return {}

    if _VALUE_PROP_BACKEND == ValuePropBackend.ONNXRUNTIME:
        run = _run_onnxruntime
        make_prop_value = PropValue.from_ort_value
    elif _VALUE_PROP_BACKEND == ValuePropBackend.REFERENCE:
        run = _run_reference_implementation
        make_prop_value = PropValue.from_ref_value
    else:
        raise ValueError(f"Unexpected backend: `{_VALUE_PROP_BACKEND}`")
    out_types = {info.name: Type._from_onnx(info.type) for info in model.graph.output}
    res = run(model, {k: v.value for k, v in input_feed.items()})

    try:
        return {k: make_prop_value(out_types[k], v) for k, v in res.items()}
    except Exception as e:
        logging.debug(
            f"propagated values had unexpected type - {type(e).__name__}: {e}"
        )
        return {}
