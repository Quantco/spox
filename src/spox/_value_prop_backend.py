# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

import onnx

from ._type_system import Type
from ._value_prop import ORTValue, PropValue, RefValue

TValue = TypeVar("TValue")


class BaseValuePropBackend(ABC, Generic[TValue]):
    @abstractmethod
    def wrap_feed(self, value: PropValue) -> TValue:
        raise NotImplementedError

    @abstractmethod
    def run(
        self, model: onnx.ModelProto, input_feed: dict[str, TValue]
    ) -> dict[str, TValue]:
        raise NotImplementedError

    @abstractmethod
    def unwrap_feed(self, typ: Type, value: TValue) -> PropValue:
        raise NotImplementedError


class ReferenceValuePropBackend(BaseValuePropBackend[RefValue]):
    def wrap_feed(self, value: PropValue) -> RefValue:
        return value.to_ref_value()

    def run(
        self, model: onnx.ModelProto, input_feed: dict[str, RefValue]
    ) -> dict[str, RefValue]:
        try:
            session = onnx.reference.ReferenceEvaluator(model)  # type: ignore
            output_feed = dict(zip(session.output_names, session.run(None, input_feed)))
        except Exception as e:
            # Give up on value propagation if an implementation is missing.
            logging.debug(
                f"Value propagation in {model} on the ONNX reference implementation failed with - "
                f"{type(e).__name__}: {e}"
            )
            return {}
        return output_feed

    def unwrap_feed(self, typ: Type, value: RefValue) -> PropValue:
        return PropValue.from_ref_value(typ, value)


class OnnxruntimeValuePropBackend(BaseValuePropBackend[ORTValue]):
    def wrap_feed(self, value: PropValue) -> ORTValue:
        return value.to_ort_value()

    def run(
        self, model: onnx.ModelProto, input_feed: dict[str, ORTValue]
    ) -> dict[str, ORTValue]:
        import onnxruntime

        try:
            session_options = onnxruntime.SessionOptions()
            session_options.log_severity_level = 3
            session = onnxruntime.InferenceSession(
                model.SerializeToString(), session_options
            )
            output_names = [output.name for output in session.get_outputs()]
            output_feed = dict(zip(output_names, session.run(None, input_feed)))
        except Exception as e:
            logging.debug(
                f"Value propagation in {model} on the onnxruntime failed with - "
                f"{type(e).__name__}: {e}"
            )
            return {}
        return output_feed

    def unwrap_feed(self, typ: Type, value: ORTValue) -> PropValue:
        return PropValue.from_ref_value(typ, value)


_VALUE_PROP_BACKEND: BaseValuePropBackend | None = ReferenceValuePropBackend()


def get_value_prop_backend() -> BaseValuePropBackend | None:
    return _VALUE_PROP_BACKEND


def set_value_prop_backend(backend: BaseValuePropBackend | None) -> None:
    global _VALUE_PROP_BACKEND
    _VALUE_PROP_BACKEND = backend


@contextmanager
def value_prop_backend(backend: BaseValuePropBackend | None) -> Iterator[None]:
    prev_backend = _VALUE_PROP_BACKEND
    set_value_prop_backend(backend)
    yield
    set_value_prop_backend(prev_backend)
