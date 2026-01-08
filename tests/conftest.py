# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import sys

import ml_dtypes
import numpy as np
import onnxruntime
import pytest

from spox import _debug, _value_prop
from spox._future import set_type_warning_level
from spox._graph import Graph
from spox._node import TypeWarningLevel

set_type_warning_level(TypeWarningLevel.CRITICAL)
_value_prop.VALUE_PROP_STRICT_CHECK = True
_debug.STORE_TRACEBACK = True


class ONNXRuntimeHelper:
    _build_cache: dict[Graph, bytes]
    _last_graph: Graph | None
    _last_session: onnxruntime.InferenceSession | None

    def __init__(self):
        self._build_cache = {}
        self._last_graph = None
        self._last_session = None

    def run(self, graph: Graph, unwrap: str | None = None, **kwargs):
        debug_index = {
            **graph._get_build_result().scope.var.of_name,
            **graph._get_build_result().scope.node.of_name,
        }
        if self._last_graph is graph:
            print("[ONNXRuntimeHelper] Reusing previous session.", file=sys.stderr)
            session = self._last_session
        else:
            if graph not in self._build_cache:
                model_proto = graph.to_onnx_model()
                model_bytes = model_proto.SerializeToString()
                self._build_cache[graph] = model_bytes
            else:
                model_bytes = self._build_cache[graph]
            with _debug.show_construction_tracebacks(debug_index):
                session = onnxruntime.InferenceSession(model_bytes)
        self._last_graph = graph
        self._last_session = session
        assert isinstance(session, onnxruntime.InferenceSession)
        with _debug.show_construction_tracebacks(debug_index):
            result = {
                output.name: result
                for output, result in zip(
                    session.get_outputs(), session.run(None, kwargs)
                )
            }
        if unwrap is not None:
            return result[unwrap]
        return result

    @staticmethod
    def assert_close(given, expected, rtol=1e-7):
        if given is None:
            assert expected is None
        else:
            if isinstance(given, list):
                for subarray in given:
                    np.testing.assert_allclose(
                        given, np.array(expected, dtype=subarray.dtype), rtol=rtol
                    )
            else:
                np.testing.assert_allclose(
                    given, np.array(expected, dtype=given.dtype), rtol=rtol
                )


@pytest.fixture(scope="session")
def onnx_helper():
    return ONNXRuntimeHelper()


@pytest.fixture(
    params=[
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.bool_,
        np.str_,
        ml_dtypes.bfloat16,
        ml_dtypes.float8_e4m3fn,
        ml_dtypes.float8_e4m3fnuz,
        ml_dtypes.float8_e5m2,
        ml_dtypes.float8_e5m2fnuz,
        ml_dtypes.uint4,
        ml_dtypes.int4,
        # TODO: Add after the opset 23 is supported by the conda-forge onnxruntime release
        # ml_dtypes.float4_e2m1fn,
        # ml_dtypes.float8_e8m0fnu
    ],
)
def dtype(request: pytest.FixtureRequest) -> np.dtype:
    """Fixture of all supported data types."""
    return np.dtype(request.param)
