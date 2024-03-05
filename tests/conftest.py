import sys
from typing import Dict, Optional

import numpy
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
    _build_cache: Dict[Graph, bytes]
    _last_graph: Optional[Graph]
    _last_session: Optional[onnxruntime.InferenceSession]

    def __init__(self):
        self._build_cache = {}
        self._last_graph = None
        self._last_session = None

    def run(self, graph: Graph, unwrap: Optional[str] = None, **kwargs):
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
                    numpy.testing.assert_allclose(
                        given, numpy.array(expected, dtype=subarray.dtype), rtol=rtol
                    )
            else:
                numpy.testing.assert_allclose(
                    given, numpy.array(expected, dtype=given.dtype), rtol=rtol
                )


@pytest.fixture(scope="session")
def onnx_helper():
    return ONNXRuntimeHelper()
