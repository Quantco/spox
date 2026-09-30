# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import spox.opset.ai.onnx.v17 as op


def test_integer_overflow_during_shape_inference():
    a = op.const(np.array([1, np.iinfo(np.int64).min], np.int64))
    b = op.const(np.array(1000, np.int64))
    op.mul(a, b)
