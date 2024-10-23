# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from typing import Any

import numpy as np
import pytest

from spox._future import initializer

TESTED_INITIALIZER_ROWS: list[list[Any]] = [
    [0, 1, 2],
    [0.0, 1.0, 2.0],
    [np.float16(3.14), np.float16(5.3)],
    ["abc", "def"],
    [True, False],
]


def assert_expected_initializer(var, value):
    np.testing.assert_equal(var._get_value(), np.array(value))
    assert var.unwrap_tensor().dtype.type == np.array(value).dtype.type
    assert var.unwrap_tensor().shape == np.array(value).shape


@pytest.mark.parametrize(
    "value", itertools.chain.from_iterable(TESTED_INITIALIZER_ROWS)
)
def test_initializer_scalar(value):
    assert_expected_initializer(initializer(value), value)


@pytest.mark.parametrize("row", TESTED_INITIALIZER_ROWS)
def test_initializer_iter(row):
    assert_expected_initializer(initializer(row), row)


@pytest.mark.parametrize("row", TESTED_INITIALIZER_ROWS)
def test_initializer_matrix(row):
    assert_expected_initializer(initializer([row, row]), [row, row])
