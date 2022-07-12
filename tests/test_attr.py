import pytest


def test_bad_attribute_type_cast_fails(op):
    with pytest.raises(TypeError):
        op.cast(op.const(1), to="abc")  # cast expects a type[numpy.generic], not str


def test_bad_overriden_attribute_type_cast_fails(op):
    with pytest.raises(TypeError):
        op.cast(
            op.const(1), to=object
        )  # cast expects a type[numpy.generic], not Python type
