import pytest


@pytest.mark.skip
def test_bad_attribute_type_cast_fails(op):
    with pytest.raises(TypeError):
        op.cast(op.const(1), to="abc")  # cast expects a type[numpy.generic], not str


@pytest.mark.skip
def test_bad_overriden_attribute_type_cast_fails(op):
    with pytest.raises(TypeError):
        op.cast(
            op.const(1), to=object
        )  # cast expects a type[numpy.generic], not Python type
