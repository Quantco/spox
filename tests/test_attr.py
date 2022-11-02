import pytest


def test_bad_attribute_type_cast_fails(op):
    with pytest.raises(TypeError):
        # to must be an value which can be handled by `np.dtype`.
        op.cast(op.const(1), to="abc")


def test_cast_with_build_in_type(op):
    # Use python build in types
    with pytest.raises(TypeError):
        op.cast(op.const(1), to=str)


def test_float_instead_of_int_attr(op):
    with pytest.raises(TypeError):
        op.concat([op.const(1)], axis=3.14)
