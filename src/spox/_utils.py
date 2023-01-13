from typing import Optional

import numpy as np
import numpy.typing as npt
import onnx
from onnx import TensorProto


def tensor_type_to_dtype(ttype: int) -> np.dtype:
    """Convert integer tensor types to ``numpy.dtype`` objects."""
    if ttype == onnx.TensorProto.STRING:
        return np.dtype(str)  # Spox uses the str datatype for strings, not object
    return onnx.helper.tensor_dtype_to_np_dtype(ttype)


def dtype_to_tensor_type(dtype_like: npt.DTypeLike) -> int:
    """Convert numpy data types into integer tensor types.

    Raises
    ------
    TypeError:
        If ``dtype_like`` has no corresponding tensor type in the ONNX standard.
    """
    err_msg = f"{dtype_like} is not a valid ONNX tensor element type."
    if dtype_like is None:  # numpy would default to float64
        raise TypeError(err_msg)
    # normalize in the case of aliases like ``long`` which are missing in the lookup
    dtype = np.dtype(np.dtype(dtype_like).type)
    if dtype == np.dtype(object):
        raise TypeError(
            "`np.dtype('object')` is not supported as a tensor element type. "
            "Hint: Spox uses `np.dtype('str')` for the string datatype."
        )
    elif dtype == np.dtype(str):
        return onnx.TensorProto.STRING
    try:
        return onnx.helper.np_dtype_to_tensor_dtype(dtype)
    except KeyError:
        raise TypeError(err_msg)


def from_array(arr: np.ndarray, name: Optional[str] = None) -> TensorProto:
    """Convert the given ``numpy.array`` into an ``onnx.TensorProto``.

    As it may be useful to name the TensorProto (e.g. in
    initializers), there is a ``name`` parameter.

    This function differs from ``onnx.numpy_helper.from_array`` by not
    using the ``raw_data`` field.
    """
    cast_to_bytes = False
    if arr.dtype.type in [np.str_, np.object_]:
        cast_to_bytes = True
    return onnx.helper.make_tensor(
        name=name or "",
        data_type=dtype_to_tensor_type(arr.dtype),
        dims=arr.shape,
        # make_tensor fails on scalars. We fix it by calling flatten
        vals=(
            np.char.encode(arr, encoding="utf-8") if cast_to_bytes else arr
        ).flatten(),
        raw=False,
    )
