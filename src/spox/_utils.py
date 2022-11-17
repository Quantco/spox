from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from onnx import TensorProto
from onnx.helper import make_tensor, mapping

_DTYPE_TO_TENSOR_TYPE: Dict[np.dtype, int] = {
    **{
        dtype: ttype
        for dtype, ttype in mapping.NP_TYPE_TO_TENSOR_TYPE.items()
        if dtype != np.object_
    },
    np.dtype(str): TensorProto.STRING,
}

_TENSOR_TYPE_TO_DTYPE = {ttype: dtype for dtype, ttype in _DTYPE_TO_TENSOR_TYPE.items()}


def tensor_type_to_dtype(ttype: int) -> np.dtype:
    """Convert integer tensor types to ``numpy.dtype`` objects."""
    return _TENSOR_TYPE_TO_DTYPE[ttype]


def dtype_to_tensor_type(dtype_like: npt.DTypeLike) -> int:
    """Convert numpy data types into integer tensor types.

    Raises
    ------
    TypeError:
        If ``dtype_like`` has no corresponding tensor type in the ONNX
        standard.
    """
    err_msg = f"{dtype_like} has no corresponding tensor type in the ONNX standard."
    if dtype_like is None:
        # Numpy defaults implicitly to float64. I don't think we want
        # to do the same?
        raise TypeError(err_msg)
    # normalize string data types
    dtype = np.dtype(np.dtype(dtype_like).type)
    try:
        return _DTYPE_TO_TENSOR_TYPE[dtype]
    except KeyError:
        raise TypeError(err_msg)


def from_array(arr: np.ndarray, name: Optional[str] = None) -> TensorProto:
    """Convert the given ``numpy.array`` into a ``onnx.TensorProto``.

    As it may be useful to name the TensorProto (e.g. in
    initializers), there is a ``name`` parameter.

    This function differs from ``onnx.numpy_helper.from_array`` by not
    using the ``raw_data`` field.
    """
    cast_to_bytes = False
    if arr.dtype.type in [np.str_, np.object_]:
        cast_to_bytes = True
    return make_tensor(
        name=name or "",
        data_type=dtype_to_tensor_type(arr.dtype),
        dims=arr.shape,
        # make_tensor fails on scalars. We fix it by calling flatten
        vals=(arr.astype(bytes) if cast_to_bytes else arr).flatten(),
        raw=False,
    )
