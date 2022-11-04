from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from onnx import TensorProto, numpy_helper
from onnx.helper import mapping

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
    """Convert numpy data types into integer tensor types."""
    # normalize string data types
    dtype = np.dtype(np.dtype(dtype_like).type)
    return _DTYPE_TO_TENSOR_TYPE[dtype]


def from_array(array: np.ndarray, name: Optional[str] = None) -> TensorProto:
    """
    Helper function for converting numpy arrays into TensorProto.

    As it may be useful to name the TensorProto (e.g. in
    initializers), there is a ``name`` parameter.

    Uses ``numpy.str_`` instead of ``numpy.object_`` for strings,
    calling ``onnx.numpy_helper.from_array`` internally.
    """
    if array.dtype.type is np.str_:
        array = array.astype(np.object_)
    return numpy_helper.from_array(array, name=name)
