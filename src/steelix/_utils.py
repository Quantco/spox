from typing import Optional

import numpy as np
from onnx import TensorProto, numpy_helper


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
