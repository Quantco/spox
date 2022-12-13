import onnx.shape_inference

InferenceError = onnx.shape_inference.InferenceError
ValidationError = onnx.checker.ValidationError


class InferenceWarning(Warning):
    """Warning related to partial typing of Variables.

    Incomplete type information may lead to reduced code safety or
    failure to build the model. The most common underlying cause for
    this warning is a missing or incomplete type inference for an
    operator in the upstream ``onnx`` project. Type and shape
    information may be manually reset by using (unsafe) casts or
    reshapes.
    """


class BuildError(Exception):
    """An error within the build process.

    Usually this means that Spox failed to resolve the graph structure
    due to an internal error - for example if the implicit Node graph
    was tampered with or the algorithm had a fault.
    """

    pass


__all__ = ["InferenceWarning", "InferenceError", "ValidationError"]
