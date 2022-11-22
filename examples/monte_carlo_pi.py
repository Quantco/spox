import math
import random
import time

import numpy
import onnxruntime

import spox
import spox._graph
import spox.opset.ai.onnx.v17 as op


def uniform_of_shape(shape: spox.Var) -> spox.Var:
    # Not providing the seed for randomness yields an undefined one - ONNX runtime seems to use the same one twice,
    # so we use a Python seed.
    return op.cast(
        op.random_uniform_like(
            op.constant_of_shape(shape), seed=random.getrandbits(32)
        ),
        to=numpy.float64,
    )


def get_pi_graph() -> spox._graph.Graph:
    (n,) = spox._graph.arguments(n=spox._type_system.Tensor(numpy.int64, ()))

    xs = uniform_of_shape(op.reshape(n, op.const([1])))
    ys = uniform_of_shape(op.reshape(n, op.const([1])))
    hypot = op.add(op.mul(xs, xs), op.mul(ys, ys))
    in_circle = op.less_or_equal(hypot, op.const(numpy.float64(1)))
    count = op.reduce_sum(op.cast(in_circle, to=numpy.int64), op.const([-1]))
    pi = op.mul(
        op.const(numpy.float64(4.0)),
        op.div(op.cast(count, to=numpy.float64), op.cast(n, to=numpy.float64)),
    )

    return (
        spox._graph.results(pi=pi)
        .with_name("monte_carlo_pi")
        .with_doc("Computes Pi by Monte Carlo with n samples.")
    )


session_construct = time.time()
session = onnxruntime.InferenceSession(
    get_pi_graph().to_onnx_model().SerializeToString()
)
session_constructed = time.time()

print(f"ONNX session time - {1000 * (session_constructed - session_construct):.1f} ms")


def numpy_get_pi(n: int) -> float:
    xs, ys = numpy.random.uniform(size=n), numpy.random.uniform(size=n)
    return float(4 * numpy.sum(xs**2 + ys**2 <= 1) / n)


def onnx_get_pi(n: int) -> float:
    return float(session.run(None, {"n": numpy.array(numpy.int64(n))})[0])


test_n = 10**7
print(f"Computation at {test_n = }:")

numpy_get = time.time()
numpy_pi = numpy_get_pi(test_n)
numpy_got = time.time()

print(f"- numpy {1000 * (numpy_got - numpy_get):.1f} ms -> {numpy_pi}")

onnx_get = time.time()
onnx_pi = onnx_get_pi(test_n)
onnx_got = time.time()

print(f"- ONNX  {1000 * (onnx_got - onnx_get):.1f} ms -> {onnx_pi}")
print(f"real => {math.pi}")
