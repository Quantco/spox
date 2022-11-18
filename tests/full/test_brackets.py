import numpy
import pytest

from spox._graph import arguments, results
from spox._type_system import Tensor


@pytest.fixture(scope="session")
def bracket_graph(ext):
    return results(pairs=ext.match_brackets(*arguments(xs=Tensor(numpy.int64, ("N",)))))


@pytest.mark.parametrize(
    "brackets,result",
    [
        ([], []),
        ([1, -1], [(0, 1)]),
        ([1, 1, 1, -1, -1, 1, -1, -1], [(2, 3), (1, 4), (5, 6), (0, 7)]),
        ([1, 1, 0, -1, 0, -1], [(1, 3), (0, 5)]),
        ([1], None),
        ([-1], None),
        ([1, 1, -1], None),
        ([1, -1, -1], None),
    ],
)
def test_sequence_bracket_matching(onnx_helper, bracket_graph, brackets, result):
    def get_bracket_matching(seq):
        ret = onnx_helper.run(bracket_graph, "pairs", xs=numpy.array(seq, dtype=int))
        return [(x, y) for x, y in ret] if ret is not None else None

    assert get_bracket_matching(brackets) == result
