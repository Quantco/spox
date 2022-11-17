from typing import Tuple

import numpy
import pytest

import steelix.opset.ai.onnx.v17 as op
from steelix._arrow import Arrow
from steelix._internal_op import unsafe_reshape


# noinspection PyMethodParameters
class Extras:
    _false = None
    _true = None
    _empty_i64 = None

    def false(self):
        if self._false is None:
            self._false = op.const(numpy.array(False))
        return self._false

    def true(self):
        if self._true is None:
            self._true = op.const(numpy.array(True))
        return self._true

    def empty_i64(self):
        if self._empty_i64 is None:
            self._empty_i64 = op.const(numpy.array([], dtype=numpy.int64))
        return self._empty_i64

    @staticmethod
    def maybe(cond: Arrow, arrow: Arrow) -> Arrow:
        (i,) = op.xif(
            cond,
            else_branch=[op.optional(type=arrow.type)],
            then_branch=[op.optional(arrow)],
        )
        return i

    @staticmethod
    def push(arrow: Arrow, what: Arrow) -> Arrow:
        return op.sequence_insert(arrow, what)

    @staticmethod
    def top(arrow: Arrow) -> Arrow:
        return op.sequence_at(arrow, op.const(-1))

    @staticmethod
    def pop(arrow: Arrow) -> Arrow:
        return op.sequence_erase(arrow, op.const(-1))

    @staticmethod
    def at(t: Arrow, j: Arrow) -> Arrow:
        j = op.reshape(j, op.const(numpy.array([1], dtype=numpy.int64)))
        return op.reshape(
            op.slice(t, j, op.add(j, op.const(1))),
            op.const(numpy.array([], dtype=numpy.int64)),
        )

    @staticmethod
    def empty(s: Arrow) -> Arrow:
        return op.equal(op.sequence_length(s), op.const(0))

    def match_brackets(ext, xs: Arrow) -> Arrow:
        def bracket_matcher_step(
            i: Arrow, _cond: Arrow, stack: Arrow, result: Arrow, _: Arrow
        ) -> Tuple[Arrow, Arrow, Arrow, Arrow]:
            closing = op.less(*op.promote(ext.at(xs, i), 0))
            ignore = op.equal(*op.promote(ext.at(xs, i), 0))
            pair = op.concat([ext.top(stack), i], axis=-1)
            ok = op.not_(op.and_(closing, ext.empty(stack)))
            stack_after, result_after = op.xif(
                ignore,
                then_branch=(stack, result),
                else_branch=(
                    op.xif(
                        ok,
                        then_branch=op.xif(
                            closing,
                            then_branch=(ext.pop(stack), ext.push(result, pair)),
                            else_branch=(ext.push(stack, i), result),
                        ),
                        else_branch=(stack, result),
                    )
                ),
            )
            return ok, stack_after, result_after, ok

        unpaired, pairs, all_ok = op.xloop(
            op.reshape(op.size(xs), op.const([1])),
            None,
            [
                op.sequence_empty(dtype=numpy.int64),
                op.sequence_empty(dtype=numpy.int64),
                op.const(numpy.array(True)),
            ],
            fun=bracket_matcher_step,
        )
        return ext.maybe(op.and_(all_ok, ext.empty(unpaired)), pairs)

    def scalars(ext, *args: Arrow) -> Arrow:
        return op.concat([op.unsqueeze(arg, op.const([0])) for arg in args], axis=-1)

    def onehot(ext, n: Arrow, i: Arrow) -> Arrow:
        return op.pad(op.const([1]), ext.scalars(i, op.sub(op.sub(n, i), op.const(1))))

    def set_to(ext, t: Arrow, j: Arrow, x: Arrow) -> Arrow:
        return op.where(op.cast(ext.onehot(op.size(t), j), to=numpy.bool_), x, t)

    @staticmethod
    def is_token(arrow: Arrow, token: str) -> Arrow:
        return op.equal(
            op.cast(arrow, to=numpy.int32),
            op.cast(op.const(numpy.uint8(ord(token))), to=numpy.int32),
        )

    def flat_concat(ext, s: Arrow) -> Arrow:
        return op.xif(
            ext.empty(s),
            then_branch=(ext.empty_i64(),),
            else_branch=(unsafe_reshape(op.concat_from_sequence(s, axis=0), (None,)),),
        )[0]

    def remap(ext, s: Arrow, t: Arrow, x: Arrow) -> Arrow:
        return op.reshape(op.compress(t, op.equal(s, x)), ext.empty_i64())

    def find(ext, t: Arrow, x: Arrow) -> Arrow:
        x = op.reshape(op.identity(x), ext.empty_i64())
        (ret,) = op.xloop(
            op.size(t),
            initial=[op.const([-1])],
            fun=lambda i, _1, _2: op.xif(
                op.equal(ext.at(t, i), x),
                then_branch=(ext.false(), i, ext.at(t, i)),
                else_branch=(ext.true(), op.const([-1]), ext.at(t, i)),
            ),
        )
        return ret


@pytest.fixture(scope="session")
def ext():
    return Extras()
