from typing import Tuple

import numpy
import pytest

import spox.opset.ai.onnx.v17 as op
from spox._internal_op import unsafe_reshape
from spox._var import Var


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
    def maybe(cond: Var, var: Var) -> Var:
        (i,) = op.if_(
            cond,
            else_branch=lambda: [op.optional(type=var.type)],
            then_branch=lambda: [op.optional(var)],
        )
        return i

    @staticmethod
    def push(var: Var, what: Var) -> Var:
        return op.sequence_insert(var, what)

    @staticmethod
    def top(var: Var) -> Var:
        return op.sequence_at(var, op.const(-1))

    @staticmethod
    def pop(var: Var) -> Var:
        return op.sequence_erase(var, op.const(-1))

    @staticmethod
    def at(t: Var, j: Var) -> Var:
        j = op.reshape(j, op.const(numpy.array([1], dtype=numpy.int64)))
        return op.reshape(
            op.slice(t, j, op.add(j, op.const(1))),
            op.const(numpy.array([], dtype=numpy.int64)),
        )

    @staticmethod
    def empty(s: Var) -> Var:
        return op.equal(op.sequence_length(s), op.const(0))

    def match_brackets(ext, xs: Var) -> Var:
        def bracket_matcher_step(
            i: Var, _cond: Var, stack: Var, result: Var, _: Var
        ) -> Tuple[Var, Var, Var, Var]:
            closing = op.less(ext.at(xs, i), op.const(0))
            ignore = op.equal(ext.at(xs, i), op.const(0))
            pair = op.concat([ext.top(stack), i], axis=-1)
            ok = op.not_(op.and_(closing, ext.empty(stack)))
            stack_after, result_after = op.if_(
                ignore,
                then_branch=lambda: (stack, result),
                else_branch=lambda: op.if_(
                    ok,
                    then_branch=lambda: op.if_(
                        closing,
                        then_branch=lambda: (ext.pop(stack), ext.push(result, pair)),
                        else_branch=lambda: (ext.push(stack, i), result),
                    ),
                    else_branch=lambda: (stack, result),
                ),
            )
            return ok, stack_after, result_after, ok

        unpaired, pairs, all_ok = op.loop(
            op.reshape(op.size(xs), op.const([1])),
            None,
            [
                op.sequence_empty(dtype=numpy.int64),
                op.sequence_empty(dtype=numpy.int64),
                op.const(numpy.array(True)),
            ],
            body=bracket_matcher_step,
        )
        return ext.maybe(op.and_(all_ok, ext.empty(unpaired)), pairs)

    def scalars(ext, *args: Var) -> Var:
        return op.concat([op.unsqueeze(arg, op.const([0])) for arg in args], axis=-1)

    def onehot(ext, n: Var, i: Var) -> Var:
        return op.pad(op.const([1]), ext.scalars(i, op.sub(op.sub(n, i), op.const(1))))

    def set_to(ext, t: Var, j: Var, x: Var) -> Var:
        return op.where(op.cast(ext.onehot(op.size(t), j), to=numpy.bool_), x, t)

    @staticmethod
    def is_token(var: Var, token: str) -> Var:
        return op.equal(
            op.cast(var, to=numpy.int32),
            op.cast(op.const(numpy.uint8(ord(token))), to=numpy.int32),
        )

    def flat_concat(ext, s: Var) -> Var:
        return op.if_(
            ext.empty(s),
            then_branch=lambda: (ext.empty_i64(),),
            else_branch=lambda: (
                unsafe_reshape(op.concat_from_sequence(s, axis=0), (None,)),
            ),
        )[0]

    def remap(ext, s: Var, t: Var, x: Var) -> Var:
        return op.reshape(op.compress(t, op.equal(s, x)), ext.empty_i64())

    def find(ext, t: Var, x: Var) -> Var:
        x = op.reshape(op.identity(x), ext.empty_i64())
        (ret,) = op.loop(
            op.size(t),
            v_initial=[op.const([-1])],
            body=lambda i, _1, _2: op.if_(
                op.equal(ext.at(t, i), x),
                then_branch=lambda: (ext.false(), i, ext.at(t, i)),
                else_branch=lambda: (ext.true(), op.const([-1]), ext.at(t, i)),
            ),
        )
        return ret


@pytest.fixture(scope="session")
def ext():
    return Extras()
