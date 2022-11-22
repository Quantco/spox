import numpy
import pytest

import spox.opset.ai.onnx.v17 as op
from spox._graph import arguments, results
from spox._internal_op import unsafe_reshape
from spox._type_system import Tensor
from spox._var import Var

HELLO_WORLD_IMPL = """
++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>."     Comments work!
">---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.   Wow!
"""

_graph_cache = None


@pytest.fixture(scope="session")
def bf_interpreter_graph(ext):
    (prog, inputs) = arguments(
        prog=Tensor(numpy.uint8, ("N",)), inputs=Tensor(numpy.int64, ("I",))
    )

    brackets = op.add(
        op.where(ext.is_token(prog, "["), op.const(+1), op.const(0)),
        op.where(ext.is_token(prog, "]"), op.const(-1), op.const(0)),
    )
    maybe_matches = ext.match_brackets(brackets)
    matches = op.optional_get_element(maybe_matches)
    (matches,) = op.if_(
        ext.empty(matches),
        then_branch=lambda: (op.reshape(ext.empty_i64(), op.const([0, 2])),),
        else_branch=lambda: (
            unsafe_reshape(
                op.concat_from_sequence(matches, axis=0, new_axis=1), (None, 2)
            ),
        ),
    )
    lefts, rights = (
        op.squeeze(var, axes=op.const([-1]))
        for var in op.split(matches, op.const([1, 1]), outputs_count=2, axis=1)
    )

    TAPE = 2**10
    TERM = 2**16

    def step(
        _i: Var,
        _cond: Var,
        ip: Var,
        ptr: Var,
        iptr: Var,
        tape: Var,
        output_seq: Var,
    ):
        at_tape = op.unsqueeze(ext.at(tape, ptr), op.const([0]))
        cmd = ext.at(prog, ip)

        (tape_inc,) = op.if_(
            ext.is_token(cmd, "+"),
            then_branch=lambda: (op.add(tape, ext.onehot(op.size(tape), ptr)),),
            else_branch=lambda: (tape,),
        )
        (tape_dec,) = op.if_(
            ext.is_token(cmd, "-"),
            then_branch=lambda: (op.sub(tape, ext.onehot(op.size(tape), ptr)),),
            else_branch=lambda: (tape_inc,),
        )

        (at_iptr,) = op.if_(
            op.less(iptr, op.size(inputs)),
            then_branch=lambda: (ext.at(inputs, iptr),),
            else_branch=lambda: (op.const(0),),
        )
        tape_read, iptr_read = op.if_(
            ext.is_token(cmd, ","),
            then_branch=lambda: (
                ext.set_to(tape, ptr, at_iptr),
                op.add(iptr, op.const(1)),
            ),
            else_branch=lambda: (tape_dec, iptr),
        )

        (output_seq_write,) = op.if_(
            ext.is_token(cmd, "."),
            then_branch=lambda: (op.sequence_insert(output_seq, at_tape),),
            else_branch=lambda: (output_seq,),
        )

        (ptr_inc,) = op.if_(
            ext.is_token(cmd, ">"),
            then_branch=lambda: (op.mod(op.add(ptr, op.const(1)), op.const(TAPE)),),
            else_branch=lambda: (ptr,),
        )
        (ptr_dec,) = op.if_(
            ext.is_token(cmd, "<"),
            then_branch=lambda: (op.mod(op.sub(ptr, op.const(1)), op.const(TAPE)),),
            else_branch=lambda: (ptr_inc,),
        )

        zero_at_tape = op.equal(at_tape, op.const(0))
        (ip_jump_right,) = op.if_(
            op.and_(ext.is_token(cmd, "["), zero_at_tape),
            then_branch=lambda: (ext.remap(lefts, rights, ip),),
            else_branch=lambda: (ip,),
        )
        (ip_jump_left,) = op.if_(
            op.and_(ext.is_token(cmd, "]"), op.not_(zero_at_tape)),
            then_branch=lambda: (ext.remap(rights, lefts, ip),),
            else_branch=lambda: (ip_jump_right,),
        )

        next_ip = op.add(ip_jump_left, op.const(1))
        next_ptr = ptr_dec
        next_iptr = iptr_read
        next_tape = tape_read
        next_output_seq = output_seq_write

        return op.if_(
            op.less(ip, op.size(prog)),
            then_branch=lambda: (
                ext.true(),
                next_ip,
                next_ptr,
                next_iptr,
                next_tape,
                next_output_seq,
            ),
            else_branch=lambda: (ext.false(), ip, ptr, iptr, tape, output_seq),
        )

    _final_ip, _final_ptr, _final_iptr, final_tape, final_output_seq = op.loop(
        op.const([TERM]),
        v_initial=[
            op.const(0),
            op.const(0),
            op.const(0),
            op.const(numpy.array([0] * TAPE, dtype=numpy.int64)),
            op.sequence_empty(dtype=numpy.int64),
        ],
        body=step,
    )

    output = ext.flat_concat(final_output_seq)
    success_then_output = ext.maybe(op.optional_has_element(maybe_matches), output)

    return results(output=success_then_output).with_arguments(prog, inputs)


@pytest.mark.parametrize(
    "prog,inputs,result",
    [
        ("", "", ""),
        ("", "nothing", ""),
        ("[", "", None),
        ("+" * ord(":") + "." + "-" * ord(":") + "+" * ord(")") + ".", "", ":)"),
        ("+" * ord(":") + ".>" + "+" * ord(")") + ".", "", ":)"),
        (",." * 6, "abcxyz", "abcxyz"),
        ("+[-],.", "!", "!"),
        ("[]" + ord("!") * "+" + ".>" + "[]", "", "!"),
        ("[,.],.", "!!", "!"),
        (",[.,]", "I'm a cat.", "I'm a cat."),
        (HELLO_WORLD_IMPL, "", "Hello World!\n"),
    ],
)
def test_turing_completeness(onnx_helper, bf_interpreter_graph, prog, inputs, result):
    def bf(program, input_values=(), chars=False):
        input_values = [ord(x) if isinstance(x, str) else x for x in input_values]
        run_results = onnx_helper.run(
            bf_interpreter_graph,
            prog=numpy.array([ord(x) for x in program], dtype=numpy.uint8),
            inputs=numpy.array(input_values, dtype=numpy.int64),
        )
        ret = run_results["output"]
        if ret is not None:
            if chars:
                return "".join(chr(x) for x in ret)
            return list(ret)
        return None

    assert bf(prog, inputs, chars=True) == result
