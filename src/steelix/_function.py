import inspect
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Optional, TypeVar

import onnx
from typing_extensions import TypeAlias

from . import _attributes
from ._arrow import Arrow
from ._arrowfields import ArrowFields
from ._internal_op import _InternalNode
from ._node import Node, OpType
from ._type_system import Type

if TYPE_CHECKING:
    from . import _graph

DEFAULT_FUNCTION_DOMAIN = "steelix.default"

Constructor: TypeAlias = Callable[..., Iterable[Arrow]]
ConstructorT = TypeVar("ConstructorT", bound=Constructor)


class Function(_InternalNode):
    """
    Type of ``Node`` that is defined in terms of its abstract ``constructor``, which may invoke standard operators.
    Can be built into an ONNX function, and when a full model is built all instances inheriting from Function
    are converted to ONNX functions and stored alongside the built graph.

    ONNX Functions are untyped in inputs and outputs (like operators), so all type checking is done within
    the operators themselves.

    Function constructors must always be deterministic up to graph structure irrespective of attributes/inputs.
    In essence, the protobuf build result must be the same in every built instance of the function.

    Functions are in a way dimorphic - on one hand they serve as normal Nodes/operators created by operator
    constructors, but internally the overriden ``Function.constructor`` gets called to access the types.
    The ``func_*`` fields are then used for the construction of an implicit graph (BuildeR), which is built into ONNX
    via the ``to_onnx_function`` method.
    """

    func_args: Dict[str, Arrow]
    func_attrs: Dict[str, _attributes._Ref]
    func_inputs: ArrowFields
    func_outputs: ArrowFields
    func_graph: "_graph.Graph"

    def constructor(self, attrs, inputs):
        """
        Abstract method for functions.

        Takes attributes (as refs) and inputs of this function, and constructs the outputs.

        Operates on a graph separate from the rest, and the types of the outputs are extracted into what goes in
        the actual graph.
        """
        raise NotImplementedError(
            f"Function {type(self).__name__} does not implement a constructor."
        )

    def infer_output_types(self) -> Dict[str, Type]:
        from . import _graph

        self.func_args = _graph.arguments_dict(
            **{name: arrow.type for name, arrow in self.inputs.as_dict().items()}
        )

        func_attrs = {}
        for name, attr in self.attrs.__dict__.items():
            func_attrs[name] = _attributes._Ref(concrete=attr, outer_name=name)
        self.func_attrs = func_attrs

        self.func_inputs = self.Inputs(**self.func_args)
        self.func_outputs = self.constructor(self.func_attrs, self.func_inputs)
        self.func_graph = _graph.results(**self.func_outputs.as_dict()).with_arguments(
            *self.func_args.values()
        )

        return {
            name: arrow.type
            for name, arrow in self.func_outputs.as_dict().items()
            if arrow.type
        }

    @property
    def opset_req(self):
        node_opset_req = Node.opset_req.fget(self)  # type: ignore
        return node_opset_req | self.func_graph._get_build_result().opset_req

    def update_metadata(self, opset_req, initializers, functions):
        super().update_metadata(opset_req, initializers, functions)
        functions.append(self)
        functions.extend(self.func_graph._get_build_result().functions)

    def to_onnx_function(self, name: Optional[str] = None) -> onnx.FunctionProto:
        """
        Translate self into an ONNX FunctionProto, based on the ``func_*`` attributes set when this operator
        was constructed. It is later assumed that all functions sharing the ``op_type`` have the same body.

        Functions do not attempt to adapt nodes into homogenous versions.
        """
        node_protos = itertools.chain.from_iterable(
            self.func_graph._get_build_result().nodes.values()
        )
        return onnx.helper.make_function(
            self.op_type.domain,
            self.op_type.identifier,
            self.func_inputs.get_kwargs(),
            self.func_outputs.get_kwargs(),
            list(node_protos),
            [
                onnx.helper.make_operatorsetid(domain, version)
                for domain, version in self.func_graph.get_opsets().items()
            ],
            self.Attributes.__dataclass_fields__.keys(),
        )


def _make_function_cls(fun, num_inputs, num_outputs, domain, version, name):
    class _FuncInputs(ArrowFields):
        ...

    class _FuncOutputs(ArrowFields):
        ...

    _FuncInputs.__annotations__ = {f"in{i}": "Arrow" for i in range(num_inputs)}
    _FuncOutputs.__annotations__ = {f"out{i}": "Arrow" for i in range(num_outputs)}

    class _Func(Function):
        @dataclass
        class Attributes:
            pass

        Inputs = _FuncInputs
        Outputs = _FuncOutputs
        op_type = OpType(name, domain, version)

        def constructor(self, attrs, inputs):
            return self.Outputs(*fun(*inputs.unpack()))

    return _Func


def to_function(name: str, domain: str = "steelix.function", *, _version: int = 0):
    """
    Decorate a given function to make the operation performed by it add a Steelix function to the graph.

    The function must be deterministic in the performed operations, as otherwise an error will be raised at build
    due to inconsistent function bodies.

    ``fun`` is assumed to take only Arrow arguments and return an iterable of them. These will be used to generate the
    function class signature.

    Keep in mind that functions with the same name & domain will be merged together.
    Versions should only be specified when it's necessary for the output to have this information
    (e.g. providing functions for existing operators).

    """

    def inner(fun: ConstructorT) -> ConstructorT:
        sig = inspect.signature(fun)

        num_inputs = len(sig.parameters)
        _num_outputs = None
        _cls = None

        def get_num_outputs(*args: Arrow) -> int:
            nonlocal _num_outputs
            if _num_outputs is None:
                _num_outputs = sum(1 for _ in fun(*args))
            return _num_outputs

        def init(*args: Arrow):
            nonlocal _cls
            if _cls is not None:
                return _cls

            _cls = _make_function_cls(
                fun, num_inputs, get_num_outputs(*args), domain, _version, name
            )
            return _cls

        def alt_fun(*args: Arrow) -> Iterable[Arrow]:
            cls = init(*args)
            return cls(cls.Attributes(), cls.Inputs(*args)).outputs.unpack()

        return alt_fun  # type: ignore

    return inner
