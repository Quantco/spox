import inspect
import itertools
from dataclasses import dataclass, make_dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Tuple, TypeVar

import onnx

from . import _attributes
from ._fields import BaseAttributes, BaseInputs, BaseOutputs
from ._internal_op import _InternalNode
from ._node import Node, OpType
from ._type_system import Type
from ._var import Var

if TYPE_CHECKING:
    from . import _graph

DEFAULT_FUNCTION_DOMAIN = "spox.default"

ConstructorT = TypeVar("ConstructorT", bound=Callable[..., Iterable[Var]])


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

    func_args: Dict[str, Var]
    func_attrs: Dict[str, _attributes.Attr]
    func_inputs: BaseInputs
    func_outputs: BaseOutputs
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
            **{name: var.type for name, var in self.inputs.get_vars().items()}
        )

        self.func_attrs = {}
        for name, attr in self.attrs.get_fields().items():
            if attr is None:
                raise TypeError(
                    f"Function attributes is not optional, but {name} is None."
                )
            self.func_attrs[name] = attr

        self.func_inputs = self.Inputs(**self.func_args)  # type: ignore
        self.func_outputs = self.constructor(self.func_attrs, self.func_inputs)
        self.func_graph = _graph.results(**self.func_outputs.get_vars()).with_arguments(
            *self.func_args.values()
        )

        return {
            name: var.type
            for name, var in self.func_outputs.get_vars().items()
            if var.type
        }

    @property
    def opset_req(self):
        node_opset_req = Node.opset_req.fget(self)  # type: ignore
        return node_opset_req | self.func_graph._get_build_result().opset_req

    def update_metadata(self, opset_req, initializers, functions):
        super().update_metadata(opset_req, initializers, functions)
        functions.append(self)
        functions.extend(self.func_graph._get_build_result().functions)

    def to_onnx_function(
        self, *, extra_opset_req: Iterable[Tuple[str, int]] = ()
    ) -> onnx.FunctionProto:
        """
        Translate self into an ONNX FunctionProto, based on the ``func_*`` attributes set when this operator
        was constructed. It is later assumed that all functions sharing the ``op_type`` have the same body.

        Functions do not attempt to adapt nodes into homogenous versions.
        """
        graph = self.func_graph.with_opset(*extra_opset_req)
        node_protos = itertools.chain.from_iterable(graph.get_adapted_nodes().values())
        return onnx.helper.make_function(
            self.op_type.domain,
            self.op_type.identifier,
            list(self.func_inputs.get_fields().keys()),
            list(self.func_outputs.get_fields().keys()),
            list(node_protos),
            [
                onnx.helper.make_operatorsetid(domain, version)
                for domain, version in graph.get_opsets().items()
            ],
            list(self.func_attrs.keys()),
        )


def _make_function_cls(fun, num_inputs, num_outputs, domain, version, name):
    _FuncInputs = make_dataclass(
        "_FuncInputs", ((f"in{i}", Var) for i in range(num_inputs)), bases=(BaseInputs,)
    )
    _FuncOutputs = make_dataclass(
        "_FuncOutputs",
        ((f"out{i}", Var) for i in range(num_outputs)),
        bases=(BaseOutputs,),
    )

    class _Func(Function):
        @dataclass
        class Attributes(BaseAttributes):
            pass

        Inputs = _FuncInputs
        Outputs = _FuncOutputs
        op_type = OpType(name, domain, version)

        def constructor(self, attrs, inputs):
            return self.Outputs(*fun(*inputs.get_fields().values()))

    return _Func


def to_function(name: str, domain: str = "spox.function", *, _version: int = 0):
    """
    Decorate a given function to make the operation performed by it add a Spox function to the graph.

    The function must be deterministic in the performed operations, as otherwise an error will be raised at build
    due to inconsistent function bodies.

    ``fun`` is assumed to take only Var arguments and return an iterable of them. These will be used to generate the
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

        def get_num_outputs(*args: Var) -> int:
            nonlocal _num_outputs
            if _num_outputs is None:
                _num_outputs = sum(1 for _ in fun(*args))
            return _num_outputs

        def init(*args: Var):
            nonlocal _cls
            if _cls is not None:
                return _cls

            _cls = _make_function_cls(
                fun, num_inputs, get_num_outputs(*args), domain, _version, name
            )
            return _cls

        def alt_fun(*args: Var) -> Iterable[Var]:
            cls = init(*args)
            return (
                cls(cls.Attributes(), cls.Inputs(*args)).outputs.get_fields().values()
            )

        return alt_fun  # type: ignore

    return inner
