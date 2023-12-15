"""Internal module implementing the low-level Graph object and functions for creating arguments and Graphs."""

import dataclasses
import itertools
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import numpy
import onnx
import onnx.shape_inference

from . import _build
from ._adapt import adapt_best_effort
from ._attributes import AttrString, AttrTensor, AttrType
from ._fields import BaseInputs
from ._internal_op import Argument, _Initializer
from ._node import Node
from ._schemas import max_opset_policy
from ._type_system import Tensor, Type
from ._utils import from_array
from ._var import Var


def arguments_dict(**kwargs: Optional[Union[Type, numpy.ndarray]]) -> Dict[str, Var]:
    """
    Parameters
    ----------
    kwargs
        Types or arrays for the newly created arguments.
        Keyword argument names are meaningful and used to name the arguments of the final graph.
        A numpy array is interpreted as an initializer (default argument value),
        and its type is used to create a respective Tensor.
    Returns
    -------
    Dict[str, Var]
        Argument Vars of given Types, named the same as kwargs.
    """
    result = {}
    for name, info in kwargs.items():
        attr_name = AttrString(value=name, name="dummy")
        if isinstance(info, Type):
            result[name] = Argument(
                Argument.Attributes(
                    name=attr_name,
                    type=AttrType(value=info, name="dummy"),
                    default=None,
                ),
                BaseInputs(),
            ).outputs.arg
        elif isinstance(info, numpy.ndarray):
            ty = Tensor(info.dtype, info.shape)
            result[name] = Argument(
                Argument.Attributes(
                    name=attr_name,
                    type=AttrType(value=ty, name="dummy"),
                    default=AttrTensor(value=info, name="dummy"),
                ),
                BaseInputs(),
            ).outputs.arg
        else:
            raise TypeError(f"Cannot construct argument from {type(info)}.")
    return result


def arguments(**kwargs: Optional[Union[Type, numpy.ndarray]]) -> Tuple[Var, ...]:
    """This function is a shorthand for a respective call to ``arguments_dict``, unpacking the Vars from the dict."""
    return tuple(arguments_dict(**kwargs).values())


def enum_arguments(
    *infos: Union[Type, numpy.ndarray], prefix: str = "in"
) -> Tuple[Var, ...]:
    """
    Convenience function for creating an enumeration of arguments, prefixed with ``prefix``.
    Calls ``arguments`` internally.

    This is a function useful for creating subgraphs, where the exact names don't really matter, only their order.
    Note that repeated use of this in the same graph may repeat names if the prefix is also the same.

    Parameters
    ----------
    infos
        Types/initializers for the created arguments.
    prefix
        String to prefix the names of created arguments with.
    Returns
    -------
    Tuple[Var, ...]
        Argument Vars as specified, in the same order as information ``infos``.
    """
    return arguments(**{f"{prefix}{i}": info for i, info in enumerate(infos)})


def initializer(arr: numpy.ndarray) -> Var:
    """
    Create a single initializer (frozen argument) with a given array value.

    This is an alternate method to creating a constant from using a dedicated Constant constructor.
    As a convention, initializers may be used for more global-scope constants.

    Parameters
    ----------
    arr
        Value of the initializer.
    Returns
    -------
        Var which is always equal to the respective value provided by `arr`.
    """
    return _Initializer(
        _Initializer.Attributes(value=AttrTensor(value=arr, name="dummy")),
        BaseInputs(),
    ).outputs.arg


@dataclass(frozen=True, eq=False)
class Graph:
    """
    Represents an abstraction for a wrapped up ONNX computation graph,
    that can be built into ONNX GraphProto & ModelProto.

    Should be constructed only with the ``results`` functions.

    Use the methods ``rename``, ``doc``, ``with_arguments`` to set additional data for the graph.
    These methods return a new instance of Graph with a respective private attribute set.

    Note that to not only fix results (which a Graph is constructed with), but also arguments, ``with_arguments``
    should be used.

    Note: building a Graph is cached, so changing it in-place without the setters will invalidate the build.
    """

    _results: Dict[str, Var]
    _name: Optional[str] = None
    _doc_string: Optional[str] = None
    _arguments: Optional[Tuple[Var, ...]] = None
    _extra_opset_req: Optional[Set[Tuple[str, int]]] = None
    _constructor: Optional[Callable[..., Iterable[Var]]] = None
    _build_result: "_build.Cached[_build.BuildResult]" = dataclasses.field(
        default_factory=_build.Cached
    )

    def __repr__(self):
        name_repr = self._name if self._name is not None else "?"
        args_repr = (
            f"{', '.join(str(a) for a in self._arguments)}"
            if self._arguments is not None
            else "..."
        )
        res_repr = f"{', '.join(f'{k}: {a}' for k, a in self._results.items())}"
        comments: List[str] = []
        if self._doc_string is not None:
            comments.append(f'"{self._doc_string[:10]}..."')
        if self._extra_opset_req is not None:
            comments.append(f"+{len(self._extra_opset_req)} opset req")
        return f"<Graph '{name_repr}' ({args_repr}) -> ({res_repr}){': ' if comments else ''}{', '.join(comments)}>"

    def __post_init__(self):
        if any(not isinstance(var, Var) for var in self._results.values()):
            seen_types = {type(obj) for obj in self._results.values()}
            raise TypeError(f"Graph results must be Vars, not {seen_types - {Var}}.")
        if self._arguments is not None and any(
            not isinstance(var, Var) for var in self._arguments
        ):
            seen_types = {type(obj) for obj in self._arguments}
            raise TypeError(f"Build outputs must be Vars, not {seen_types - {Var}}.")

    def with_name(self, name: str) -> "Graph":
        """Return a Graph with its name set to ``name``."""
        return replace(self, _name=name)

    def with_doc(self, doc_string: str) -> "Graph":
        """Return a Graph with its doc string set to ``doc``."""
        return replace(self, _doc_string=doc_string)

    def with_arguments(self, *args: Var) -> "Graph":
        """
        Return a Graph with given Vars marked as exactly its arguments.
        A useful idiom is ``results(...).with_arguments(...)`` when you want to specify both results and arguments.
        """
        return replace(self, _arguments=args)

    def with_opset(self, *args: Tuple[str, int]) -> "Graph":
        """
        Add the given minimum opset requirements to the graph.
        Useful when the graph is using legacy nodes, but Spox should attempt to convert them to a required version.
        """
        extra_opset_req = set(args)
        if self._extra_opset_req is not None:
            extra_opset_req |= self._extra_opset_req
        return replace(self, _extra_opset_req=extra_opset_req)

    def _with_constructor(self, fun: Callable[..., Iterable[Var]]) -> "Graph":
        """Assign a constructor that constructed this Graph given ``self.requested_arguments``."""
        return replace(self, _constructor=fun)

    def _reconstruct(self, *args: Var) -> "Graph":
        assert self._constructor is not None
        return (
            results(**dict(zip(self._results, self._constructor(*args))))
            .with_arguments(*args)
            ._with_constructor(self._constructor)
        )

    def _inject_build_result(self, what: "_build.BuildResult") -> "Graph":
        """
        Internal function used to build a Graph with a custom build result.
        Used when building subgraphs to have further control over the build state.
        """
        return replace(self, _build_result=_build.Cached(what))

    @property
    def requested_arguments(self) -> Optional[Iterable[Var]]:
        """Arguments requested by this Graph (for building) - ``None`` if unspecified."""
        return self._arguments

    @property
    def requested_results(self) -> Dict[str, Var]:
        """Results (named) requested by this Graph (for building)."""
        return self._results

    def get_arguments(self) -> Dict[str, Var]:
        """
        Get the effective named arguments (after build) of this Graph.

        May be expensive, as it has to build Use ``requested_arguments`` for a cheaper variant that may be sufficient.
        """
        return {
            self._get_build_result().scope.var[var]: var
            for var in self._get_build_result().arguments
        }

    def get_results(self) -> Dict[str, Var]:
        """
        Get the effective named results (after build) of this Graph.

        May be expensive, as it has to build. Use ``requested_results`` for a cheaper variant that may be sufficient.
        """
        return {
            self._get_build_result().scope.var[var]: var
            for var in self._get_build_result().results
        }

    def get_opsets(self) -> Dict[str, int]:
        """
        Get the effective opsets used by this Graph. The used policy for mixed versions is maximum-requested.

        May be expensive, as it has to build.
        """
        return max_opset_policy(self._get_opset_req())

    def _get_build_result(self) -> "_build.BuildResult":
        """Internal function for getting (with cache) the build result structure for this Graph."""
        if self._build_result._value is None:
            self._build_result.value = _build.Builder(self).build_main()
        return self._build_result.value

    def _get_opset_req(self) -> Set[Tuple[str, int]]:
        """Internal function for accessing the opset requirements, including extras requested by the Graph itself."""
        return self._get_build_result().opset_req | (
            self._extra_opset_req if self._extra_opset_req is not None else set()
        )

    def _get_initializers_by_name(self) -> Dict[str, numpy.ndarray]:
        """Internal function for accessing the initializers by name in the build."""
        return {
            self._get_build_result().scope.var[var]: init
            for var, init in self._get_build_result().initializers.items()
        }

    def get_adapted_nodes(self) -> Dict[Node, Tuple[onnx.NodeProto, ...]]:
        """
        Do a best-effort at generating NodeProtos of consistent versions, matching ``self.opsets``.
        In essence, the policy is to upgrade to the highest used version.
        This does not attempt to fix too complicated nodes, but should work for inline models and simple single nodes.

        Note that onnx.version_converter only implements conversion for the default domain.
        """
        nodes = self._get_build_result().nodes
        consistent_nodes = nodes.copy()
        for node, protos in nodes.items():
            best_effort = adapt_best_effort(
                node,
                list(protos),
                self.get_opsets(),
                self._get_build_result().scope.var.name_of,
                self._get_build_result().scope.node.name_of,
            )
            consistent_nodes[node] = (
                tuple(best_effort) if best_effort is not None else protos
            )

        return consistent_nodes

    def to_onnx(self, *, concrete: bool = False) -> onnx.GraphProto:
        """Perform the Spox build process, gathering arguments,
        results, nodes and other information.

        - Saves type information for arguments & results.
        - Sets the name of the graph, with defaults if it is not set
          or if it is a subgraphs
        - Saves initializers.
        - Sets the docstring if one is set.

        Note
        ----

        This function returns a `onnx.GraphProto` object! If
        you want to serialize this model in a way that is readable by
        the `onnxruntime`, you should use the `to_onnx_model` method
        instead!

        Returns
        -------
        onnx.GraphProto
            Translation of this Graph into an ONNX GraphProto object.

        """
        if not self.get_results():
            raise ValueError("Attempt to build graph without results.")

        argument_info = [
            var.unwrap_type()._to_onnx_value_info(
                name, concrete=concrete, _traceback_name=f"argument {name} ({var})"
            )
            for name, var in self.get_arguments().items()
        ]
        result_info = [
            var.unwrap_type()._to_onnx_value_info(
                name, concrete=concrete, _traceback_name=f"result {name} ({var})"
            )
            for name, var in self.get_results().items()
        ]

        if self._name:
            name = self._name
        else:
            name = "spox_graph"

        initializer_tensors = [
            from_array(arr, name)
            for name, arr in self._get_initializers_by_name().items()
        ]

        node_protos = itertools.chain.from_iterable(self.get_adapted_nodes().values())
        return onnx.helper.make_graph(
            list(node_protos),
            name,
            argument_info,
            result_info,
            initializer_tensors,
            self._doc_string,
        )

    def to_onnx_model(
        self,
        *,
        producer_name: str = "spox",
        model_doc_string: str = "",
        infer_shapes: bool = False,
        check_model: Union[Literal[0], Literal[1], Literal[2]] = 1,
        ir_version=8,
        concrete: bool = True,
    ) -> onnx.ModelProto:
        """
        Internally, this function first obtains a GraphProto from ``.to_onnx()``. Additionally:

        - Function definitions are collected and built into FunctionProtos.
        - Opset requirements are collected (consistency policy is to use the highest version).
            ONNX only allows one version of each domain per model, so some attempt at conversion of nodes is made.
        - Checks are performed, at the level described by the respective arguments.

        Parameters
        ----------
        producer_name
            Value of the ONNX ModelProto producer name field.
        model_doc_string
            Doc string for the ONNX ModelProto.
        infer_shapes
            If the value is True, the model is passed through `onnx.shape_inference.infer_shapes`.
        check_model
            If the value is at least 1 (default), `onnx.checker.check_model` is executed on the model.
            If it is 2, `full_check` of the `check_model` call is set to `True` (e.g. tests against shape inference).
        concrete
            Whether to raise for non-concrete value infos (like missing shape information).
        Returns
        -------
            Translation of this Graph into an ONNX ModelProto object.
        """

        opsets = self.get_opsets()
        if not opsets:
            raise RuntimeError(
                "ONNX often does not properly handle graphs which are empty, "
                "and this one seems to contain no opset imports (only internal nodes?). "
                "Consider adding an Identity operator if you are just copying arguments."
            )

        opset_req: List[tuple[str, int]] = list(opsets.items())  # type: ignore
        function_protos: Dict[Tuple[str, str], onnx.FunctionProto] = {}
        for fun in self._get_build_result().functions:
            proto = fun.to_onnx_function(extra_opset_req=opset_req)
            if proto is None:
                continue
            key = (proto.domain, proto.name)
            if key in function_protos and proto != function_protos[key]:
                raise RuntimeError(
                    f"Built dependency function {proto.domain}:{proto.name} has two different definitions. "
                    f"Was its implementation non-deterministic or is there a naming collision?"
                )
            function_protos[key] = proto

        model = onnx.helper.make_model(
            self.to_onnx(concrete=concrete),
            producer_name=producer_name,
            doc_string=model_doc_string,
            functions=list(function_protos.values()),
            opset_imports=[
                onnx.helper.make_operatorsetid(domain, version)
                for domain, version in opsets.items()
            ],
            ir_version=ir_version,
        )

        if infer_shapes:
            model = onnx.shape_inference.infer_shapes(model)
        if check_model:
            onnx.checker.check_model(model, full_check=check_model >= 2)
        return model


def results(**kwargs: Var) -> Graph:
    """
    Use this function to construct a ``Graph`` object.

    Parameters
    ----------
    kwargs
        Vars to be marked as results in the created Graph.
    Returns
    -------
    Graph
        Graph with the results given in `kwargs`, in the same order. Keys are used as names for the results.
    """
    return Graph(kwargs)


def enum_results(*vars: Var, prefix="out") -> Graph:
    """
    Use this function to construct a ``Graph`` object, whenever the exact names are not important.
    Useful when creating subgraphs.

    Parameters
    ----------
    vars
        Vars to be marked as results.
    prefix
        String to prefix the names of created results with.
    Returns
    -------
        Graph with the results given in `vars`, in the same order.
        Names are the `prefix` with an enumeration index at the end.
    """
    return results(**{f"{prefix}{i}": var for i, var in enumerate(vars)})


def subgraph(types: Iterable[Type], fun: Callable[..., Iterable[Var]]) -> Graph:
    """
    Convenience function for creating a subgraph, for use in an operator like If or Loop.
    However, for those operators one may prefer to use alternative constructors like ``xif`` or ``xloop``
    (which use this function internally).

    Parameters
    ----------
    types
        A list of argument types for the subgraph.
    fun
        A function taking as many Var arguments as the length of `types`, and returning the results of the subgraph.
    Returns
    -------
    Graph
        Graph with results based on the return value of `fun`.
    """
    if not (
        isinstance(types, Iterable) and all(isinstance(typ, Type) for typ in types)
    ):
        raise TypeError("Subgraph input types must be an Iterable of Type.")
    ins = enum_arguments(*types)
    for var in ins:
        var._rename(None)
    if not callable(fun):
        raise TypeError("Subgraph callback must be callable.")
    outs = fun(*ins)
    if not (isinstance(outs, Iterable) and all(isinstance(out, Var) for out in outs)):
        raise TypeError("Subgraph result must be an Iterable of Var.")
    return enum_results(*outs).with_arguments(*ins)._with_constructor(fun)
