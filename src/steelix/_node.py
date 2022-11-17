import itertools
import traceback
import typing
from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import onnx

from ._arrow import Arrow
from ._arrowfields import ArrowFields
from ._attributes import AttrGraph
from ._fields import Fields
from ._type_inference import _warn_unknown_types, get_hint
from ._type_system import Type

if typing.TYPE_CHECKING:
    from ._graph import Graph
    from ._scope import Scope


FieldsT = TypeVar("FieldsT", bound=Fields)
NodeT = TypeVar("NodeT", bound="Node")
T = TypeVar("T")


@dataclass(frozen=True)
class OpType:
    """Stores information on an ONNX operator, like its identifier and domain."""

    identifier: str
    domain: str
    version: int


class Dataclass(Protocol):
    __dataclass_fields__: Dict


class Node(ABC):
    """
    Abstract base class for representing operators in the Steelix graph, both standard ONNX and some internal.
    Should not be created directly - proper instances are created by various operator constructors internally.

    When subclassing, ``ArrowFields`` subtypes must be hinted in
    ``inputs``, ``outputs``. These hints by default specify results of
    ``in_type``, ``out_type``. ``Attributes`` must be a ``dataclass``
    where its members are subclasses of :class:`steelix._attributes.Attr`.

    Names of fields in ``attrs`` and order of fields in ``inputs``, ``outputs`` are interpreted during building (ONNX).

    Note that type hints in the Fields types have significance and are introspected to run the type inference system,
    and for attributes to perform the right checks and casts.

    ``out_variadic`` is used for the number of outputs of the possible variadic fields, as its length must be explicit.
    """

    op_type: ClassVar[OpType] = OpType("", "", 0)

    Attributes: ClassVar[typing.Type]
    Inputs: ClassVar[typing.Type]
    Outputs: ClassVar[typing.Type]

    attrs: Dataclass
    inputs: ArrowFields
    outputs: ArrowFields

    out_variadic: Optional[int]
    _traceback: List[str]

    def __init__(
        self,
        attrs: Optional[Any] = None,
        inputs: Optional[ArrowFields] = None,
        outputs: Optional[ArrowFields] = None,
        *,
        out_variadic: Optional[int] = None,
        infer_types: bool = True,
        propagate_values: bool = True,
        validate: bool = True,
        warn_unknown: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        attrs
            Attributes to set for this Node. If None, initializes Attributes with no parameters.
        inputs
            Inputs (arrows) to set for this Node. If None, initializes Inputs with no parameters.
        outputs
            Outputs (arrows) to set for this Node, usually left as None. If None, outputs are initialized
            (via _init_output_arrows) with no types but set operator to this node.
        out_variadic
            Number of variadic outputs to generate for this node's outputs field in the respective member list.
        infer_types
            Whether to run type inference - setting types for output arrows if previously None. Should always succeed
            if possible, possibly raising type errors if inputs/attributes are not correctly typed.
        propagate_values
            Whether to run value propagation - setting values for output arrows if previously None. Should only succeed
            if all inputs are constant (attributes always are).
        validate
            Whether to run some extra validation. The default validation only warns against unknown types.
        warn_unknown
            Whether to raise wiarnings when inputs/outputs have unknown (or non-concrete, like without shapes) types.
            Only has an effect when ``validate`` is true.
        kwargs
            Extra arguments to pass into ``pre_init`` and ``post_init`` hooks, which may be overriden by child classes.
        """
        self.pre_init(**kwargs)
        self.attrs = attrs if attrs is not None else self.Attributes()
        self.inputs = inputs if inputs is not None else self.Inputs()
        self.out_variadic = out_variadic
        # Initialize output arrows - run type inference and value propagation routines
        if not outputs:
            # As inference functions may access which output arrows we initialized (e.g. variadics)
            # we inject uninitialized arrows first
            self.outputs = self._init_output_arrows({}, {})
            output_types = self.infer_output_types() if infer_types else {}
            self.outputs = self._init_output_arrows(output_types, {})
            output_values = self.propagate_values() if propagate_values else {}
            self.outputs = self._init_output_arrows(output_types, output_values)
        else:
            self.outputs = outputs
        self._traceback = traceback.format_stack()
        # Performs type checking using known flags (like type_members)
        # and warns if type inference failed (some types are None).
        if validate:
            self.validate_types(warn_unknown)
        self.post_init(**kwargs)

    @property
    def opset_req(self) -> Set[Tuple[str, int]]:
        """
        Set of the opset requirements -- (domain, version) -- brought in by this node.
        Does not include subgraphs.
        """
        return {(self.op_type.domain, self.op_type.version)}

    @property
    def min_input(self) -> int:
        """
        Sets the minimum number of inputs in the ONNX representation.
        Sometimes needed due to issues with interpretation of NodeProto in type inference by ONNX.

        Some operator schemas may allow not specifying trailing optional inputs with "" (represented as _nil/NilArrow),
        while also requiring you to pass them in anyhow. On the other hand, some operators do not support optional
        inputs via "", so in that case trailing optionals should be removed.

        Hence, we prefer to remove trailing unspecified inputs/outputs whenever possible - as it makes more operator
        schemas work. But we can't remove too much, as then some schemas may not accept the operator.

        As for non-standard operators we want to avoid messing with the output, the default is to not remove anything
        from the tail (minimum inputs is exactly the number of inputs).

        """
        return len(self.inputs.as_dict())

    @property
    def min_output(self) -> int:
        """
        Sets the minimum number of outputs in the ONNX representation.

        See the docstring for ``min_input`` for a rationale.
        """
        return len(self.outputs.as_dict())

    @property
    def signature(self) -> str:
        """Get a signature of this Node, including its inputs and attributes (but not outputs)."""

        def fmt_input(key, arrow):
            return f"{key}: {arrow.type}" + (
                f" = {arrow._value}" if arrow._value is not None else ""
            )

        sign = ", ".join(
            fmt_input(key, arrow) for key, arrow in self.inputs.as_dict().items()
        )
        sign = f"inputs [{sign}]"
        shown_attrs = {
            k: v.value for k, v in self.attrs.__dict__.items() if v is not None
        }
        if shown_attrs:
            sign_attrs = ", ".join(f"{k} = {v}" for k, v in shown_attrs.items())
            sign = f"{sign} & attributes [{sign_attrs}]"
        return sign

    @classmethod
    def get_op_repr(cls) -> str:
        """Get a short representation of the ``op_type`` of this Node."""
        domain = cls.op_type.domain if cls.op_type.domain != "" else "ai.onnx"
        return f"{domain}@{cls.op_type.version}::{cls.op_type.identifier}"

    def pre_init(self, **kwargs):
        """Pre-initialization hook. Called during ``__init__`` before any field on the object is set."""

    def post_init(self, **kwargs):
        """Post-initialization hook. Called at the end of ``__init__`` after other default fields are set."""

    def propagate_values(self) -> Dict[str, Any]:
        """
        Propagate values from inputs, and, if possible, compute values for outputs as well.
        This method is used to implement ONNX partial data propagation - for example so that
        we can reshape into a sum of constant vectors.
        """
        return {}

    def infer_output_types(self) -> Dict[str, Type]:
        """
        Inference routine for output types. Often overriden by inheriting Node types.

        Returns a dictionary of output field names into Types for the respective Arrows.
        """
        return {}

    def inference(
        self, infer_types: bool = True, propagate_values: bool = True, **kwargs
    ):
        # Type inference routine - call infer_output_types if required
        # and check if it provides the expected outputs.
        out_types = self.infer_output_types() if infer_types else {}
        out_names = set(self.outputs.get_types())

        for name in out_names:
            arrow = getattr(self.outputs, name)
            if arrow.type is None:  # If no existing type from init_output_arrows
                # Attempt to use the ones from kwargs, if none then what type inference gave
                arrow.type = kwargs.get(name, out_types.get(name))

        # After typing everything, try to get values for outputs
        out_values = self.propagate_values() if propagate_values else {}
        for name in out_names:
            arrow = getattr(self.outputs, name)
            if arrow.value is None:
                arrow.value = out_values.get(name)

    def validate_types(self, warn_unknown: bool = True) -> None:
        """Validation of types, ran at the end of Node creation."""
        if warn_unknown:
            for name, _, value_type in self._type_checks:
                _warn_unknown_types(value_type, name, self.get_op_repr())

    @property
    def _type_checks(self):
        for source in (self.inputs, self.outputs):
            for name, typ in source.get_types().items():
                arrow = getattr(source, name)
                if not arrow:
                    continue
                hint = get_hint(typ)
                value_type = arrow.type
                yield name, hint, value_type

    def _init_output_arrows(
        self, types: Dict[str, Type], values: Dict[str, Any]
    ) -> ArrowFields:
        """
        Initialize empty output arrows bound to this Node and return the respective Fields object.
        Their type is bound in the create method.
        Note: called in ``__init__`` while the state may be partially initialized.
        """

        def arr(name):
            return Arrow(self, types.get(name), values.get(name))

        var = self.Outputs.get_variadic_name()
        outputs: Dict[str, Union[Arrow, Sequence[Arrow]]] = {
            name: arr(name) for name in self.Outputs.get_kwargs() if name != var
        }
        if var is not None:
            assert self.out_variadic is not None
            outputs[var] = [arr(f"{var}_{i}") for i in range(self.out_variadic)]
        return self.Outputs(**outputs)

    @property
    def dependencies(self) -> Iterable[Arrow]:
        """List of input Arrows into this Node."""
        return (arrow for arrow in self.inputs.as_dict().values() if arrow)

    @property
    def dependents(self) -> Iterable[Arrow]:
        """List of output Arrows from this Node."""
        return (arrow for arrow in self.outputs.as_dict().values() if arrow)

    @property
    def incident(self) -> Iterable[Arrow]:
        """List of both input and output Arrows for this Node."""
        return itertools.chain(self.dependencies, self.dependents)

    @property
    def subgraphs(self) -> Iterable["Graph"]:
        for attr in self.attrs.__dict__.values():
            if isinstance(attr, AttrGraph):
                yield attr.value

    def update_metadata(self, opset_req, initializers, functions):
        opset_req.update(self.opset_req)

    def to_onnx(
        self,
        scope: "Scope",
        doc_string: Optional[str] = None,
        build_subgraph: Optional[typing.Callable] = None,
    ) -> List[onnx.NodeProto]:
        """Translates self into an ONNX NodeProto."""
        assert self.op_type.identifier
        input_names = [scope.arrow[arrow] for arrow in self.inputs.as_dict().values()]
        output_names = [scope.arrow[arrow] for arrow in self.outputs.as_dict().values()]
        while len(input_names) > self.min_input and not input_names[-1]:
            input_names.pop()
        while len(output_names) > self.min_output and not output_names[-1]:
            output_names.pop()
        node_proto = onnx.helper.make_node(
            self.op_type.identifier,
            input_names,
            output_names,
            scope.node[self],
            doc_string,
            self.op_type.domain,
        )

        # We add all attributes manually since not everything (e.g. refs) is supported by make_node
        # Subgraphs are also special-cased here
        for key, attr in self.attrs.__dict__.items():
            if attr is not None:
                if isinstance(attr, AttrGraph):
                    assert build_subgraph is not None
                    subgraph = build_subgraph(self, key, attr.value)
                    attr_proto = onnx.helper.make_attribute(key, subgraph)
                else:
                    attr_proto = attr._to_onnx(key)
                node_proto.attribute.append(attr_proto)

        return [node_proto]
