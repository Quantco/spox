import dataclasses
import enum
import itertools
import traceback
import typing
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import onnx

from ._attributes import AttrGraph
from ._debug import STORE_TRACEBACK
from ._exceptions import InferenceWarning
from ._fields import BaseAttributes, BaseInputs, BaseOutputs, VarFieldKind
from ._type_system import Type
from ._value_prop import PropValue, PropValueType
from ._var import Var

if typing.TYPE_CHECKING:
    from ._graph import Graph
    from ._scope import Scope


class TypeWarningLevel(enum.IntEnum):
    """
    None - no type check warnings at all
    Critical - warn on missing types (type is None)
    Initial - warn on incomplete types, but only when all the input types were known
    Outputs - warn on all output types that are incomplete (or missing)
    """

    NONE = 0
    CRITICAL = 1
    INITIAL = 2
    OUTPUTS = 3


_TYPE_WARNING_LEVEL: TypeWarningLevel = TypeWarningLevel.INITIAL


@dataclass(frozen=True)
class OpType:
    """Stores information on an ONNX operator, like its identifier and domain."""

    identifier: str
    domain: str
    version: int


class Node(ABC):
    """
    Abstract base class for representing operators in the Spox graph, both standard ONNX and some internal.
    Should not be created directly - proper instances are created by various operator constructors internally.

    Subclasses may specify ``Attributes``, ``Inputs`` and ``Outputs``,
    which must be marked as ``@dataclass`` and inherit from
    ``BaseAttributes``, ``BaseInputs`` and ``BaseOutputs``, respectively.

    Additionally, a subclass may hint that its definitions of the
    above classes are types of ``attrs``, ``inputs`` and ``outputs``.
    This is convenient when accessing these objects in inference routines.

    Names of fields in ``attrs``, order of fields in ``inputs``, ``outputs``,
    as well as all hints are significant during construction and building ONNX.
    Names of fields in ``inputs`` and ``outputs`` impact the ONNX naming.

    ``out_variadic`` is used for the number of outputs of the possible variadic fields, as its length must be explicit.
    """

    op_type: ClassVar[OpType] = OpType("", "", 0)

    Attributes: ClassVar[typing.Type[BaseAttributes]]
    Inputs: ClassVar[typing.Type[BaseInputs]]
    Outputs: ClassVar[typing.Type[BaseOutputs]]

    attrs: BaseAttributes
    inputs: BaseInputs
    outputs: BaseOutputs

    out_variadic: Optional[int]
    _traceback: Union[List[str], None]

    def __init__(
        self,
        attrs: Optional[BaseAttributes] = None,
        inputs: Optional[BaseInputs] = None,
        outputs: Optional[BaseOutputs] = None,
        *,
        out_variadic: Optional[int] = None,
        infer_types: bool = True,
        propagate_values: bool = True,
        validate: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        attrs
            Attributes to set for this Node. If None, initializes Attributes with no parameters.
        inputs
            Inputs (vars) to set for this Node. If None, initializes Inputs with no parameters.
        outputs
            Outputs (vars) to set for this Node, usually left as None. If None, outputs are initialized
            (via _init_output_vars) with no types but set operator to this node.
        out_variadic
            Number of variadic outputs to generate for this node's outputs field in the respective member list.
        infer_types
            Whether to run type inference - setting types for output vars if previously None. Should always succeed
            if possible, possibly raising type errors if inputs/attributes are not correctly typed.
        propagate_values
            Whether to run value propagation - setting values for output vars if previously None. Should only succeed
            if all inputs are constant (attributes always are).
        validate
            Whether to run some extra validation. The default validation only warns against unknown types.
        kwargs
            Extra arguments to pass into ``pre_init`` and ``post_init`` hooks, which may be overriden by child classes.
        """
        self.pre_init(**kwargs)
        self.attrs = attrs if attrs is not None else self.Attributes()
        self.inputs = inputs if inputs is not None else self.Inputs()
        self.out_variadic = out_variadic
        # Initialize output vars - run type inference and value propagation routines
        if not outputs:
            # As inference functions may access which output vars we initialized (e.g. variadics)
            # we inject uninitialized vars first
            self.outputs = self._init_output_vars()
            self.inference(infer_types, propagate_values)
        else:
            self.outputs = outputs

        # Optionally store debug information about where this node was created
        self._traceback = traceback.format_stack() if STORE_TRACEBACK else None

        # Performs type checking using known flags (like type_members)
        # and warns if type inference failed (some types are None).
        if validate:
            self.validate_types()
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

        Some operator schemas may allow not specifying trailing optional inputs with "" (usually represented as None),
        while also requiring you to pass them in anyhow. On the other hand, some operators do not support optional
        inputs via "", so in that case trailing optionals should be removed.

        Hence, we prefer to remove trailing unspecified inputs/outputs whenever possible - as it makes more operator
        schemas work. But we can't remove too much, as then some schemas may not accept the operator.

        As for non-standard operators we want to avoid messing with the output, the default is to not remove anything
        from the tail (minimum inputs is exactly the number of inputs).

        """
        return len(self.inputs)

    @property
    def min_output(self) -> int:
        """
        Sets the minimum number of outputs in the ONNX representation.

        See the docstring for ``min_input`` for a rationale.
        """
        return len(self.outputs)

    @property
    def signature(self) -> str:
        """Get a signature of this Node, including its inputs and attributes (but not outputs)."""

        def fmt_input(key, var):
            return f"{key}: {var.type}" + (
                f" = {var._value}" if var._value is not None else ""
            )

        sign = ", ".join(
            fmt_input(key, var) for key, var in self.inputs.get_vars().items()
        )
        sign = f"inputs [{sign}]"
        shown_attrs = {
            k: v.value for k, v in self.attrs.get_fields().items() if v is not None
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

    def propagate_values(self) -> Dict[str, PropValueType]:
        """
        Propagate values from inputs, and, if possible, compute values for outputs as well.
        This method is used to implement ONNX partial data propagation - for example so that
        we can reshape into a sum of constant vectors.
        """
        return {}

    def infer_output_types(self) -> Dict[str, Type]:
        """
        Inference routine for output types. Often overriden by inheriting Node types.

        Returns a dictionary of output field names into Types for the respective Vars.
        """
        return {}

    def inference(self, infer_types: bool = True, propagate_values: bool = True):
        # Type inference routine - call infer_output_types if required
        # and check if it provides the expected outputs.
        out_types = self.infer_output_types() if infer_types else {}

        for key, var in self.outputs.get_vars().items():
            if var.type is None:  # If no existing type from init_output_vars
                # Attempt to use the ones from kwargs, if none then what type inference gave
                var.type = out_types.get(key)

        # After typing everything, try to get values for outputs
        out_values = self.propagate_values() if propagate_values else {}
        for key, var in self.outputs.get_vars().items():
            if var.type is not None and var._value is None and key in out_values:
                prop = PropValue(var.type, out_values.get(key))
                if prop.check():
                    var._value = prop
                else:
                    warnings.warn(
                        InferenceWarning(
                            f"Propagated value {prop} does not type-check, dropping. "
                            f"Hint: this indicates a bug with the current value prop backend or type inference."
                        )
                    )

    def validate_types(self) -> None:
        """Validation of types, ran at the end of Node creation."""
        if _TYPE_WARNING_LEVEL <= TypeWarningLevel.NONE:
            return
        for name, value_type in self._list_types(self.outputs):
            if value_type is None:
                warnings.warn(
                    InferenceWarning(
                        f"Output type for variable {name} of {self.get_op_repr()} is missing."
                    )
                )
        if _TYPE_WARNING_LEVEL <= TypeWarningLevel.CRITICAL:
            return
        all_inputs_concrete = True
        for name, value_type in self._list_types(self.inputs):
            if self._check_concrete_type(value_type) is not None:
                all_inputs_concrete = False
        if not all_inputs_concrete and _TYPE_WARNING_LEVEL <= TypeWarningLevel.INITIAL:
            return
        for name, value_type in self._list_types(self.outputs):
            msg = self._check_concrete_type(value_type)
            if value_type is not None and msg:
                warnings.warn(
                    InferenceWarning(
                        f"Output type for variable {name} of {self.get_op_repr()} was not concrete - {msg}"
                    ),
                    stacklevel=4,
                )

    def _check_concrete_type(self, value_type: Type) -> Optional[str]:
        if value_type is None:
            return "type is None"
        try:
            value_type._assert_concrete()
        except Exception as e:
            return f"{type(e).__name__}: {str(e)}"
        return None

    def _list_types(self, source):
        return ((key, var.type) for key, var in source.get_vars().items())

    def _init_output_vars(self) -> BaseOutputs:
        """
        Initialize empty output vars bound to this Node and return the respective Fields object.
        Their type is bound in the create method.
        Note: called in ``__init__`` while the state may be partially initialized.
        """
        variadics = {
            field.name
            for field in dataclasses.fields(self.Outputs)
            if self.Outputs._get_field_type(field) == VarFieldKind.VARIADIC
        }
        if variadics:
            (variadic,) = variadics
        else:
            variadic = None
        outputs: Dict[str, Union[Var, Sequence[Var]]] = {
            field.name: Var(self, None, None)
            for field in dataclasses.fields(self.Outputs)
            if field.name != variadic
        }
        if variadic is not None:
            assert self.out_variadic is not None
            outputs[variadic] = [
                Var(self, None, None) for _ in range(self.out_variadic)
            ]
        return self.Outputs(**outputs)  # type: ignore

    @property
    def dependencies(self) -> Iterable[Var]:
        """List of input Vars into this Node."""
        return (var for var in self.inputs.get_vars().values())

    @property
    def dependents(self) -> Iterable[Var]:
        """List of output Vars from this Node."""
        return (var for var in self.outputs.get_vars().values())

    @property
    def incident(self) -> Iterable[Var]:
        """List of both input and output Vars for this Node."""
        return itertools.chain(self.dependencies, self.dependents)

    @property
    def subgraphs(self) -> Iterable["Graph"]:
        for attr in self.attrs.get_fields().values():
            if isinstance(attr, AttrGraph):
                yield attr.value

    def update_metadata(self, opset_req, initializers, functions):
        opset_req.update(self.opset_req)

    def to_onnx(
        self,
        scope: "Scope",
        doc_string: Optional[str] = None,
        build_subgraph: Optional[
            typing.Callable[["Node", str, "Graph"], onnx.GraphProto]
        ] = None,
    ) -> List[onnx.NodeProto]:
        """Translates self into an ONNX NodeProto."""
        assert self.op_type.identifier
        input_names = [scope.var[var] if var is not None else "" for var in self.inputs]
        output_names = [
            scope.var[var] if var is not None else "" for var in self.outputs
        ]
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
        for key, attr in self.attrs.get_fields().items():
            if attr is not None:
                if isinstance(attr, AttrGraph):
                    assert build_subgraph is not None
                    subgraph = build_subgraph(self, key, attr.value)
                    attr_proto = onnx.helper.make_attribute(key, subgraph)
                else:
                    attr_proto = attr._to_onnx()
                node_proto.attribute.append(attr_proto)

        return [node_proto]
