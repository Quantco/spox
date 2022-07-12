from typing import ClassVar, Dict, List, Optional, Set, Tuple, TypeVar, Union
from typing import cast as typing_cast

from ._type_inference import InferenceFlag, _resolve_generic, _run_type_checks, get_hint
from .node import Node
from .type_system import Type


class DefNode(Node):
    """
    Subclass of node which uses a Python-based simple type inference system, with definitions based on
    class variables and type annotations in attrs/inputs/outputs Fields.

    This interprets type annotations placed within inputs & outputs.

    - ``Arrow`` annotations generic in ``typing.TypeVar`` have their types matched together.
    - Rules of matching type variables are changed by the ``inference_flags`` dictionary for each type variable.
      - Default only matches tensor element types.
      - ``BROADCAST`` performs shape broadcasting on tensors.
      - ``STRICT`` asserts the shape is the same in every instance of the type.
    - The ``type_members`` dictionary describes upper-bound constraints for the type inferred for a given variable.
      - If there is only one member possible, the type variable is taken to be equal to it by default.
    """

    type_members: ClassVar[Dict[str, Set[Type]]] = {}
    inference_flags: ClassVar[Dict[str, Set[InferenceFlag]]] = {}

    @classmethod
    def get_type_members(cls, t: TypeVar):
        return cls.type_members.get(t.__name__)

    @classmethod
    def get_inference_flags(cls, t: TypeVar):
        return cls.inference_flags.get(t.__name__, set())

    def infer_output_types(self) -> Dict[str, Type]:
        """
        Returns a dictionary of inferred types for output field names (whichever were successful).
        May be overriden by child classes to infer types in a custom way, if existing rules do not support it.
        """
        generics = self.resolve_hints()

        out_types: Dict[str, Type] = {}
        for name, typ in self.outputs.get_types().items():
            if (out_hint := get_hint(typ)) is None:
                continue
            if isinstance(out_hint, Type):  # only from Annotated[Arrow, ...]
                out_types[name] = out_hint
            elif isinstance(out_hint, TypeVar):
                mem = self.get_type_members(out_hint)
                if out_hint.__name__ in generics:
                    out_types[name] = generics[out_hint.__name__]
                elif mem is not None and len(mem) == 1:
                    (out_types[name],) = mem

        return out_types

    def validate_types(self, warn_unknown=True):
        """Collects all input and output fields to typecheck and throws when invalid types are encountered."""
        super().validate_types(warn_unknown=warn_unknown)
        _run_type_checks(self._type_checks, self.get_type_members, self.get_op_repr())

    def resolve_hints(self) -> Dict[str, Type]:
        """
        Collects all input fields, finding assignments for each variable,
        and resolves the matching generic type using ``type_inference.resolve_generic``.
        """
        op_name = type(self).__name__

        assignments: Dict[TypeVar, Dict[str, Optional[Type]]] = {}
        for name, typ in self.inputs.get_types().items():
            in_hint: Union[Type, TypeVar]
            if (in_hint := get_hint(typ)) is None:
                continue
            in_type: Optional[Type] = getattr(self.inputs, name).type
            if not isinstance(in_hint, TypeVar):
                continue
            if in_hint not in assignments:
                assignments[in_hint] = {}
            assignments[in_hint][name] = in_type

        generics: Dict[str, Type] = {}
        for in_var in assignments:
            # Workaround for PyCharm, as it doesn't type list(dict.items()) properly.
            types = typing_cast(
                List[Tuple[str, Optional[Type]]], list(assignments[in_var].items())
            )
            res = _resolve_generic(
                types, self.get_inference_flags(in_var), op_name, in_var
            )
            if res is not None:
                generics[in_var.__name__] = res

        return generics
