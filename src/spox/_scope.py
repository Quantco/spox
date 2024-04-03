from typing import Dict, Generic, Hashable, Optional, Set, TypeVar, Union, overload

from ._node import Node
from ._var import Var

H = TypeVar("H", bound=Hashable)


class ScopeError(Exception):
    """Represents an error related to mishandling a scope, like overwriting or accessing a missing entry."""

    pass


class ScopeSpace(Generic[H]):
    """
    Represents the namespace of a scope for some type H, like Node or Var.

    Methods (and operators) on the namespace work both ways: both with names (str) and the named type (H).
    So ``__getitem__`` (``ScopeSpace[item]``) may be used for both the name of an object and the object of a name.
    """

    name_of: Dict[H, str]
    of_name: Dict[str, H]
    reserved: Set[str]
    parent: "Optional[ScopeSpace[H]]"

    def __init__(
        self,
        name_of: Optional[Dict[H, str]] = None,
        of_name: Optional[Dict[str, H]] = None,
        reserved: Optional[Set[str]] = None,
        parent: "Optional[ScopeSpace[H]]" = None,
    ):
        """
        Parameters
        ----------
        name_of
            Name of a given object in this namespace.
        of_name
            Object with a given name in this namespace.
        reserved
            Set of reserved names, taken up by anonymous objects.
        parent
            Namespace of a parent scope. Is accessed first before all checks, but never modified.
        """
        self.name_of = name_of.copy() if name_of is not None else {}
        self.of_name = of_name.copy() if of_name is not None else {}
        self.reserved = reserved.copy() if reserved is not None else set()
        self.parent = parent

    def __contains__(self, item: Union[str, H]) -> bool:
        """Checks if a given name or object is declared in this (or outer) namespace."""
        return (
            (self.parent is not None and item in self.parent)
            or item in self.reserved
            or item in self.name_of
            or item in self.of_name
        )

    @overload
    def __getitem__(self, item: H) -> str: ...

    @overload
    def __getitem__(self, item: str) -> H: ...

    def __getitem__(self, item: Union[str, H]):
        """Access the name of an object or an object with a given name in this (or outer) namespace."""
        if self.parent is not None and item in self.parent:
            return self.parent[item]
        elif isinstance(item, str):
            return self.of_name[item]
        else:
            return self.name_of[item]

    @overload
    def __setitem__(self, key: str, value: H): ...

    @overload
    def __setitem__(self, key: H, value: str): ...

    def __setitem__(self, _key, _value):
        """Set the name of an object in exactly this namespace. Both ``[name] = obj`` and ``[obj] = name`` work."""
        if isinstance(_value, str):
            _key, _value = _value, _key
        key: str = _key
        value: H = _value
        assert isinstance(key, str)
        if key in self and self[key] != value:
            raise ScopeError(
                f"Failed to name {value}, as its name {key} "
                f"was already taken by {self[key]}."
            )
        if value in self:
            if key != self[value]:
                raise ScopeError(
                    f"Attempt to implicitly rename {value} to {key} "
                    f"from {self[value]}."
                )
            return
        self.of_name[key] = value
        self.name_of[value] = key

    def __delitem__(self, item: Union[str, H]):
        """Delete a both the name and object from exactly this namespace."""
        if isinstance(item, str):
            key, value = item, self.of_name[item]
        else:
            key, value = self.name_of[item], item
        if key not in self.of_name:
            raise ScopeError(f"Cannot remove missing key: {key}")
        if value not in self.name_of:
            raise ScopeError(f"Cannot remove missing value: {value}")
        del self.of_name[key]
        del self.name_of[value]

    def enum(self, base: str, suffix: str = "_{}") -> str:
        """Find an unused name by enumerating the pattern ``base + suffix.format(i)`` through `i = 0, 1, ...`"""
        i = 0
        while (name := f"{base}{suffix.format(i)}") in self:
            i += 1
        return name

    def maybe_enum(self, base: str, suffix: str = "_{}") -> str:
        """Attempt to use ``base`` as a name, or return the result of ``self.enum`` for it otherwise."""
        if base not in self:
            return base
        return self.enum(base, suffix)

    def reserve(self, name: str) -> str:
        if name in self:
            raise ScopeError(f"Reserved name is already in use: {name}")
        self.reserved.add(name)
        return name


class Scope:
    """
    Class representing the state of an ONNX-rules scope.

    Has namespaces (represented by a ScopeSpace) for Vars and Nodes.
    """

    var: ScopeSpace[Var]
    node: ScopeSpace[Node]

    def __init__(
        self,
        sub_var: Optional[ScopeSpace[Var]] = None,
        sub_node: Optional[ScopeSpace[Node]] = None,
        parent: Optional["Scope"] = None,
    ):
        self.var = sub_var if sub_var is not None else ScopeSpace()
        self.node = sub_node if sub_node is not None else ScopeSpace()
        if parent is not None:
            self.var = ScopeSpace(self.var.name_of, self.var.of_name, parent=parent.var)
            self.node = ScopeSpace(
                self.node.name_of, self.node.of_name, parent=parent.node
            )

    @classmethod
    def of(cls, *what):
        """Convenience constructor for filling a Scope with known names."""
        scope = cls()
        for key, value in what:
            if not isinstance(key, str):
                key, value = value, key
            assert isinstance(key, str)
            if isinstance(value, Var):
                scope.var[key] = value
            elif isinstance(value, Node):
                scope.node[key] = value
            else:
                raise TypeError(f"Unknown value type for Scope.of: {type(value)}")
        return scope

    def update(self, node: Node, prefix: str = "", force: bool = True):
        """
        Function used for introducing a Node and its outputs into the scope in the build routine.

        The node is named by the pattern ``{node.op_type.identifier}_{i}``, where ``i` is a generated index.

        The var is named by the pattern ``{node_name}_{output_field_name}``, unless it has a ``._name`` field set.
        (as arguments and results of the main graph do, for example).

        Parameters
        ----------
        node
            Node to introduce in the scope.
        prefix
            What value to prefix the node name with. If the Var has a predeclared name, it does not get the prefix.
        force
            Whether to attempt to overwrite existing names (possibly raising a ScopeError if they were different).
            By default, this is set to True to be more strict, so we see if the scoping algorithm failed to only
            introduce a node once where it is needed.
        """
        if force or node not in self.node:
            self.node[node] = self.node.enum(prefix + node.op_type.identifier)
        for field, arr in node.outputs.get_vars().items():
            if arr._name is None:
                base = f"{self.node[node]}_{field}"
                name = self.var.maybe_enum(base)
            else:
                name = arr._name
            if force or arr not in self.var:
                self.var[arr] = name
