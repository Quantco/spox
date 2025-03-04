# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar, overload

from ._node import Node
from ._var import _VarInfo

H = TypeVar("H", bound=Hashable)


class ScopeError(Exception):
    """Represents an error related to mishandling a scope, like overwriting or accessing a missing entry."""

    pass


class ScopeSpace(Generic[H]):
    """
    Represents the namespace of a scope for some type H, like ``Node`` or ``_VarInfo``.

    Methods (and operators) on the namespace work both ways: both with names (str) and the named type (H).
    So ``__getitem__`` (``ScopeSpace[item]``) may be used for both the name of an object and the object of a name.
    """

    name_of: dict[H, str]
    of_name: dict[str, H]
    reserved: set[str]
    base_name_counters: dict[str, int]
    parent: ScopeSpace[H] | None

    def __init__(
        self,
        name_of: dict[H, str] | None = None,
        of_name: dict[str, H] | None = None,
        reserved: set[str] | None = None,
        parent: ScopeSpace[H] | None = None,
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
        # Reference to a single `base_name_counters` object across
        # all scopes.
        # While the standard is more lenient in this respect, we
        # simply don't allow any name reuse across the entire model.
        self.base_name_counters = (
            parent.base_name_counters if parent is not None else dict()
        )

    def __contains__(self, item: str | H) -> bool:
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

    def __getitem__(self, item: str | H) -> str | H:
        """Access the name of an object or an object with a given name in this (or outer) namespace."""
        if self.parent is not None and item in self.parent:
            return self.parent[item]
        elif isinstance(item, str):
            return self.of_name[item]
        else:
            return self.name_of[item]

    @overload
    def __setitem__(self, key: str, value: H) -> None: ...

    @overload
    def __setitem__(self, key: H, value: str) -> None: ...

    def __setitem__(self, _key: str | H, _value: H | str) -> None:
        """Set the name of an object in exactly this namespace. Both ``[name] = obj`` and ``[obj] = name`` work."""
        if isinstance(_value, str):
            _key, _value = _value, _key
        assert isinstance(_key, str)
        key: str = _key
        value: H = _value  # type: ignore
        if key in self and self[key] != value:
            raise ScopeError(
                f"Failed to name {value}, as its name {key} "
                f"was already taken by {self[key]}."
            )
        if value in self:
            if key != self[value]:
                raise ScopeError(
                    f"Attempt to implicitly rename {value} to {key} from {self[value]}."
                )
            return
        self.of_name[key] = value
        self.name_of[value] = key

    def __delitem__(self, item: str | H) -> None:
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

    def enum(self, base: str) -> str:
        """Find an unused name by enumerating the pattern ``f"{base}_{i}"`` through `i = 0, 1, ...`"""
        self.base_name_counters.setdefault(base, 0)

        name = f"{base}_{self.base_name_counters[base]}"
        self.base_name_counters[base] = self.base_name_counters[base] + 1
        return name

    def maybe_enum(self, base: str) -> str:
        """Attempt to use ``base`` as a name, or return the result of ``self.enum`` for it otherwise."""
        if base not in self.base_name_counters:
            self.base_name_counters[base] = 0
            return base
        return self.enum(base)

    def reserve(self, name: str) -> str:
        if name in self:
            raise ScopeError(f"Reserved name is already in use: {name}")
        self.reserved.add(name)
        return name


class Scope:
    """
    Class representing the state of an ONNX-rules scope.

    Has namespaces (represented by a ScopeSpace) for VarInfos and Nodes.
    """

    var: ScopeSpace[_VarInfo]
    node: ScopeSpace[Node]

    def __init__(
        self,
        sub_var: ScopeSpace[_VarInfo] | None = None,
        sub_node: ScopeSpace[Node] | None = None,
        parent: Scope | None = None,
    ):
        self.var = sub_var if sub_var is not None else ScopeSpace()
        self.node = sub_node if sub_node is not None else ScopeSpace()
        if parent is not None:
            self.var = ScopeSpace(self.var.name_of, self.var.of_name, parent=parent.var)
            self.node = ScopeSpace(
                self.node.name_of, self.node.of_name, parent=parent.node
            )

    @classmethod
    def of(
        cls,
        *what: tuple[str, _VarInfo | Node] | tuple[_VarInfo | Node, str],
    ) -> Scope:
        """Convenience constructor for filling a Scope with known names."""
        scope = cls()
        for key, value in what:
            if not isinstance(key, str):
                key, value = value, key
            assert isinstance(key, str)
            if isinstance(value, _VarInfo):
                scope.var[key] = value
            elif isinstance(value, Node):
                scope.node[key] = value
            else:
                raise TypeError(f"Unknown value type for Scope.of: {type(value)}")
        return scope

    def update(self, node: Node, prefix: str = "", force: bool = True) -> None:
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
            What value to prefix the node name with. If the VarInfo has a predeclared name, it does not get the prefix.
        force
            Whether to attempt to overwrite existing names (possibly raising a ScopeError if they were different).
            By default, this is set to True to be more strict, so we see if the scoping algorithm failed to only
            introduce a node once where it is needed.
        """
        if force or node not in self.node:
            self.node[node] = self.node.enum(prefix + node.op_type.identifier)
        for field, arr in node.outputs.get_var_infos().items():
            if arr._name is None:
                base = f"{self.node[node]}_{field}"
                name = self.var.maybe_enum(base)
            else:
                name = arr._name
            if force or arr not in self.var:
                self.var[arr] = name
