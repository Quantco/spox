"""Exposes information related to reference ONNX operator schemas, used by StandardOpNode."""

import itertools
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)

from onnx.defs import OpSchema, get_all_schemas_with_history


class _Comparable(Protocol):
    def __lt__(self, other): ...

    def __gt__(self, other): ...


S = TypeVar("S")
K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T", bound=_Comparable)


def _key_groups(seq: Iterable[S], key: Callable[[S], T]):
    """Group a sequence by a given key."""
    return itertools.groupby(sorted(seq, key=key), key)


def _current_schema(
    schemas: Iterable[OpSchema], version: Optional[int] = None
) -> Optional[OpSchema]:
    """
    Find the schema for the current ``version`` from the list (the latest existing version).
    If ``version`` is None (or left to default), the newest of the schemas is returned.
    """
    available = (
        [schema for schema in schemas if schema.since_version <= version]
        if version is not None
        else schemas
    )
    return max(available, key=lambda s: s.since_version) if available else None


def _get_schemas_versioned(
    all_schemas: List[OpSchema],
) -> Dict[str, Dict[str, OpSchema]]:
    """Get a map into a list of schemas for all domain/names."""
    return {
        domain: {
            name: sorted(op_group, key=lambda s: s.since_version)
            for name, op_group in _key_groups(domain_group, lambda s: s.name)
        }
        for domain, domain_group in _key_groups(all_schemas, lambda s: s.domain)
    }


def _get_schemas_map(
    schemas_ver_lists: Dict[str, Dict[str, List[OpSchema]]],
    domain_versions: Dict[str, Set[int]],
) -> Dict[str, Dict[int, Dict[str, OpSchema]]]:
    """Get a map into a schema for every domain/version/name."""
    return {
        domain: {
            version: {
                name: _current_schema(this_schemas, version)
                for name, this_schemas in domain_schemas.items()
                if _current_schema(this_schemas, version)
            }
            for version in range(
                min(domain_versions[domain]), max(domain_versions[domain]) + 1
            )
        }
        for domain, domain_schemas in schemas_ver_lists.items()
    }


ALL_SCHEMAS: List[OpSchema] = get_all_schemas_with_history()

DOMAINS: Set[str] = {s.domain for s in ALL_SCHEMAS}

# Assumes that each version does change at least one of the operators from the available schemes.
DOMAIN_VERSIONS: Dict[str, Set[int]] = {
    domain: {s.since_version for s in ALL_SCHEMAS if s.domain == domain}
    for domain in DOMAINS
}

# SCHEMAS_VER_LISTS[domain][identifier] = [..schemas by version..]
SCHEMAS_VER_LISTS = _get_schemas_versioned(ALL_SCHEMAS)

# SCHEMAS[domain][version][identifier] = [schema]
SCHEMAS = _get_schemas_map(SCHEMAS_VER_LISTS, DOMAIN_VERSIONS)


def max_opset_policy(opset_req: Set[Tuple[str, int]]) -> Dict[str, int]:
    """Use the highest required version for every opset."""
    opset_req = {(k if k != "ai.onnx" else "", v) for k, v in opset_req}
    grouping = itertools.groupby(sorted(opset_req), key=lambda x: x[0])
    return {domain: max(v for _, v in group) for domain, group in grouping}
