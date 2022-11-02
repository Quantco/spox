import importlib.resources
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import jinja2
import onnx

from steelix.schemas import DOMAIN_VERSIONS, SCHEMAS

DEFAULT_DOMAIN = "ai.onnx"

CONSTRUCTOR_RENAMES = {
    "if": "if_",
    "or": "or_",
    "and": "and_",
    "not": "not_",
    "max": "maximum",
    "min": "minimum",
    "is_na_n": "isnan",
}

# Mapping from attribute proto type integers to Python types.
ATTRIBUTE_PROTO_TO_INPUT_TYPE = {
    onnx.AttributeProto.FLOAT: "float",
    onnx.AttributeProto.INT: "int",
    onnx.AttributeProto.STRING: "str",
    onnx.AttributeProto.TENSOR: "ndarray",
    onnx.AttributeProto.GRAPH: "Graph",
    onnx.AttributeProto.TYPE_PROTO: "Type",
    onnx.AttributeProto.FLOATS: "Iterable[float]",
    onnx.AttributeProto.INTS: "Iterable[int]",
    onnx.AttributeProto.STRINGS: "Iterable[str]",
    onnx.AttributeProto.TENSORS: "Iterable[ndarray]",
    onnx.AttributeProto.TYPE_PROTOS: "Iterable[steelix.Type]",
}

ATTRIBUTE_PROTO_TO_MEMBER_TYPE = {
    onnx.AttributeProto.FLOAT: "AttrFloat32",
    onnx.AttributeProto.INT: "AttrInt64",
    onnx.AttributeProto.STRING: "AttrString",
    onnx.AttributeProto.TENSOR: "AttrTensor",
    onnx.AttributeProto.GRAPH: "AttrGraph",
    onnx.AttributeProto.TYPE_PROTO: "AttrType",
    onnx.AttributeProto.FLOATS: "AttrFloat32s",
    onnx.AttributeProto.INTS: "AttrInt64s",
    onnx.AttributeProto.STRINGS: "AttrStrings",
    onnx.AttributeProto.TENSORS: "AttrTensors",
    onnx.AttributeProto.TYPE_PROTOS: "AttrTypes",
}

_TEMPLATE_DIR = Path(str(importlib.resources.path("steelix", "."))).parent / "templates"


@dataclass
class Attribute:
    # The name of the attribute used as argument and member name
    name: str
    # Default value used in the constructor function
    constructor_default: Optional[str]
    # Type hint used in the constructor function.  May be wrapped in `Optional`.
    constructor_type_hint: str
    # Member type without a potential ``Optional`` wrapper
    _member_type: str

    @property
    def member_type(self) -> str:
        """Type used in the ``Attribute`` class. May be wrapped in `Optional`."""
        if self.constructor_default == "None":
            return f"Optional[{self._member_type}]"
        return self._member_type

    @property
    def attr_constructor(self) -> str:
        """Constructor of for the corresponding ``Attr*`` class."""
        return self._member_type


def get_attributes(schema: onnx.defs.OpSchema, attr_type_overrides) -> List[Attribute]:
    out = []
    for name, attr in schema.attributes.items():
        default = _get_default_value(attr, attr_type_overrides)
        # Special case; not supported
        if attr.type == onnx.AttributeProto.SPARSE_TENSOR:
            continue

        if py_and_attr_type := attr_type_overrides.get(name):
            constructor_type_hint, member_type = (str(el) for el in py_and_attr_type)
        else:
            constructor_type_hint = ATTRIBUTE_PROTO_TO_INPUT_TYPE[attr.type]
            member_type = ATTRIBUTE_PROTO_TO_MEMBER_TYPE[attr.type]

        if default == "None":
            constructor_type_hint = f"Optional[{constructor_type_hint}]"

        out.append(
            Attribute(
                name=name,
                _member_type=member_type,
                constructor_type_hint=constructor_type_hint,
                constructor_default=default,  # type: ignore
            )
        )
    return out


def _get_default_value(attr, attr_type_overrides) -> Optional[str]:
    """Get default value if any as a string ready to be used in a template.

    This function has a special handling with respect to ``attr_type_overrides``.
    """
    if attr.required:
        return None

    default = (
        onnx.helper.get_attribute_value(attr.default_value)
        if attr.default_value.type
        else None
    )

    if default is None:
        return "None"

    # We want to use e.g. np.int32 instead of an integer for dtypes
    if (
        attr.name in attr_type_overrides
        and "numpy.generic" in attr_type_overrides[attr.name][0]
    ):
        return f"numpy.{onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[default].name}"

    # Strings are bytes at this point and they
    # need to be wrapped in quotes.
    if attr.type == onnx.AttributeProto.STRING:
        default = f'"{default.decode()}"'
    elif attr.type == onnx.AttributeProto.STRINGS:
        default = tuple([val.decode() for val in default])
    elif isinstance(default, list):
        default = tuple(default)

    return str(default)


def get_constructor_name(string: str) -> str:
    """Jinja filter. Returns the name of a constructor given the base name of an operator."""
    lim = [
        i
        for i, (x, y) in enumerate(zip(string[1:], string[2:]), 2)
        if not x.isupper() and y.isupper()
    ]
    lim = [0] + lim + [len(string)]
    result = "_".join([string[i:j] for i, j in zip(lim, lim[1:])]).lower()
    if result in CONSTRUCTOR_RENAMES:
        result = CONSTRUCTOR_RENAMES[result]
    return result


def get_constructor_return(schema: onnx.defs.OpSchema) -> str:
    """Jinja filter. Returns the return type hint for a constructor given the operator schema."""
    if not schema.outputs:
        return "None"
    if len(schema.outputs) > 1:
        return f"_{schema.name}.Outputs"
    (out,) = schema.outputs
    if is_variadic(out):
        return "Sequence[Arrow]"
    return "Arrow"


def format_github_markdown(doc: str) -> str:
    """Jinja filter. Makes some attempt at fixing "Markdown" into RST."""
    lines = [line.replace("\t", " " * 4).rstrip() for line in doc.splitlines()]
    lines = [line for line in lines if line.rstrip()]
    space_lcm = 0
    while lines and all(line[: space_lcm + 1].isspace() for line in lines):
        space_lcm += 1
    lines = [line[space_lcm:] for line in lines]
    doc = "\n".join(lines).strip()
    doc = doc.replace("<br>", "\n\n")
    doc = re.sub(r"<i>(.*)</i>", r"`\1`", doc)
    doc = re.sub(r"<b>(.*)</b>", r"**\1**", doc)
    doc = re.sub(r"\[(.+)\]\((.+)\)", r"\1 (\2)", doc)
    return doc


def is_variadic(param: onnx.defs.OpSchema.FormalParameter) -> bool:
    """Check if a given parameter is variadic (accepts 'any' number of inputs, like a list)"""
    return param.option == onnx.defs.OpSchema.FormalParameterOption.Variadic


def is_optional(param: onnx.defs.OpSchema.FormalParameter) -> bool:
    """Check if a given parameter is optional (may or may not be passed in/skipped)."""
    return param.option == onnx.defs.OpSchema.FormalParameterOption.Optional


def get_env():
    """
    Construct the Jinja environment for the generator.
    Exposes some functions from the global scope of this script to the templating engine.
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        extensions=["jinja2.ext.do"],
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["format_github_markdown"] = format_github_markdown
    env.filters["get_constructor_name"] = get_constructor_name
    env.globals["is_variadic"] = is_variadic
    env.globals["is_optional"] = is_optional
    env.globals["get_constructor_return"] = get_constructor_return
    env.globals["get_attributes"] = get_attributes
    return env


def write_schemas_code(
    file,
    schemas: List[onnx.defs.OpSchema],
    type_inference: Dict[str, str],
    value_propagation: Dict[str, str],
    out_variadic_solutions: Dict[str, str],
    attr_type_overrides: List[Tuple[Optional[str], str, Tuple[str, str]]],
    extras: Sequence[str],
    gen_docstrings: bool,
) -> None:
    """Write code for all of ``schemas`` to ``file``. Uses parameters as documented in ``main``."""
    env = get_env()

    preamble, class_, constructor = (
        env.get_template(f"{key}.jinja2")
        for key in ("preamble", "class", "constructor")
    )

    schemas = [s for s in schemas]

    # Preamble
    print(preamble.render(), file=file, end="\n\n\n")

    built_names = set()

    # Operator classes
    for schema in sorted(schemas, key=lambda s: s.name):
        if not schema.has_type_and_shape_inference_function:
            print(schema.name, ":(")
        # Override for attribute type
        attr_over = {
            name: target
            for key, name, target in attr_type_overrides
            if key is None or key == schema.name
        }
        # Type inference
        inf = (
            env.get_template(f"type_inference/{type_inference[schema.name]}.jinja2")
            if schema.name in type_inference
            else None
        )
        # Value propagation
        prop = (
            env.get_template(
                f"value_propagation/{value_propagation[schema.name]}.jinja2"
            )
            if schema.name in value_propagation
            else None
        )

        print(
            class_.render(
                schema=schema,
                type_inference=inf,
                value_propagation=prop,
                attr_type_overrides=attr_over,
            ),
            file=file,
            end="\n\n",
        )
        built_names.add(schema.name)

    # Operator constructors
    for schema in sorted(schemas, key=lambda s: s.name):
        if schema.name not in built_names:
            continue
        # Output variadic solution
        var_sol = out_variadic_solutions.get(schema.name)
        # Override for attribute type
        attr_over = {
            name: target
            for key, name, target in attr_type_overrides
            if key is None or key == schema.name
        }
        print(
            constructor.render(
                schema=schema,
                gen_docstrings=gen_docstrings,
                out_variadic_solution=var_sol,
                attr_type_overrides=attr_over,
            ),
            file=file,
            end="\n\n\n",
        )

    # Extras
    for key in extras:
        extra = env.get_template(f"extras/{key}.jinja2")
        print(extra.render(), file=file, end="\n\n\n")

    print(
        env.get_template("summary.jinja2").render(built_names=sorted(built_names)),
        file=file,
        end="\n",
    )


def run_pre_commit_hooks(filenames: Union[str, Iterable[str]]):
    """
    Calls repo pre-commit hooks for the given ``filenames``.
    Used by the script to verify that the generated files as valid, as usually failing pre-commit hooks
    mean that generation is bugged.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    return subprocess.run(
        f"pre-commit run --files {' '.join(filenames)} --color always", shell=True
    )


def main(
    domain: str,
    version: Optional[int] = None,
    type_inference: Optional[Dict[str, str]] = None,
    value_propagation: Optional[Dict[str, str]] = None,
    out_variadic_solutions: Optional[Dict[str, str]] = None,
    attr_type_overrides: Optional[
        List[Tuple[Optional[str], str, Tuple[str, str]]]
    ] = None,
    extras: Sequence[str] = (),
    target: str = "src/steelix/opset/",
    pre_commit_hooks: bool = True,
    gen_docstrings: bool = True,
):
    """
    Generate opset module code and save it in a `.py` source code file.

    Parameters
    ----------
    domain
        Domain to generate the opset for, like "" ("ai.onnx"), "ai.onnx.ml".
        The module is saved under the directory spelt out by the domain path.
    version
        Target version to generate the opset. The module is named ``v{version}.py``.
    type_inference
        Plugins for type inference, when they are missing in ONNX, for example in If or Loop.
        Keys are operator names, values are names of templates under ``jinja_templates/type_inference/``.
    value_propagation
        Plugins for value propagation, like for Constant, as we don't use ONNX partial data propagation.
        Keys are operator names, values are names of templates under ``jinja_templates/value_propagation/``.
    out_variadic_solutions
        String for an expression that evaluates to the number of outputs of a variadic-output operator.
        Evaluated within the constructor.
        Keys are operator names, values are the expression strings.
    attr_type_overrides
        List of replacements for constructor-level types.
        For example, in Cast the ``to`` attribute accepts ``numpy.generic`` instead of ``int``.
        First element is the name of an operator or None, if an override applies to all operators with an attribute.
        Second element is the attribute name to override the type for.
        Last element is a tuple where the first element is the user
        facing type hint used in the constructor function and the
        second element is the Attr type used internally.
    extras
        List of template names under ``jinja_templates/extras/`` to add at the end of the code.
        This includes convenience functions that may use the rest of the operator set.
    target
        Based directory to save the generated operator set file (not counting subdirectory from ``domain``).
    pre_commit_hooks
        Whether to call the pre-commit hooks on the generated code.
    gen_docstrings
        Whether to generate docstrings for the operator constructors.
    """
    if type_inference is None:
        type_inference = {}
    if value_propagation is None:
        value_propagation = {}
    if out_variadic_solutions is None:
        out_variadic_solutions = {}
    if attr_type_overrides is None:
        attr_type_overrides = []

    onnx_domain = domain if domain != DEFAULT_DOMAIN else ""
    if version is None:
        version = max(DOMAIN_VERSIONS[onnx_domain])
    schemas = list(SCHEMAS[onnx_domain][version].values())

    domain_path = "/".join(domain.split("."))
    path = Path(target) / Path(domain_path) / Path(f"v{version}.py")
    path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Writing {domain}@{version} to `{path}`...")

    with path.open("w") as file:
        write_schemas_code(
            file,
            schemas,
            type_inference,
            value_propagation,
            out_variadic_solutions,
            attr_type_overrides,
            extras,
            gen_docstrings,
        )

    print("Done!\n")

    if pre_commit_hooks:
        print("Running pre-commit hooks to format & verify...")
        if run_pre_commit_hooks(str(path)).returncode:
            print("Running second pass of pre-commit hooks...")
            if run_pre_commit_hooks(str(path)).returncode:
                raise RuntimeError(
                    "Pre-commit hooks failed twice. Is the generated code malformed?"
                )
        print("Done!")


if __name__ == "__main__":
    main(
        "ai.onnx",
        17,
        extras=["const", "xloop", "xif", "promote"],
        type_inference={},
        value_propagation={"Constant": "constant13"},
        out_variadic_solutions={
            "If": "len(else_branch.requested_results)",
            "Loop": "len(body.requested_results) - 1",
        },
        attr_type_overrides=[
            (None, "dtype", ("typing.Type[numpy.generic]", "AttrDtype")),
            ("Cast", "to", ("typing.Type[numpy.generic]", "AttrDtype")),
        ],
    )
    main(
        "ai.onnx.ml",
        3,
        attr_type_overrides=[
            (None, "dtype", ("typing.Type[numpy.generic]", "AttrDtype"))
        ],
        type_inference={
            "ArrayFeatureExtractor": "arrayfeatureextractor1",
            "Binarizer": "binarizer1",
            "CategoryMapper": "categorymapper1",
            "Imputer": "imputer1",
            "Normalizer": "normalizer1",
            "OneHotEncoder": "onehotencoder1",
            "Scaler": "scaler1",
            "TreeEnsembleRegressor": "treeensembleregressor3",
        },
    )
