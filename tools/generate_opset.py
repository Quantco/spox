import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import jinja2
import onnx

from spox._schemas import DOMAIN_VERSIONS, SCHEMAS

DEFAULT_DOMAIN = "ai.onnx"

CONSTRUCTOR_RENAMES = {
    "if": "if_",
    "or": "or_",
    "and": "and_",
    "not": "not_",
    "is_na_n": "isnan",
    "is_inf": "isinf",
    "mat_mul": "matmul",
    "mat_mul_integer": "matmul_integer",
    "qlinear_mat_mul": "qlinear_matmul",
    "cum_sum": "cumsum",
}
CONSTRUCTOR_ALIASES = {
    "ai.onnx": [
        ("cumsum", "cum_sum"),
    ]
}

# Mapping from attribute proto type integers to Python types.
ATTRIBUTE_PROTO_TO_INPUT_TYPE = {
    onnx.AttributeProto.FLOAT: "float",
    onnx.AttributeProto.INT: "int",
    onnx.AttributeProto.STRING: "str",
    onnx.AttributeProto.TENSOR: "np.ndarray",
    onnx.AttributeProto.GRAPH: "Callable[..., Iterable[Var]]",
    onnx.AttributeProto.TYPE_PROTO: "Type",
    onnx.AttributeProto.FLOATS: "Iterable[float]",
    onnx.AttributeProto.INTS: "Iterable[int]",
    onnx.AttributeProto.STRINGS: "Iterable[str]",
    onnx.AttributeProto.TENSORS: "Iterable[np.ndarray]",
    onnx.AttributeProto.TYPE_PROTOS: "Iterable[Type]",
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

IF16_OUT_VARIADIC_SOLUTION = "len(_else_branch_subgraph.requested_results)"
LOOP16_OUT_VARIADIC_SOLUTION = "len(_body_subgraph.requested_results) - 1"
SCAN16_OUT_VARIADIC_SOLUTION = "len(_body_subgraph.requested_results)"
SEQUENCEMAP17_OUT_VARIADIC_SOLUTION = "len(_body_subgraph.requested_results)"
SPLIT18_OUT_VARIADIC_SOLUTION = "num_outputs"

IF16_SUBGRAPH_SOLUTION = {"else_branch": "()", "then_branch": "()"}
LOOP16_SUBGRAPH_SOLUTION = {
    "body": "typing_cast(List[Type], [Tensor(np.int64, (1,)), Tensor(np.bool_, (1,))])"
    "+ [var.unwrap_type() for var in v_initial]"
}
SCAN16_SUBGRAPH_SOLUTION = {
    "body": "[Tensor(var.unwrap_tensor().dtype, "
    "   (lambda x: x[1:] if x is not None else None)(var.unwrap_tensor().shape)) "
    "for var in initial_state_and_scan_inputs[:num_scan_inputs]] + "
    "[Tensor(var.unwrap_tensor().dtype) "
    "for var in initial_state_and_scan_inputs[num_scan_inputs:]]"
}
SEQUENCEMAP17_SUBGRAPH_SOLUTION = {
    "body": "[typing_cast(SpoxSequence, input_sequence.unwrap_type()).elem_type] + "
    "[typing_cast(SpoxSequence, var.unwrap_type()).elem_type for var in additional_inputs]"
}

V16_OUT_VARIADIC_SOLUTIONS = {
    "If": IF16_OUT_VARIADIC_SOLUTION,
    "Loop": LOOP16_OUT_VARIADIC_SOLUTION,
    "Scan": SCAN16_OUT_VARIADIC_SOLUTION,
    "SequenceMap": SEQUENCEMAP17_OUT_VARIADIC_SOLUTION,
}
V18_OUT_VARIADIC_SOLUTIONS = {
    **V16_OUT_VARIADIC_SOLUTIONS,
    "Split": SPLIT18_OUT_VARIADIC_SOLUTION,
}
V16_SUBGRAPH_SOLUTIONS = {
    "If": IF16_SUBGRAPH_SOLUTION,
    "Loop": LOOP16_SUBGRAPH_SOLUTION,
    "Scan": SCAN16_SUBGRAPH_SOLUTION,
    "SequenceMap": SEQUENCEMAP17_SUBGRAPH_SOLUTION,
}
DEFAULT_ATTR_TYPE_OVERRIDES = [
    (None, "dtype", ("npt.DTypeLike", "AttrDtype")),
    ("Cast", "to", ("npt.DTypeLike", "AttrDtype")),
    ("If", "then_branch", ("Callable[[], Iterable[Var]]", "AttrGraph")),
    ("If", "else_branch", ("Callable[[], Iterable[Var]]", "AttrGraph")),
    ("Split", "num_outputs", ("int", "AttrInt64")),
]

_TEMPLATE_DIR = Path(__file__).parent / "templates/"


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
    # Python expression for generating the argument types for this subgraph
    subgraph_solution: Optional[str] = None
    # Mark whether generating extra constructor arguments caused by this should raise
    allow_extra: bool = False

    def __post_init__(self):
        if self.attr_constructor != "AttrGraph" and self.subgraph_solution is not None:
            raise TypeError(
                "Subgraph input types should only be specified for an AttrGraph."
            )
        if (
            self.attr_constructor == "AttrGraph"
            and self.subgraph_solution is None
            and not self.allow_extra
        ):
            raise RuntimeError(
                f"Attribute {self.name} is a subgraph but no input types solution was provided. "
                f"An argument {self.name}_input_types would be generated if it was in the allow list."
            )

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


def get_attributes(
    schema: onnx.defs.OpSchema,
    attr_type_overrides,
    subgraph_solutions: Dict[str, str],
    allow_extra: bool,
) -> List[Attribute]:
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
                subgraph_solution=subgraph_solutions.get(name),
                allow_extra=allow_extra,
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
        and attr_type_overrides[attr.name][0] == "npt.DTypeLike"
    ):
        return f"np.{onnx.helper.tensor_dtype_to_np_dtype(default).name}"

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
        return f"Tuple[{', '.join('Sequence[Var]' if is_variadic(out) else 'Var' for out in schema.outputs)}]"
    (out,) = schema.outputs
    if is_variadic(out):
        return "Sequence[Var]"
    return "Var"


_PANDOC_SEP = "\U0001f6a7"  # U+1F6A7 CONSTRUCTION SIGN
_PANDOC_GFM_TO_RST_CACHE: Dict[str, str] = {}


def _pandoc_run(text: str):
    return subprocess.run(
        ["pandoc", "--from=gfm", "--to=rst"], input=text.encode(), capture_output=True
    ).stdout.decode()


def _pandoc_gfm_to_rst_run(*args: str) -> Tuple[str, ...]:
    if not args:
        return ()

    sep = f"\n\n{_PANDOC_SEP}{_PANDOC_SEP}\n\n"
    acc = sep.join([_PANDOC_SEP] + list(args) + [_PANDOC_SEP])
    acc_results = _pandoc_run(acc)
    _, *results, _ = acc_results.split(sep)
    for arg, result in zip(args, results):
        if _PANDOC_SEP in result:
            raise ValueError(
                f"Pandoc separator character '{_PANDOC_SEP}' found in a result (bad convert)."
            )
        _PANDOC_GFM_TO_RST_CACHE[arg] = result + "\n"
    return results


def _pandoc_gfm_to_rst(*args: str) -> Tuple[str, ...]:
    args = tuple(arg.strip() for arg in args)
    if any(_PANDOC_SEP in arg for arg in args):
        raise ValueError(
            f"Pandoc separator character '{_PANDOC_SEP}' cannot appear in any of the arguments."
        )
    valid = [
        i
        for i, arg in enumerate(args)
        if not (arg in _PANDOC_GFM_TO_RST_CACHE or not arg)
    ]
    results = _pandoc_gfm_to_rst_run(*[args[i] for i in valid])
    sub: List[Optional[str]] = [None] * len(args)
    for i, result in zip(valid, results):
        sub[i] = result
    for i, arg in enumerate(args):
        if not arg:
            sub[i] = ""
        elif arg in _PANDOC_GFM_TO_RST_CACHE:
            sub[i] = _PANDOC_GFM_TO_RST_CACHE[arg]
    if any(r is None for r in sub):
        raise ValueError("Missing processed pandoc result.")
    return tuple(sub)  # type: ignore


def pandoc_gfm_to_rst(doc: str) -> str:
    (result,) = _pandoc_gfm_to_rst(doc)
    return result


def format_github_markdown(doc: str, *, to_batch: Optional[List[str]] = None) -> str:
    """Jinja filter. Makes some attempt at fixing "Markdown" into RST."""
    # Sometimes Tensor<T> is used in the docs (~17 instances at 1.13)
    # and is treated as invalid HTML tags by pandoc.
    doc = doc.replace("<T>", "&lt;T&gt;")
    # Point hyperlinks to onnx/docs
    rel = "https://github.com/onnx/onnx/blob/main/docs"
    doc = re.sub(
        r"\[(.*)]\((\w+.md)\)", lambda match: f"[{match[1]}]({rel}/{match[2]})", doc
    )
    if to_batch is not None:
        to_batch.append(doc)
        return doc
    else:
        return pandoc_gfm_to_rst(doc).rstrip()


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
    domain: str,
    schemas: List[onnx.defs.OpSchema],
    type_inference: Dict[str, str],
    value_propagation: Dict[str, str],
    out_variadic_solutions: Dict[str, str],
    subgraphs_solutions: Dict[str, Dict[str, str]],
    attr_type_overrides: List[Tuple[Optional[str], str, Tuple[str, str]]],
    allow_extra_constructor_arguments: Set[str],
    inherited_schemas: Dict[onnx.defs.OpSchema, str],
    extras: Sequence[str],
    gen_docstrings: bool,
):
    """Write code for all of ``schemas`` to ``file``. Uses parameters as documented in ``main``."""
    env = get_env()

    preamble, class_, constructor, inherit = (
        env.get_template(f"{key}.jinja2")
        for key in ("preamble", "class", "constructor", "inherit")
    )

    schemas = [s for s in schemas]

    # Preamble
    print(preamble.render(), file=file, end="\n\n\n")

    for schema in sorted(schemas, key=lambda s: s.name):
        if schema in inherited_schemas:
            print(
                inherit.render(schema=schema, module=inherited_schemas[schema]),
                file=file,
                end="\n",
            )

    built_schemas: Set[onnx.defs.OpSchema] = set()

    pandoc_batch: List[str] = []
    for schema in schemas:
        if schema in inherited_schemas:
            continue
        todo = [schema.doc] + [
            p.description
            for p in (
                list(schema.inputs)
                + list(schema.outputs)
                + list(schema.attributes.values())
            )
        ]
        for doc in todo:
            format_github_markdown(doc, to_batch=pandoc_batch)
    _pandoc_gfm_to_rst(*pandoc_batch)

    # Operator classes
    for schema in sorted(schemas, key=lambda s: s.name):
        if schema in inherited_schemas:
            continue
        built_schemas.add(schema)
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
        allow_extra = schema.name in allow_extra_constructor_arguments
        print(
            class_.render(
                schema=schema,
                attributes=get_attributes(
                    schema,
                    attr_over,
                    subgraphs_solutions.get(schema.name, {}),
                    allow_extra,
                ),
                type_inference=inf,
                value_propagation=prop,
                attr_type_overrides=attr_over,
                subgraph_solutions=subgraphs_solutions.get(schema.name, {}),
            ),
            file=file,
            end="\n\n",
        )

    # Operator constructors
    for schema in sorted(schemas, key=lambda s: s.name):
        if schema not in built_schemas or schema in inherited_schemas:
            continue
        allow_extra = schema.name in allow_extra_constructor_arguments
        # Output variadic solution
        var_sol = out_variadic_solutions.get(schema.name)
        if is_variadic(schema.outputs[-1]) and var_sol is None and not allow_extra:
            raise RuntimeError(
                f"Operator {schema.name} has a variadic output but no solution was provided. "
                f"An argument {schema.outputs[-1].name}_count would be generated if it was in the allow list."
            )
        # Override for attribute type
        attr_over = {
            name: target
            for key, name, target in attr_type_overrides
            if key is None or key == schema.name
        }
        print(
            constructor.render(
                schema=schema,
                attributes=get_attributes(
                    schema,
                    attr_over,
                    subgraphs_solutions.get(schema.name, {}),
                    allow_extra,
                ),
                gen_docstrings=gen_docstrings,
                out_variadic_solution=var_sol,
                subgraph_solutions=subgraphs_solutions.get(schema.name, {}),
                attr_type_overrides=attr_over,
                allow_extra=allow_extra,
            ),
            file=file,
            end="\n\n\n",
        )

    # Extras
    for key in extras:
        extra = env.get_template(f"extras/{key}.jinja2")
        print(extra.render(), file=file, end="\n\n\n")

    for name, alias in CONSTRUCTOR_ALIASES.get(domain, ()):
        print(f"{alias} = {name}", file=file, end="\n")

    print(
        env.get_template("summary.jinja2").render(
            built_names=sorted({schema.name for schema in schemas})
        ),
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
    subgraphs_solutions: Optional[Dict[str, Dict[str, str]]] = None,
    attr_type_overrides: Optional[
        List[Tuple[Optional[str], str, Tuple[str, str]]]
    ] = None,
    allow_extra_constructor_arguments: Iterable[str] = (),
    inherited_schemas: Optional[Dict[onnx.defs.OpSchema, str]] = None,
    extras: Sequence[str] = (),
    target: str = "src/spox/opset/",
    pre_commit_hooks: bool = True,
    gen_docstrings: bool = True,
) -> Tuple[List[onnx.defs.OpSchema], str]:
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
    subgraphs_solutions
        Dictionary from operator names, into attribute names, into strings representing input types for the subgraph
        of that name. The string is a Python expression evaluating int an Iterable of Types.
    attr_type_overrides
        List of replacements for constructor-level types.
        For example, in Cast the ``to`` attribute accepts ``numpy.generic`` instead of ``int``.
        First element is the name of an operator or None, if an override applies to all operators with an attribute.
        Second element is the attribute name to override the type for.
        Last element is a tuple where the first element is the user
        facing type hint used in the constructor function and the
        second element is the Attr type used internally.
    allow_extra_constructor_arguments
        List of operators for which creating an extra constructor argument should not raise.
        This happens in the case of missing solutions for output variadic count or subgraph input types.
    inherited_schemas
        Dictionary of schemas into source modules that may be inherited.
        This means there exists an implementation of its class and constructor in the module.
    extras
        List of template names under ``jinja_templates/extras/`` to add at the end of the code.
        This includes convenience functions that may use the rest of the operator set.
    target
        Base directory to save the generated operator set file (not
        counting subdirectories from ``domain``). An error is raised if
        it does not exist.
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
    if subgraphs_solutions is None:
        subgraphs_solutions = {}
    if attr_type_overrides is None:
        attr_type_overrides = []
    if allow_extra_constructor_arguments is None:
        allow_extra_constructor_arguments = ()
    if inherited_schemas is None:
        inherited_schemas = {}
    allow_extra_constructor_arguments = set(allow_extra_constructor_arguments)

    onnx_domain = domain if domain != DEFAULT_DOMAIN else ""
    if version is None:
        version = max(DOMAIN_VERSIONS[onnx_domain])
    schemas = [
        schema
        for schema in SCHEMAS[onnx_domain][version].values()
        if not schema.deprecated
    ]

    domain_path = "/".join(domain.split("."))

    if not Path(target).exists():
        raise ValueError("Target folder does not exist.")

    path = Path(target) / Path(domain_path) / Path(f"v{version}.py")
    path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Writing {domain}@{version} to `{path}`...")

    with path.open("w") as file:
        write_schemas_code(
            file,
            domain,
            schemas,
            type_inference,
            value_propagation,
            out_variadic_solutions,
            subgraphs_solutions,
            attr_type_overrides,
            allow_extra_constructor_arguments,
            inherited_schemas,
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

    return schemas, f"spox.opset.{domain}.v{version}"


if __name__ == "__main__":
    gen_all_docstrings = True
    ai_onnx_v17_schemas, ai_onnx_v17_module = main(
        "ai.onnx",
        17,
        extras=["const"],
        type_inference={"Compress": "compress11", "Loop": "loop16-fix"},
        value_propagation={"Constant": "constant13"},
        out_variadic_solutions=V16_OUT_VARIADIC_SOLUTIONS,
        subgraphs_solutions=V16_SUBGRAPH_SOLUTIONS,
        attr_type_overrides=DEFAULT_ATTR_TYPE_OVERRIDES,
        allow_extra_constructor_arguments=["Split"],
        gen_docstrings=gen_all_docstrings,
    )
    ai_onnx_v18_schemas, ai_onnx_v18_module = main(
        "ai.onnx",
        18,
        extras=["const"],
        type_inference={"Compress": "compress11"},
        value_propagation={"Constant": "constant13"},
        out_variadic_solutions=V18_OUT_VARIADIC_SOLUTIONS,
        subgraphs_solutions=V16_SUBGRAPH_SOLUTIONS,
        attr_type_overrides=DEFAULT_ATTR_TYPE_OVERRIDES,
        gen_docstrings=gen_all_docstrings,
        inherited_schemas={s: ai_onnx_v17_module for s in ai_onnx_v17_schemas},
    )
    ai_onnx_v19_schemas, ai_onnx_v19_module = main(
        "ai.onnx",
        19,
        extras=["const"],
        type_inference={"Compress": "compress11"},
        value_propagation={"Constant": "constant13"},
        out_variadic_solutions=V18_OUT_VARIADIC_SOLUTIONS,
        subgraphs_solutions=V16_SUBGRAPH_SOLUTIONS,
        attr_type_overrides=DEFAULT_ATTR_TYPE_OVERRIDES,
        gen_docstrings=gen_all_docstrings,
        inherited_schemas={s: ai_onnx_v18_module for s in ai_onnx_v18_schemas},
    )
    ai_onnx_v20_schemas, ai_onnx_v20_module = main(
        "ai.onnx",
        20,
        extras=["const"],
        value_propagation={"Constant": "constant13"},
        out_variadic_solutions=V18_OUT_VARIADIC_SOLUTIONS,
        subgraphs_solutions=V16_SUBGRAPH_SOLUTIONS,
        attr_type_overrides=DEFAULT_ATTR_TYPE_OVERRIDES,
        gen_docstrings=gen_all_docstrings,
        inherited_schemas={s: ai_onnx_v19_module for s in ai_onnx_v19_schemas},
    )
    ai_onnx_v21_schemas, ai_onnx_v21_module = main(
        "ai.onnx",
        21,
        extras=["const"],
        value_propagation={"Constant": "constant13"},
        out_variadic_solutions=V18_OUT_VARIADIC_SOLUTIONS,
        subgraphs_solutions=V16_SUBGRAPH_SOLUTIONS,
        attr_type_overrides=DEFAULT_ATTR_TYPE_OVERRIDES,
        gen_docstrings=gen_all_docstrings,
        inherited_schemas={s: ai_onnx_v20_module for s in ai_onnx_v20_schemas},
    )
    ai_onnx_ml_v3_schemas, ai_onnx_ml_v3_module = main(
        "ai.onnx.ml",
        3,
        attr_type_overrides=[(None, "dtype", ("npt.DTypeLike", "AttrDtype"))],
        type_inference={
            "ArrayFeatureExtractor": "arrayfeatureextractor1",
            "Binarizer": "binarizer1",
            "CategoryMapper": "categorymapper1",
            "Imputer": "imputer1",
            "LinearRegressor": "linearregressor1",
            "Normalizer": "normalizer1",
            "OneHotEncoder": "onehotencoder1",
            "Scaler": "scaler1",
            "TreeEnsembleClassifier": "treeensembleclassifier3",
            "TreeEnsembleRegressor": "treeensembleregressor3",
        },
        gen_docstrings=gen_all_docstrings,
    )
    ai_onnx_ml_v4_schemas, ai_onnx_ml_v4_module = main(
        "ai.onnx.ml",
        4,
        attr_type_overrides=[(None, "dtype", ("npt.DTypeLike", "AttrDtype"))],
        type_inference={
            "Binarizer": "binarizer1",
            "Imputer": "imputer1",
            "LinearRegressor": "linearregressor1",
            "Normalizer": "normalizer1",
            "Scaler": "scaler1",
        },
        gen_docstrings=gen_all_docstrings,
        inherited_schemas={s: ai_onnx_ml_v3_module for s in ai_onnx_ml_v3_schemas},
    )
    ai_onnx_ml_v5_schemas, ai_onnx_ml_v5_module = main(
        "ai.onnx.ml",
        5,
        attr_type_overrides=[(None, "dtype", ("npt.DTypeLike", "AttrDtype"))],
        type_inference={
            "Binarizer": "binarizer1",
            "Imputer": "imputer1",
            "LinearRegressor": "linearregressor1",
            "Normalizer": "normalizer1",
            "Scaler": "scaler1",
        },
        gen_docstrings=gen_all_docstrings,
        inherited_schemas={s: ai_onnx_ml_v4_module for s in ai_onnx_ml_v4_schemas},
    )
