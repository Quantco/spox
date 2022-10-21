import re
import subprocess
import typing
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import jinja2
import numpy
import onnx

import steelix
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


def get_attr_hint(
    attr: onnx.defs.OpSchema.Attribute,
    attr_type_overrides: Dict[str, type],
    func: bool = False,
) -> str:
    """
    Parameters
    ----------
    attr
        Attribute to get type hint for.
    func
        If true, this is for a constructor, which may get a different hint.
    attr_type_overrides
        Used when an attribute type is overriden, like for Cast where the constructor
        expects a ``type[numpy.generic]`` instead of an enum int as in the definition.

    Returns
    -------
    str
        A representation of the type hint for the attribute.
    """
    unknown_default = not attr.required and not attr.default_value.type
    typ = (
        attr_type_overrides.get(attr.name) if attr_type_overrides is not None else None
    )
    if typ is None:
        try:
            typ = steelix.attr.Attr.attr_type_to_py_type()[attr.type.value]
        except KeyError:
            print(f"Cannot get Python attribute for: {attr.type}, skipping.")
            return "Any" if not func else "None"
    if steelix.attr._is_list_attribute(typ):
        hint = (
            f"{'Iterable' if func else 'Sequence'}[{typing.get_args(typ)[0].__name__}]"
        )
    elif typ is numpy.generic:
        hint = "typing.Type[numpy.generic]" if func else "numpy.generic"
    else:
        hint = typ.__name__
    if func and unknown_default:
        hint = f"Optional[{hint}]"
    return hint


def get_attr_hint_func(
    attr: onnx.defs.OpSchema.Attribute, attr_type_overrides: Dict[str, type]
) -> str:
    """Shorthand for get_attr_hint with func=True, passing in the right constructor attribute type override."""
    return get_attr_hint(attr, attr_type_overrides, True)


def get_attr_default_func(
    attr: onnx.defs.OpSchema.Attribute, attr_type_overrides: Dict[str, type]
) -> str:
    """Get the default value for an attribute in the constructor signature."""
    s_attr = steelix.attr.Attr.from_onnx(attr.default_value)
    value = s_attr.value
    if steelix.attr._is_list_attribute(typing.get_origin(s_attr.value_type)):
        value = tuple(value)
    if attr_type_overrides.get(attr.name) is numpy.generic:
        value = steelix.type_system.Tensor.elem_type_from_onnx(value)
        return f"numpy.{value.__name__}"
    return repr(value)


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
        loader=jinja2.FileSystemLoader("templates"),
        extensions=["jinja2.ext.do"],
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["format_github_markdown"] = format_github_markdown
    env.filters["get_constructor_name"] = get_constructor_name
    env.globals["get_attr_hint"] = get_attr_hint
    env.globals["get_attr_hint_func"] = get_attr_hint_func
    env.globals["get_attr_default_func"] = get_attr_default_func
    env.globals["is_variadic"] = is_variadic
    env.globals["is_optional"] = is_optional
    env.globals["get_constructor_return"] = get_constructor_return
    env.globals["Attr"] = steelix.attr.Attr
    return env


def write_schemas_code(
    file,
    schemas: List[onnx.defs.OpSchema],
    type_inference: Dict[str, str],
    value_propagation: Dict[str, str],
    out_variadic_solutions: Dict[str, str],
    attr_type_overrides: List[Tuple[Optional[str], str, type]],
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
    attr_type_overrides: Optional[List[Tuple[Optional[str], str, type]]] = None,
    extras: Sequence[str] = (),
    target: str = "steelix/opset/",
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
        Last element is the type to override with.
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
            (None, "dtype", numpy.generic),
            ("Cast", "to", numpy.generic),
        ],
    )
    main(
        "ai.onnx.ml",
        3,
        attr_type_overrides=[(None, "dtype", numpy.generic)],
        type_inference={
            "OneHotEncoder": "onehotencoder1",
            "ArrayFeatureExtractor": "arrayfeatureextractor1"
        },
    )
