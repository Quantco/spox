# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import datetime
import importlib
import inspect
import os
import subprocess
import sys
from subprocess import CalledProcessError
from typing import cast

_mod = importlib.import_module("spox")


project = "spox"
copyright = f"{datetime.date.today().year}, QuantCo, Inc"
author = "QuantCo, Inc."

extensions = [
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "nbsphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.linkcode",
    "sphinxcontrib.apidoc",
]

apidoc_module_dir = "../src/spox"
apidoc_output_dir = "api"
apidoc_separate_modules = True
apidoc_extra_args = ["--implicit-namespaces"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# Copied and adapted from
# https://github.com/pandas-dev/pandas/blob/4a14d064187367cacab3ff4652a12a0e45d0711b/doc/source/conf.py#L613-L659
# Required configuration function to use sphinx.ext.linkcode
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a given Python object."""
    if domain != "py":
        return None

    module_name = info["module"]
    full_name = info["fullname"]

    _submodule = sys.modules.get(module_name)
    if _submodule is None:
        return None

    _object = _submodule
    for _part in full_name.split("."):
        try:
            _object = getattr(_object, _part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(_object))  # type: ignore
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, line_number = inspect.getsourcelines(_object)
    except OSError:
        line_number = None  # type: ignore

    if line_number:
        linespec = f"#L{line_number}-L{line_number + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(cast(str, _mod.__file__)))

    try:
        # See https://stackoverflow.com/a/21901260
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except CalledProcessError:
        # If subprocess returns non-zero exit status
        commit = "main"

    return (
        "https://github.com/quantco/spox"
        f"/blob/{commit}/src/{_mod.__name__.replace('.', '/')}/{fn}{linespec}"
    )
