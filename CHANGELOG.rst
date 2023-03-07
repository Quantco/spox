.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Change log
==========


0.6.1 (2023-03-07)
------------------

**Deprecations**

- An undocumented extra operator constructor (``const``) now raises a ``DeprecationWarning`` on ``float``, as its behaviour will change to follow that of ``numpy``.


0.6.0 (2023-02-27)
------------------

**New features**

- ``spox.inline`` was added to the public interface, allowing embedding existing ONNX models in Spox.

**Other changes**

- Models now have a minimum opset version of ``14`` for the ``ai.onnx`` domain to avoid issues with low-versioned models in ORT and other tooling.


0.5.0 (2023-01-20)
------------------

**New features**

- The ``spox.build`` and ``spox.argument`` functions were added enabling the building of graphs through a stable interface.

**Notable changes**

- The documentation formatting inside the (auto-generated) ``spox.opset`` module was greatly improved.


0.4.0 (2023-01-16)
------------------

**Breaking changes**

- Removed the ``Var.dtype`` and ``Var.shape`` properties in favor of the more explicit ``Var.unwrap_tensor().dtype`` and ``Var.unwrap_tensor().shape`` ones.

**Bug fixes**

- Non-ASCII characters in constant tensors are now handled correctly.
- The ``Compress`` operator has gained an explicit type and shape inference implementation


0.3.0 (2022-12-20)
------------------

**Notable changes**

- Renamed the library to Spox
- Reduced the public API surface to a bare minimum such that downstream packages may offer a usable and stable user experience with spox-based converters. This release is intended as a release candidate. Breaking changes may still occur if necessary.
