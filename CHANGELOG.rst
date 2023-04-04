.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Change log
==========

0.7.0 (2023-04-04)
------------------

This version is intended as a release candidate for ``1.0.0``.

**New features**

- The opset ``ai.onnx@18`` is now shipped with Spox (version 18 of the default domain, as introduced in ONNX 1.13). To avoid code duplication, unchanged implementations are 'inherited' from the previous version.

**Breaking changes**

- The typing rules of the (previously partially documented) extra constructor ``const`` have changed. Its signature is now ``const(npt.ArrayLike, npt.DTypeLike = None) -> Var``. In particular, ``const`` of a Python ``float`` no longer becomes ``float32``, but ``float64`` like numpy - this is a **breaking change**. The operator is redefined to be equivalent to ``constant(numpy.array(value), dtype)``, instead of a complex set of cases like before. 
- The ``Type <= Type`` (``Type.__le__``) overload is now removed.
- Deprecated operator constructors are now no longer generated after the version their schema was deprecated. Effectively, this means ``ai.onnx@17::Scatter`` and ``ai.onnx@17::Upsample`` (available as ``op.scatter`` and ``op.upsample``) are no longer available in ``spox.opset.ai.onnx.v17``. They likely were not used in practice as attempting to build deprecated operators has always failed.

**Bug fixes**

- The operator constructor ``cum_sum`` now also has an alias ``cumsum``, to mirror ``numpy``. This alias should be preferred.

**New unstable features**

- ``spox._future.initializer(ArrayLike, DTypeLike = None) -> Var`` is a new function for creating variables from a constant value. The created value is constructed with ``numpy.array`` and follows the same rules. This function is opset-independent and is implemented using initializers.

0.6.2 (2023-03-29)
------------------

**Pending breaking changes**

- The previously available ``Type <= Type`` (``Type.__le__``) overload is deprecated and will be removed in Spox ``0.7.0``, as it was unintentionally public.
- Constructors for deprecated ONNX operators (currently ``Scatter`` and ``Upsample``) now raise a warning when they are called. They will be removed entirely in ``0.7.0``.

**Bug fixes**

- ``spox.inline`` now correctly renames unused model inputs when building. This could previously cause invalid models to be built.
- Array attributes are now copied when they are passed to an operator. This avoids accidentally mutating them after the operator is constructed.
- The ``Loop`` operator now has patched type inference, so that the loop-carries in its results preserve shapes if the subgraph had them inferred.

0.6.1 (2023-03-07)
------------------

**Pending breaking changes**

- An undocumented extra operator constructor (``const``) now raises a ``DeprecationWarning`` on ``float``, as its behaviour will change in Spox ``0.7.0`` to follow that of ``numpy``.


0.6.0 (2023-02-27)
------------------

**New features**

- ``spox.inline`` was added to the public interface, allowing embedding existing ONNX models in Spox.

**Other changes**

- Models now have a minimum opset version of ``14`` for the ``ai.onnx`` domain to avoid issues with low-versioned models in ORT and other tooling.

**Breaking changes**

- The operator constructor for ``MatMul`` - ``mat_mul`` - has been renamed to ``matmul`` to follow numpy naming.

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
