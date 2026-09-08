basf2 and Sphinx interoperability
=================================

Belle II's build discovers ``index*.rst`` files in each package's ``doc``
directory. Its documentation should form one connected toctree. HyperTagging
therefore supplies ``doc/index-hypertagging.rst``, which includes
``docs/index.rst`` and then ``docs/wiki/index.rst``. This follows
`Belle II's Sphinx authoring guide
<https://software.belle2.org/development/sphinx/framework/doc/atend-doctools.html>`_.
The generated API uses standard Sphinx Python-domain objects and publishes a
local ``objects.inv`` inventory; it needs no Belle II module-discovery extension.
The isolated requirements pin Sphinx 7.3.7, the version listed by the
`Belle II externals package inventory
<https://github.com/belle2/externals#python-packages>`_ when this compatibility
boundary was verified. The same warning-fatal site is exercised in the
standalone and basf2 discovery layouts.

Package integration and standalone builds
-----------------------------------------

The standard isolated build uses a fresh output directory:

.. code-block:: bash

   docs_out="$(mktemp -d)"
   python scripts/build_docs.py --output "$docs_out"

For a larger basf2 documentation tree, stage the package's ``doc`` and ``docs``
directories together, preserving their relative locations. Generate the API,
catalog and status under the staged ``docs/wiki/_generated`` directory using
``wiki.generate(repository_root, generated_directory)`` from ``docs/_ext/wiki.py``.
Then the enclosing tree can discover ``doc/index-hypertagging.rst`` and reach
every generated page through its toctrees. Generation reads the local checkout
and emits only redacted projections. It imports no package modules.

If the enclosing Sphinx build loads the ``wiki`` extension itself, add
``docs/_ext`` to its extension search path and set ``wiki_content_path`` to the
staged package's ``docs/wiki`` directory relative to the Sphinx source root.
The extension generates the references before Sphinx reads the toctrees. Use
the enclosing site's local static asset configuration for ``wiki.css``.
The compatibility fixture validates this discovery/toctree structure; it does
not claim a build inside an installed basf2 release.

Compatible settings and the publication boundary
------------------------------------------------

The shared configuration enables ``autodoc``, ``napoleon``, ``viewcode``,
``mathjax``, ``autosectionlabel`` and ``intersphinx``. Section labels are prefixed
with their document names; ``default_role = "any"`` supports single-backtick
references and ``numfig = True`` enables numbered figures. Napoleon supports
Google and NumPy docstrings for future authored autodoc pages.

For public builds, ``html_copy_source`` and ``html_show_sourcelink`` are false.
The ``wiki`` extension blocks viewcode source discovery: enabling viewcode does
not authorize publishing package bodies or importing modules. Source references
are relative filenames, definition lines and SHA-256 values. No raw repository,
script, configuration, audit, report or package source is distributed.

MathJax syntax remains accepted, while ``html_math_renderer = "offline-text"``
uses escaped formula text and equation numbering without loading any browser
library. ``mathjax_path`` is empty. Equations remain readable in HTML and text
builds with no JavaScript. For example, :math:`p_{mother} = \sum_i p_i` means the
mother momentum is the sum of its daughter momenta. A surrounding basf2 site may
use its own locally installed math renderer; the standalone site has no CDN or
runtime network dependency.

Local cross-site references
---------------------------

Another Sphinx site can reference HyperTagging through a **local** inventory:

.. code-block:: python

   extensions = ["sphinx.ext.intersphinx"]
   intersphinx_mapping = {
       "hypertagging": (
           "SITE_BASE_URL",
           "local-inventories/hypertagging/objects.inv",
       ),
   }

Replace ``SITE_BASE_URL`` with the published site's base URL. The inventory is
local at build time; the URL is the destination used in rendered cross-references.
Copy a built inventory rather than configuring
automatic network fetching in an offline build. This wiki's default inventory
mapping is empty. The native Python domain needs no project imports.

Deployment boundary
-------------------

The current integration consumes reconstructed FSP ParticleLists and publishes
a new root list. Exporting a trusted checkpoint belongs to the training
environment. Event processing consumes a hash-checked ONNX bundle whose
manifest fixes tensor shapes, feature/PID contracts, normalizers, policy and
compatible releases. It does not load training checkpoints during events.

.. code-block:: python

   from hypertagging.basf2_integration import add_hypertagging_full_decay

   add_hypertagging_full_decay(
       path,
       manifest_path="local-bundle/manifest.json",
       input_particle_lists=("pi+:HyperTaggingFSP", "gamma:HyperTaggingFSP"),
       output_particle_list="Upsilon(4S):HyperTagging",
   )

This is a steering fragment: ``path`` and input lists must already exist, and
the selection must match training. The current deployment document records
``light-2607-kasei`` for its installed ONNX runtime; the older preprocessing
pilot records ``release-08-03-00``. Preserve those separate evidence scopes.
Consult ``docs/basf2_onnx_full_decay.md`` in the checkout for export,
runner and smoke commands, required capacities, and integration limitations.

Mocking optional imports
------------------------

Belle II documents that Python autodoc requires modules to import cleanly and
docstrings to contain valid reStructuredText; see its `Python documentation
guide <https://software.belle2.org/development/sphinx/framework/doc/doc_python.html>`_.
Ordinary API generation here parses syntax without executing modules or evaluating
annotations, defaults or decorators. ``autodoc_mock_imports`` in ``conf.py``
also lists basf2, ROOT, PyTorch and heavy optional dependencies for future
authored autodoc sections. Static generation is the default because import
mocks alone cannot reliably reproduce dynamic library inheritance or runtime
side effects. The generated reference exposes every lexical function and class
occurrence, including undocumented internals, nested functions and repeated
definitions at distinct source lines. Function-local entries retain unique
anchors but do not claim importable Python targets. Public and internal surfaces
are separated. Redacted docstrings are displayed as literal text so mixed
Markdown/reStructuredText and embedded directives cannot execute or read
external files during the build.
