Setup and first run
===================

Documentation environment
-------------------------

Use Python 3.11 for the pinned documentation toolchain. From the Git root,
create a **new** virtual environment (choose another directory if it exists):

.. code-block:: bash

   python3.11 -m venv .venv-docs
   .venv-docs/bin/python -m pip install -r docs/requirements.txt
   docs_out="$(mktemp -d)"
   .venv-docs/bin/python scripts/build_docs.py --output "$docs_out"
   docs_html="$(find "$docs_out" -maxdepth 1 -type d -name html -print -quit)"
   .venv-docs/bin/python -m http.server 8000 --directory "$docs_html"

Open the server's local address. Dependencies are needed only when preparing the
environment. With a wheelhouse, replace the installation step with
``pip install --no-index --find-links wheelhouse -r docs/requirements.txt``.
Generation and rendering then run entirely offline. The docs environment needs
neither the project package nor basf2, ROOT, CUDA, data, or checkpoints.

The ordinary Sphinx interface is also supported:

.. code-block:: bash

   python -m sphinx -W --keep-going -b html docs docs/_build/direct

The extension regenerates API, catalog, status, and inventory pages at build
initialization. The build wrapper stages them in its output directory; direct
Sphinx builds put generated sources in ignored ``docs/wiki/_generated``.

Project environment and CPU examples
------------------------------------

Project metadata requires Python >=3.10,<3.14. Inspect your installed
environment before using host-specific paths from historical documents.
``.venv`` is the normal project environment; the separately frozen GPU
environment is described in the repository's ``environment/gpu/README.md``.
On a fresh checkout with uv installed, the existing project lock can be used
without relocking:

.. code-block:: bash

   uv sync --frozen --all-extras
   source scripts/activate_env.sh project
   python scripts/check_uv_lock_direct_dependencies.py

This installs the project's pinned PyTorch build and its scientific dependencies;
it is substantially larger than the documentation environment. Running CPU
fixtures does not require a GPU allocation. Do not modify an existing frozen
environment to fix documentation dependencies.

Run these commands from the repository root in the project environment:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python examples/toy_mc_minimal/run_example.py
   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python examples/grafei_minimal/run_example.py
   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python examples/gpt_like_minimal/run_example.py

The :doc:`_generated/catalog/index` gives each runnable example's purpose,
relative provenance, invocation, prerequisites, and CLI argument names. The fixtures check software
contracts; their outputs are not physics-performance measurements.

Source references: ``pyproject.toml``, ``uv.lock``, ``scripts/activate_env.sh``,
``examples/README.md`` and ``tests/test_examples_cpu.py`` remain in the checkout.
The :doc:`_generated/repository/index` publishes only a compact hash inventory.
