Maintaining and publishing the wiki
===================================

Publication boundary
--------------------

Authored guides live in ``docs/wiki``. The standalone root is
``docs/index.rst``; Belle II discovery begins at
``doc/index-hypertagging.rst``. One connected toctree reaches all pages.

At Sphinx initialization, ``docs/_ext/wiki.py`` generates the API,
scripts/examples catalog, status dashboard and compact repository inventory.
The public artifact contains static AST signatures, redacted literal docstrings,
relative definition line provenance, hashes and explicitly allowlisted status
fields. It does not distribute repository source bodies or raw audit, config,
report, script or package downloads. Sphinx source copying and viewcode source
discovery are disabled.

Every lexical function and class occurrence in ``src/hypertagging`` is covered,
including underscore-prefixed internals, undocumented definitions, overloads,
property accessors, nested functions and classes declared inside functions.
Repeated definitions retain distinct line provenance. Public/stable and
internal/unstable surfaces have separate sections; these describe API visibility,
not a promise of scientific validation or universal backward compatibility.
Function-local definitions are inert Python-domain entries with unique source
anchors rather than importable object targets. Reexports and aliases are
supplementary. Inherited or decorator-generated runtime methods cannot be
inferred from syntax and are not invented.

Docstrings are rendered as inert literal text. Their embedded directives,
defaults and annotations are never executed. Private operational values and
string defaults are withheld. The catalog uses curated relative commands and
placeholder inputs. Its Python entries cover every lexical function and class
occurrence with dotted ownership and mark all function-local definitions
internal/unstable. It never executes a script to discover CLI arguments.

The whole Git tree is inventoried using opaque hashes of relative filenames,
fixed groups, domain-separated content fingerprints and counts. It includes tests, reports and frozen
snapshots without creating source-mirror pages or exposing their filenames.
Binary/model/data artifacts are counted without reading their contents.
Symlinks and ignored runtime files are excluded.

Generation ownership protects existing files: manually edited generated content
blocks replacement, and only unchanged owned stale files may be removed.
The wrapper requires a new or empty output directory. Prefer it to direct
Sphinx invocation when keeping generated files outside the working source tree.

Freshness and scientific evidence
----------------------------------

The dashboard reads a fixed set of tracked records. Each projected claim keeps
an opaque source identifier and content SHA-256, recorded observation date and
scope where available. It publishes no host, user, allocation, scheduler,
authority or checkpoint artifact addresses. Its clock is ``SOURCE_DATE_EPOCH``
when supplied, otherwise the HEAD commit time; missing clocks are explicit.
Wall time and file modification times cannot manufacture a fresh observation.

Audit and verification revision mismatches are warnings even when the records
were recently edited. Age warnings and unavailable metadata remain visible.
``docs/audits/current_status.md`` owns mutable audit claims in the checkout.
The dashboard preserves its NO_GO decision and separates recorded CPU-fixture
evidence from unrun real-data pilots and trained physics evaluation. Historical
pytest counts and notebook registrations do not establish current execution.
Pretraining progress is a recorded counter ratio; Stage A completion does not
override its failed promotion threshold. Documentation PASS never changes
training authorization.

Post-training dashboard updates
-------------------------------

The dashboard is a derived publication, not a live training monitor. Training
and evaluation jobs do not edit the repository, open a review request, or
publish Pages. The automatic build begins only after reduced, reviewed evidence
has been committed to a branch. Never edit or commit ``docs/wiki/_generated``
or ``docs/_build``.

Use this handoff after a training or evaluation run:

#. Verify that the run reached a valid terminal state with its campaign-specific
   receipt verifier. A scheduler exit alone is not sufficient evidence.
#. Add only a small tracked metadata record containing the allowlisted values
   needed by the dashboard. Checkpoints, raw logs, data, absolute paths, host or
   user names, scheduler identifiers, and authority locations remain outside
   the publication boundary.
#. Update the corresponding entry in ``SOURCE_PATHS`` in
   ``docs/_ext/wiki_status.py``. Dated records are immutable: add a successor
   record and repoint the allowlist rather than overwriting the prior receipt.
   Update the status-generator fixtures and expected values in the same change.
#. Append a verification record or change ``docs/audits/current_status.md`` and
   its issue ledger only when those reviewed claims actually changed. Notebook
   registry status changes only when the registered notebook run and any
   required human review occurred.
#. Build from a clean, dedicated publication checkout and inspect the generated
   status values, source hashes, revision/freshness warnings, and scientific
   evidence labels before opening review.

Run the complete documentation gate locally:

.. code-block:: bash

   python -m pytest -q tests/test_docs_*_cpu.py
   docs_check="$(mktemp -d)"
   python scripts/build_docs.py --output "$docs_check"
   docs_html="$(find "$docs_check" -maxdepth 1 -type d -name html -print -quit)"
   docs_generated="$(find "$docs_check" -type d -name _generated -print -quit)"
   python scripts/validate_docs.py --html "$docs_html" --generated "$docs_generated" --check-generation --workflow
   git diff --check

Commit the reduced evidence and any required allowlist/test changes on a branch,
then open a pull request or merge request (PR/MR) targeting ``master``. The
review must establish that the source receipt is valid, the projection contains
no operational secrets, and dashboard claims do not exceed the evidence.

GitHub pull requests are validated automatically by
``.github/workflows/docs.yml``. A successful trusted merge to ``master`` builds
the dashboard again and deploys the validated Pages artifact. The deployment
environment exposes the canonical page URL; the dashboard is under
``wiki/_generated/status/``.

A GitLab mirror can use the same merge-request review procedure, but this
repository does not include a GitLab runner or Pages pipeline. The mirror must
configure its MR job to run the commands above, and its merged revision must be
synchronized to the canonical GitHub ``master`` branch for this Pages deployment
to update. Do not describe a GitLab MR as automatically deployed until that
external runner, synchronization, protected branch, and Pages configuration
have been verified.

Validation
----------

Install tools in a new documentation environment. Existing project and GPU
locks do not need to change. Dependencies are pinned separately; install from
a local wheelhouse for an entirely offline environment setup. Generation,
rendering, search and validation require no network after setup.

.. code-block:: bash

   python -m pip install -r docs/requirements-test.txt
   python -m pytest -q tests/test_docs_*_cpu.py
   docs_check="$(mktemp -d)"
   docs_text="$(mktemp -d)"
   docs_basf2="$(mktemp -d)"
   python scripts/build_docs.py --output "$docs_check"
   docs_html="$(find "$docs_check" -maxdepth 1 -type d -name html -print -quit)"
   docs_generated="$(find "$docs_check" -type d -name _generated -print -quit)"
   python scripts/validate_docs.py --html "$docs_html" --generated "$docs_generated" --check-generation --workflow
   python scripts/build_docs.py --builder text --output "$docs_text"
   python scripts/build_docs.py --layout basf2 --output "$docs_basf2"

The wrapper runs warning-fatal Sphinx and independently checks all package and
script/example lexical definition occurrences, module inventories and catalog
entries. HTML validation checks
local links, fragments, Sphinx object inventory and runtime assets. Artifact
privacy validation scans rendered HTML, decoded JSON/search indexes and text
for sensitive field assignments, recognized private patterns, nontrivial
string/numeric structured-source literals, raw downloads and forbidden source
directories. Short/common scalar values cannot be attributed to a private
source after extraction, so the primary confidentiality boundary is the
allowlisted projection: raw configuration/report fields and source files never
enter the staged site. Schema descriptions are treated as descriptions rather
than recorded credentials; schema defaults and constants remain scanned. The
validator runs inside ordinary Sphinx builds too, before a successful build is
reported. A failing privacy check blocks artifact upload.

Readable source files and structured privacy sources above 5,000,000 bytes make
the publication check fail closed; they must be reduced, excluded by the
repository inventory policy, or handled by a deliberately reviewed scanner
change before Pages can publish.

``--check-generation`` compares every generated byte against a fresh
generation. Stable source bytes, Git metadata and the explicit clock produce
stable reference files. Full HTML bytes can vary with Sphinx/dependency versions.
``--workflow`` checks parsed permissions, event gates, exact action pins,
explicit bash/pipefail and mandatory build checks using ordinary exceptions,
so optimized Python cannot disable security validation. Run
``actionlint .github/workflows/docs.yml`` when installed for broader
GitHub Actions expression validation. External linkcheck is a separate online
operation and is not required by offline builds.

GitHub Pages
------------

The workflow uses ``contents: read`` for builds and validates pull requests.
Only trusted pushes to ``master`` or manual runs on ``master``
may upload the validated HTML and enter the separate deployment job.
That job grants only Pages write and ID-token write permissions; it
configures Pages and deploys the artifact without checking out or running
repository code. Checkout credentials are not persisted. Every action is
pinned to its verified immutable commit. Every shell step uses bash with
error checking and pipefail.

A repository administrator must select **Settings > Pages > Build and
deployment > Source: GitHub Actions** and restrict the ``github-pages``
environment to master. These remote settings cannot be established by a local
file edit. After the workflow is merged, a successful trusted run publishes
and exposes the URL in the deployment environment output. Relative site assets
support a project subpath without a custom domain.

The workflow follows `GitHub's custom Pages workflow contract
<https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>`_.
