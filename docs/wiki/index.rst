HyperTagging documentation
==========================

HyperTagging reconstructs particle decay trees from detector final-state
particles using a shared hyperbolic encoder and a level-autoregressive decoder.
This wiki documents the current implementation, its historical compatibility
paths, and the evidence available in this checkout.

Start with :doc:`setup` for an isolated documentation build or CPU examples,
then :doc:`architecture` for the data and reconstruction contracts.
:doc:`_generated/status/index` distinguishes implementation checks, recorded
training observations, and scientific results that remain unverified.

.. toctree::
   :maxdepth: 2
   :caption: Learn and operate

   setup
   architecture
   workflows
   basf2
   _generated/catalog/index

.. toctree::
   :maxdepth: 1
   :caption: Reference and evidence

   _generated/api/index
   _generated/status/index
   _generated/repository/index
   maintaining

Search works with local assets, including when the HTML directory is served
under a GitHub Pages repository subpath. No live job polling, remote badges,
fonts, diagrams, or JavaScript services are needed.

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
