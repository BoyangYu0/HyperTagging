HyperTagging documentation
==========================

HyperTagging reconstructs particle decay trees from detector final-state
particles using a shared hyperbolic encoder and a level-autoregressive decoder.
This wiki documents the current implementation, its historical compatibility
paths, and the evidence available in this checkout.

Start with :doc:`dataset`, follow :doc:`training` and :doc:`reconstruction`,
then use :doc:`evaluation` to interpret results. The
:doc:`_generated/status/index` leads with recorded model performance and
identifies measurements that remain unavailable.

.. toctree::
   :maxdepth: 2
   :caption: Physics and implementation

   dataset
   training
   reconstruction
   evaluation
   phase45
   phase44
   phase43
   phase42
   phase41
   _generated/status/index
   setup
   architecture
   workflows
   basf2
   _generated/catalog/index

.. toctree::
   :maxdepth: 1
   :caption: Reference and evidence

   _generated/api/index
   _generated/repository/index
   maintaining

Search works with local assets, including when the HTML directory is served
under a GitHub Pages repository subpath. No live job polling, remote badges,
fonts, diagrams, or JavaScript services are needed.

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
