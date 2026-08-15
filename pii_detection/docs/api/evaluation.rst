``evaluation`` package
======================

Tooling to measure the detectors against a small, hand-annotated corpus of
synthetic documents (block B4, Step 10).

Annotated corpus (``evaluation.corpus``)
----------------------------------------

.. automodule:: pii_detection.evaluation.corpus
   :members:
   :member-order: bysource

Corpus generator (``evaluation.corpus_generator``)
--------------------------------------------------

.. automodule:: pii_detection.evaluation.corpus_generator
   :members:
   :member-order: bysource

Corpus renderer (``evaluation.render``)
---------------------------------------

.. automodule:: pii_detection.evaluation.render
   :members:
   :member-order: bysource

Scoring (``evaluation.scoring``)
--------------------------------

.. automodule:: pii_detection.evaluation.scoring
   :members:
   :member-order: bysource

Baseline runner (``evaluation.run_baseline``)
---------------------------------------------

.. automodule:: pii_detection.evaluation.run_baseline
   :members:
   :member-order: bysource

Presidio baseline runner (``evaluation.run_presidio_baseline``)
---------------------------------------------------------------------

.. automodule:: pii_detection.evaluation.run_presidio_baseline
   :members:
   :member-order: bysource

End-to-end pipeline runner (``evaluation.run_pipeline``)
-------------------------------------------------------------

.. automodule:: pii_detection.evaluation.run_pipeline
   :members:
   :member-order: bysource

Enterprise corpus generator
---------------------------

A whole synthetic file share — folder tree, hundreds of documents of realistic
length, and the mess a real share carries — generated from a seed, to put the
recursive scan, the folder rules, the registry and the dashboard under load.
Unlike the flat corpus above, its ground truth is keyed by the *path* the scan
computes, so gold, manifest and registry line up without translation.

.. automodule:: pii_detection.evaluation.enterprise
   :members:
   :member-order: bysource

Plan value objects (``enterprise.types``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pii_detection.evaluation.enterprise.types
   :members:
   :member-order: bysource

Document archetypes (``enterprise.content``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pii_detection.evaluation.enterprise.content
   :members:
   :member-order: bysource

Folder profiles (``enterprise.profiles``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pii_detection.evaluation.enterprise.profiles
   :members:
   :member-order: bysource

Planted mess (``enterprise.noise``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pii_detection.evaluation.enterprise.noise
   :members:
   :member-order: bysource

Planning and writing (``enterprise.builder``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pii_detection.evaluation.enterprise.builder
   :members:
   :member-order: bysource

Registry check (``enterprise.verify``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pii_detection.evaluation.enterprise.verify
   :members:
   :member-order: bysource
