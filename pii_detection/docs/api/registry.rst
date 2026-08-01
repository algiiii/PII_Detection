``registry`` package
=====================

Detected-PII registry (block B5): persist the PII found in documents as a
delta-based changelog, storing only references and never the values
(minimization). Step 1 (population) is implemented.

Data model (``registry.types``)
-------------------------------

.. automodule:: pii_detection.registry.types
   :members:
   :member-order: bysource

Repository (``registry.repository``)
------------------------------------

.. automodule:: pii_detection.registry.repository
   :members:
   :member-order: bysource

Ingest pipeline (``registry.ingest``)
-------------------------------------

.. automodule:: pii_detection.registry.ingest
   :members:
   :member-order: bysource
