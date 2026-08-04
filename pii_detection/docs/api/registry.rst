``registry`` package
=====================

Detected-PII registry (block B5): persist the PII found in documents as a
delta-based changelog, storing only references and never the values
(minimization). Population and the re-scan delta are both implemented.

Data model (``registry.types``)
-------------------------------

.. automodule:: pii_detection.registry.types
   :members:
   :member-order: bysource

Delta (``registry.diff``)
-------------------------

.. automodule:: pii_detection.registry.diff
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

Folder scan (``registry.scan_folder``)
--------------------------------------

.. automodule:: pii_detection.registry.scan_folder
   :members:
   :member-order: bysource
