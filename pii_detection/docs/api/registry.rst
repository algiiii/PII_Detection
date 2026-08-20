``registry`` package
=====================

Detected-PII registry (block B5): persist the current state of the PII found in
documents, storing only references and never the values (minimization). Each scan
fully replaces a document's recorded instances with what it finds now.

Data model (``registry.types``)
----------------------------------------

.. automodule:: pii_detection.registry.types
   :members:
   :member-order: bysource

Folder rules (``registry.folder_rules``)
----------------------------------------

.. automodule:: pii_detection.registry.folder_rules
   :members:
   :member-order: bysource

File stamp (``registry.freshness``)
----------------------------------------

Whether a file changed since the registry last analysed it — the technical
counterpart of the semantic reference date, kept deliberately cheap (modification
time and size, one ``stat``).

.. automodule:: pii_detection.registry.freshness
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
