``compliance`` package
======================

Compliance block (B6 + B7): the bridge that connects the declared side (ROPA,
block B1) with the detected side (registry, block B5) into the first end-to-end
compliance verdict. A document is associated with the processing activities it
belongs to (B6, explicit and DPO-driven), then its detected PII is compared with
what those activities declare (B7): orphan PII (detected but not declared),
declared categories never found, and an approximate retention check. Like the
registry it reads, it handles only references, never PII values (minimization).

Verdict value objects (``compliance.types``)
--------------------------------------------

.. automodule:: pii_detection.compliance.types
   :members:
   :member-order: bysource

Association (``compliance.assign``)
-----------------------------------

.. automodule:: pii_detection.compliance.assign
   :members:
   :member-order: bysource

Check (``compliance.checker``)
------------------------------

.. automodule:: pii_detection.compliance.checker
   :members:
   :member-order: bysource

Retention overview (``compliance.overview``)
--------------------------------------------

The same verdict, asked of the whole registry instead of one document: everything
kept past its declared term, worst first. Read-only — looking at the corpus never
writes to it.

.. automodule:: pii_detection.compliance.overview
   :members:
   :member-order: bysource
