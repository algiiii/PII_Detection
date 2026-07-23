``ropa`` package
================

Data model of the ROPA (Record of Processing Activities, block B1). The flat
CNIL sheet is normalized around :class:`~pii_detection.ropa.types.ProcessingActivity`
and resolved onto the shared ``pii_type`` catalog, so the declared data can be
compared with what the detection engine finds (block B7). Design rationale in
``doc/sections/5_architettura.tex`` (§``sec:modello-ropa``).

Data model (``ropa.types``)
---------------------------

.. automodule:: pii_detection.ropa.types
   :members:
   :member-order: bysource

Ingestion (``ropa.ingestion``)
------------------------------

Retention parsing (``ropa.ingestion.retention``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pii_detection.ropa.ingestion.retention
   :members:
