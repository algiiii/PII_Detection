``extraction`` package
======================

Minimal text extraction (block B3): read a born-digital PDF, Word or text file
into the ``NormalizedDocument`` consumed by the detection layer. No OCR and no
layout/table reconstruction — those are later concerns.

Document extractor (``extraction.extractor``)
---------------------------------------------

.. automodule:: pii_detection.extraction.extractor
   :members:
   :member-order: bysource

Reference date (``extraction.dates``)
-------------------------------------

How old a document is, and how much that answer is worth: the date stored inside
the file when there is one, the file system's ``mtime`` otherwise, always paired
with the provenance of the estimate. It is what the retention check of block B7
compares against the retention declared in the ROPA.

.. automodule:: pii_detection.extraction.dates
   :members:
   :member-order: bysource
