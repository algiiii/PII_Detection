Adding a custom pattern
=======================

The detection layer is **config-driven**: you can teach the system to detect a new
structured identifier — a company code, an internal reference, a sector-specific
number — **without writing any code**. Custom patterns live in
``pii_detection/config/custom_patterns.yaml`` and are executed live by the
``RegexDetector``, combined with Presidio's built-in recognizers through the
``CompositeDetector``. Adding one is a YAML edit.

Steps
-----

1. **Declare the category** (only if it is new). Add an entry to
   ``pii_detection/config/categories.yaml``:

   .. code-block:: yaml

      - id: employee_id
        label: "Employee ID"
        frameworks: [gdpr]

   Every ``pii_type`` referenced by a pattern must exist in this catalog; the
   loader rejects an unknown one.

2. **Add the pattern.** Append a rule to
   ``pii_detection/config/custom_patterns.yaml``:

   .. code-block:: yaml

      - rule_id: employee_id
        pii_type: employee_id
        pattern: '\bEMP-\d{5}\b'
        base_confidence: 0.8

3. **Restart** the service (or re-run the CLI). The pattern is now detected by
   every entry point — ``scan``, ``registry.ingest``, ``registry.scan_folder`` and
   the web app — because they all build their detectors from this config.

Rule fields
-----------

The schema is ``RegexRuleModel`` (in ``pii_detection/detection/config.py``):

``rule_id``
   Unique id of the rule within the file.
``pii_type``
   The category the rule produces; **must** exist in ``categories.yaml``.
``pattern``
   The regex source. It must compile — this is checked when the config loads, so a
   broken pattern fails fast rather than silently.
``base_confidence`` (optional)
   Confidence assigned to a match, in ``[0, 1]``. Default ``0.6``.
``flags`` (optional)
   List of Python ``re`` flag names to compile the pattern with. Default
   ``["UNICODE"]``. Allowed: ``IGNORECASE``, ``MULTILINE``, ``DOTALL``,
   ``UNICODE``, ``VERBOSE``, ``ASCII``.

How it works
------------

At startup ``build_default_detectors`` loads ``custom_patterns.yaml``, builds a
``RegexDetector`` from its rules, and wraps it together with Presidio's pattern
recognizers in a ``CompositeDetector``. The regex candidates then flow through the
same merge and persistence as everything else. Because Presidio already covers the
common identifiers (email, IBAN, credit card, IP, Italian fiscal code), this file
is meant for what Presidio does **not** cover — the Swiss AVS number ships as the
first example in it.

Limitations
-----------

- **No checksum validation.** A pattern matches by *shape*; validating a
  checksum (IBAN mod-97, Luhn, …) to cut false positives is a planned future
  development.
- **Patterns only.** This mechanism is for fixed-shape identifiers detected by
  regex. Contextual data detected by the NER model (names, addresses, free-text
  health data) is configured separately and is not covered here.
