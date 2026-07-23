``llm`` package
===============

Shared access to a local, CPU-friendly LLM runtime (Ollama) for the selective-AI
tasks of the system. A single :class:`~pii_detection.llm.client.LLMClient`, with
model and host read from the environment, is reused by every AI step so the model
choice stays a configuration decision (see ``doc/plans/ai-assessment.md``).

Client (``llm.client``)
-----------------------

.. automodule:: pii_detection.llm.client
   :members:
