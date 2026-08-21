"""Live end-to-end voice-deadline harness.

Nothing in this package is collected by the default ``pytest`` run: it places real
(non-PSTN) Vapi calls and costs money. See ``tests/e2e/README.md``.

The one exception is :mod:`tests.e2e.deadlines`, which is pure arithmetic over a
recorded Vapi call object and is covered by ``tests/test_e2e_deadlines.py`` in the
default suite -- because a timing harness whose own maths is untested is worse than
no harness at all.
"""
