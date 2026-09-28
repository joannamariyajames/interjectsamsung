"""Regression test for the conftest.py credential-isolation fixture.

Root cause this guards against: a one-time strip of GEMINI_API_KEY at
collection time is not enough, because a later import partway through the
suite - notably ``lk_agent_tool.py`` (an official Full-Duplex-Bench file
this project does not modify) - has its own unconditional ``load_dotenv()``
call and can silently repopulate GEMINI_API_KEY/GOOGLE_API_KEY with real
credentials the moment it is imported. Once that happens, every later test
that builds a real AgentRuntime/GeminiProvider unexpectedly gets a real
provider instead of MockProvider and attempts a real network call - which is
exactly what produced an apparent "hang" in tests/test_phase2c.py::
test_turn_cancellation_preserves_safety_and_checkpoints when it ran after
tests/fdb/test_adapter.py::test_resolve_realtime_model_gemini3_8.

These tests don't depend on that specific import chain (which requires the
sibling Full-Duplex-Bench checkout to be on sys.path) - they simulate the
same failure mode directly: something sets the real-looking env vars via a
raw os.environ mutation (not through monkeypatch, exactly like
resolve_realtime_model()'s own ``os.environ["GOOGLE_API_KEY"] = ...`` line,
and exactly like a stray load_dotenv() call would), without any cleanup of
its own, and proves the *next* test never observes it.
"""

from __future__ import annotations

import os

from app.config import settings


def test_1_simulates_a_raw_credential_leak_with_no_cleanup():
    """Mimics what an unguarded load_dotenv() (or resolve_realtime_model()'s
    own os.environ[...] = ... line) does: set real-looking credentials via a
    raw os.environ mutation, with no monkeypatch and no manual cleanup. If
    the isolation fixture did not exist, this state would leak into every
    later test in the process, forever.
    """
    os.environ["GEMINI_API_KEY"] = "simulated-leaked-real-key"
    os.environ["GOOGLE_API_KEY"] = "simulated-leaked-real-key"
    assert os.environ.get("GEMINI_API_KEY") == "simulated-leaked-real-key"


def test_2_next_test_never_observes_the_leaked_credentials():
    """The regression check: without the fix, this test - which does
    nothing to protect itself - would see the previous test's leaked
    GEMINI_API_KEY/GOOGLE_API_KEY and settings.use_gemini would be True,
    exactly reproducing the conditions that produced the apparent hang.
    """
    assert os.environ.get("GEMINI_API_KEY") is None
    assert os.environ.get("GOOGLE_API_KEY") is None
    assert settings.use_gemini is False


def test_3_isolation_holds_even_across_multiple_leaking_tests():
    """A second, independent simulated leak - proves this isn't a one-shot
    fixture that only protects the single test immediately after test_1."""
    os.environ["GEMINI_API_KEY"] = "another-simulated-leak"
    assert settings.use_gemini is True  # leak visible within this test, as expected


def test_4_still_clean_after_the_second_leak():
    assert os.environ.get("GEMINI_API_KEY") is None
    assert os.environ.get("GOOGLE_API_KEY") is None
    assert settings.use_gemini is False
