"""Process-stable seeding (CLAUDE.md §6.1).

Python's builtin `hash()` of strings/tuples is randomized per process (PYTHONHASHSEED),
so `hash((env_id, run_id, version)) & 0xFFFFFFFF` produced a DIFFERENT per-env seed on
every interpreter launch — measurements were not reproducible across runs. This helper
uses blake2b so the same (env_id, run_id, version) always maps to the same 32-bit seed,
regardless of process. Phase 3 v2 (Task B) fix.
"""

from __future__ import annotations

import hashlib


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from the given parts (order-sensitive)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")
