"""Runtime-owned facts for the neutral observation collector, without its clock."""
from dataclasses import asdict, dataclass


class ReadbackUnavailable(RuntimeError):
    """Bootstrap did not supply valid observed backend facts."""


@dataclass(frozen=True)
class BackendReadback:
    """Trusted backend bootstrap facts, never client or deployment desired state.

    The provider must reject readback if its loaded model/device/precision differs.
    Bounds are ceilings; token/head budgets can reject smaller complex requests.
    They are not guarantees that every request below each bound is executable.
    """
    backend: str
    precision: str
    question_types: tuple[str, ...]
    max_question_rows: int
    max_candidates_per_question: int
    max_sequence_tokens: int
    max_head_tokens: int
    max_encoded_tokens: int

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() or len(v.encode()) > 256
               or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in v)
               for v in (self.backend, self.precision)):
            raise ValueError("invalid backend readback identity")
        if (type(self.question_types) is not tuple or not self.question_types
                or len(set(self.question_types)) != len(self.question_types)
                or any(q not in ("choice", "score", "noul") for q in self.question_types)):
            raise ValueError("invalid backend question types")
        for name, value in asdict(self).items():
            if name.startswith("max_") and (type(value) is not int or not 0 < value <= 2**32 - 1):
                raise ValueError("invalid backend limits")
        if (self.max_head_tokens > self.max_sequence_tokens
                or self.max_encoded_tokens > self.max_question_rows * self.max_sequence_tokens):
            raise ValueError("inconsistent backend limits")
