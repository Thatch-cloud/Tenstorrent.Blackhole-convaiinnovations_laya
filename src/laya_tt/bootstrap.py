"""Explicit runtime composition; grant authority and device ownership stay external."""
from dataclasses import asdict
from pathlib import Path
import threading

from .admission import AdmissionAdapter
from .asgi import DecisionASGI
from .ledger import ExecutionJournal
from .ledger_client import LedgerUnavailable, LocalLedgerClient
from .readback import BackendReadback, ReadbackUnavailable
from .service import DecisionService, RuntimeIdentity


class RuntimeApplication:
    """One loaded backend, runtime generation, journal and local ledger connection.

    The trusted launcher must establish device ownership BEFORE loading a backend.
    This class neither loads a model nor listens on a network interface. Supply an
    authenticated grant verifier; no development verifier or CPU fallback exists.
    """

    def __init__(self, *, backend, runtime: RuntimeIdentity, verify_grant,
                 policy_revision: str, journal_path: Path, ledger_socket: Path,
                 ledger_uid: int, worker_limits=None, max_admitting: int = 8,
                 max_pending: int = 16):
        if (not isinstance(runtime, RuntimeIdentity) or not callable(verify_grant)
                or any(not callable(getattr(backend, name, None))
                       for name in ("prepare", "execute", "readback"))):
            raise ValueError("verified backend and grant verifier required")
        facts = backend.readback()
        if not isinstance(facts, BackendReadback) or facts.backend != runtime.backend:
            raise ReadbackUnavailable("loaded backend differs from runtime assignment")
        self.journal = ExecutionJournal(journal_path)
        self.journal.check_startup()
        self._ledger = LocalLedgerClient(ledger_socket, ledger_uid, asdict(runtime))
        self._delivery = threading.Lock()
        admission = AdmissionAdapter(verify_grant=verify_grant, prepare=backend.prepare,
            consume_reservation=self._ledger.journaled_consumption(self.journal),
            checkpoint_revision=runtime.checkpoint_revision,
            runtime_generation=runtime.runtime_generation, policy_revision=policy_revision,
            max_question_rows=min(64, facts.max_question_rows),
            max_encoded_tokens=min(32768, facts.max_encoded_tokens))
        self.service = DecisionService(admission=admission, backend=backend.execute,
            runtime=runtime, receipt_sink=self.journal, backend_readback=backend.readback,
            worker_limits=worker_limits, max_admitting=max_admitting)
        try:
            self.service.runtime_readback()
            self.asgi = DecisionASGI(self.service, max_pending=max_pending)
        except BaseException:
            self.service.drain(cancel_queued=True)
            raise

    def start(self, self_test):
        """Stay unready until recovery guard, loaded readback and real self-test pass."""
        if not callable(self_test):
            raise ValueError("loaded model self-test required")

        def checked():
            self.service.runtime_readback()
            if self_test() is not True:
                return False
            self.service.runtime_readback()
            return True

        self.service.start(checked)

    def deliver_pending(self, *, limit: int = 8) -> int:
        """Run one bounded outbox batch; the host owns scheduling and retry policy.

        Stop on the first failure and retain its original evidence. Old-generation
        receipts need their original trusted bridge assignment, never relabeling.
        Concurrent pumps on this instance fail rather than creating duplicate work.
        """
        if type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("receipt batch limit must be between 1 and 64")
        if not self._delivery.acquire(blocking=False):
            raise LedgerUnavailable("receipt delivery already running")
        try:
            delivered = 0
            for receipt in self.journal.pending_receipts(limit):
                if self._ledger.deliver_receipt(receipt) is not True:
                    raise LedgerUnavailable("ledger receipt not acknowledged")
                self.journal.acknowledge(receipt.receipt_id, receipt.payload_sha256)
                delivered += 1
            return delivered
        finally:
            self._delivery.release()

    def drain(self, *, timeout=None, cancel_queued=False):
        """Does not release device ownership or imply authoritative reconciliation."""
        return self.service.drain(timeout=timeout, cancel_queued=cancel_queued)
