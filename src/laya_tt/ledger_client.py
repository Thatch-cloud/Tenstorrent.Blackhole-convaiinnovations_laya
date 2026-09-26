"""Single-attempt, credential-free delivery to a trusted local ledger bridge."""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import socket
import stat
import struct
import time

from jsonschema import Draft202012Validator

from .contracts import load_schema
from .ledger import Receipt
from .ledger_protocol import encode_consumption


class LedgerUnavailable(RuntimeError):
    """No execution permission or receipt acknowledgment; consumption may have committed."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def _json(raw):
    def invalid(_value):
        raise ValueError()
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


_RECEIPT_SCHEMA = load_schema("execution-receipt")
_RECEIPT = Draft202012Validator(_RECEIPT_SCHEMA)
_RUNTIME = Draft202012Validator({"$ref": "#/$defs/runtime",
                               "$defs": _RECEIPT_SCHEMA["$defs"]})


class LocalLedgerClient:
    """One loaded runtime and trusted service UID, provided by host bootstrap.

    Linux SO_PEERCRED is required. No TCP, proxies, redirects, node credentials,
    automatic consumption retry, journal deletion or grant verification. Programs
    sharing the service UID are in the same trust domain.
    """

    def __init__(self, path: Path, expected_uid: int, runtime: Mapping):
        try:
            path = Path(path)
            if (not path.is_absolute() or type(expected_uid) is not int
                    or not 0 <= expected_uid <= 2**32 - 1):
                raise ValueError()
            observed = _json(json.dumps(dict(runtime), allow_nan=False))
            _RUNTIME.validate(observed)
        except Exception:
            raise ValueError("invalid local ledger assignment") from None
        self._path = path
        self._uid = expected_uid
        self._runtime = observed

    def consume_reservation(self, context: Mapping, question_rows: int,
                            encoded_tokens: int) -> bool:
        """AdmissionAdapter callback; claims must already be authenticated.

        An exception means deny execution. Do not retry consumption even after
        a disconnect: the authority may have committed without returning its ACK.
        """
        try:
            body = encode_consumption(context, self._runtime,
                                      question_rows=question_rows, encoded_tokens=encoded_tokens)
            # Use our owned encoded snapshot for the deadline, not mutable input.
            deadline_ms = _json(body)["deadline_unix_ms"]
            remaining = min(15.0, deadline_ms / 1000 - time.time())
            if remaining <= 0:
                raise ValueError()
            self._exchange("/v1/ledger/consume", body, {}, 204,
                           time.monotonic() + remaining)
            return True
        except Exception:
            raise LedgerUnavailable("ledger consumption not acknowledged") from None

    def journaled_consumption(self, journal):
        """Build the admission callback using the SAME journal as DecisionService.

        Crash/uncertainty retains the intent, including after a remote ACK but
        before local admission. Reconciliation must never replay consumption.
        """
        from .ledger import ExecutionJournal
        if not isinstance(journal, ExecutionJournal):
            raise ValueError("durable execution journal required")

        def consume(context, question_rows, encoded_tokens):
            context = _json(json.dumps(dict(context), allow_nan=False))
            payload = journal.begin_consumption(context, self._runtime, question_rows, encoded_tokens)
            if self.consume_reservation(context, question_rows, encoded_tokens) is not True:
                raise LedgerUnavailable("ledger consumption not acknowledged")
            journal.consumption_acknowledged(context, payload)
            return True

        return consume

    def recover_pending(self, journal, limit=8):
        """Recover before application construction, using the original runtime assignment.

        Only a verified unconsumed fence resolves an intent. Other outcomes are
        observations, never permission to execute or acknowledge a local receipt.
        A failed call preserves unresolved evidence; recovery may be retried.
        """
        from .ledger import ExecutionJournal
        if not isinstance(journal, ExecutionJournal) or type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("durable journal and recovery limit between 1 and 64 required")
        observations = []
        for intent in journal.pending_consumptions(limit):
            try:
                body = intent["consumption"]
                value = _json(body)
                context = _json(intent["context_json"])
                # Validate the saved claims and closed payload without replacing
                # its original bytes (the authority ACK hashes those bytes).
                expected = encode_consumption(context, self._runtime,
                    question_rows=value["prepared_question_rows"],
                    encoded_tokens=value["prepared_encoded_tokens"])
                if (type(value["deadline_unix_ms"]) is not int
                        or any(type(value["binding"][key]) is not int
                               for key in ("max_question_rows", "max_encoded_tokens"))
                        or len(body) > 16384 or value != _json(expected)):
                    raise ValueError()
                raw = self._exchange("/v1/ledger/recover", body, {}, 200, time.monotonic() + 15)
                ack = _json(raw)
                if (type(ack) is not dict
                        or set(ack) != {"schema_version", "request_body_sha256", "outcome"}
                        or ack["schema_version"] != "1"
                        or ack["request_body_sha256"] != hashlib.sha256(body).hexdigest()):
                    raise ValueError()
                outcome = ack["outcome"]
                if type(outcome) is not dict:
                    raise ValueError()
                state = outcome.get("state")
                if state == "unconsumed_fenced":
                    if set(outcome) != {"state"}:
                        raise ValueError()
                    journal.confirm_unconsumed_fence(context, body)
                elif state == "consumed":
                    if set(outcome) != {"state", "prepared_question_rows", "prepared_encoded_tokens"}:
                        raise ValueError()
                    for key, ceiling in (("prepared_question_rows", "max_question_rows"),
                                         ("prepared_encoded_tokens", "max_encoded_tokens")):
                        if type(outcome[key]) is not int or not 1 <= outcome[key] <= context[ceiling]:
                            raise ValueError()
                elif state == "settled":
                    if (set(outcome) != {"state", "receipt_id", "payload_sha256"}
                            or not isinstance(outcome["receipt_id"], str)
                            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", outcome["receipt_id"])
                            or not isinstance(outcome["payload_sha256"], str)
                            or not re.fullmatch(r"[0-9a-f]{64}", outcome["payload_sha256"])):
                        raise ValueError()
                else:
                    raise ValueError()
                observations.append({"tenant": intent["tenant"], "attempt": intent["attempt"],
                                     "outcome": outcome})
            except Exception:
                raise LedgerUnavailable("ledger recovery not acknowledged") from None
        return observations

    def deliver_receipt(self, receipt: Receipt) -> bool:
        """Return True only for exact ACK; caller retains outbox evidence otherwise."""
        try:
            if not isinstance(receipt, Receipt):
                raise ValueError()
            body = receipt.payload_json.encode("utf-8")
            if (len(body) > 16384
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", receipt.receipt_id)
                    or hashlib.sha256(body).hexdigest() != receipt.payload_sha256):
                raise ValueError()
            value = _json(body)
            _RECEIPT.validate(value)
            if value["runtime"] != self._runtime:
                raise ValueError()
            raw = self._exchange("/v1/ledger/receipts", body,
                {"x-thatch-receipt-id": receipt.receipt_id,
                 "x-thatch-receipt-sha256": receipt.payload_sha256},
                200, time.monotonic() + 15)
            if _json(raw) != {"receipt_id": receipt.receipt_id,
                              "payload_sha256": receipt.payload_sha256}:
                raise ValueError()
            return True
        except Exception:
            raise LedgerUnavailable("ledger receipt not acknowledged") from None

    @staticmethod
    def _timeout(stream, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError()
        stream.settimeout(remaining)

    def _exchange(self, route, body, headers, expected_status, deadline):
        # Revalidate path and kernel peer on every new connection. The launcher
        # owns path lifetime; private mode checks are not protection against root
        # or hostile processes sharing this trusted UID.
        if not hasattr(socket, "SO_PEERCRED"):
            raise OSError()
        directory = self._path.parent.lstat()
        endpoint = self._path.lstat()
        if (self._path.parent.resolve(strict=True) != self._path.parent
                or not stat.S_ISDIR(directory.st_mode)
                or not stat.S_ISSOCK(endpoint.st_mode)
                or directory.st_uid != self._uid or endpoint.st_uid != self._uid
                or directory.st_mode & 0o077 or endpoint.st_mode & 0o177):
            raise OSError()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            self._timeout(stream, deadline)
            stream.connect(str(self._path))
            credentials = stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                           struct.calcsize("iII"))
            _pid, uid, _gid = struct.unpack("iII", credentials)
            if uid != self._uid:
                raise OSError()
            fields = {"Host": "localhost", "Connection": "close",
                      "Content-Type": "application/json", "Content-Length": str(len(body)),
                      **headers}
            request = f"POST {route} HTTP/1.1\r\n" + "".join(
                f"{key}: {value}\r\n" for key, value in fields.items()) + "\r\n"
            self._timeout(stream, deadline)
            stream.sendall(request.encode("ascii") + body)
            return self._response(stream, expected_status, deadline)

    def _response(self, stream, expected_status, deadline):
        # The bridge uses length-delimited responses. Accept only that bounded
        # subset of HTTP; no informational responses, chunking or redirects.
        raw = bytearray()
        while b"\r\n\r\n" not in raw:
            self._timeout(stream, deadline)
            block = stream.recv(min(1024, 8192 - len(raw)))
            if not block:
                raise ValueError()
            raw.extend(block)
            if len(raw) >= 8192 and b"\r\n\r\n" not in raw:
                raise ValueError()
        head, body = bytes(raw).split(b"\r\n\r\n", 1)
        lines = head.decode("ascii").split("\r\n")
        status = lines[0].split(" ", 2)
        if len(status) != 3 or status[:2] != ["HTTP/1.1", str(expected_status)]:
            raise ValueError()
        fields = {}
        for line in lines[1:]:
            key, value = line.split(":", 1)
            key = key.lower()
            if not re.fullmatch(r"[a-z0-9-]+", key) or key in fields:
                raise ValueError()
            fields[key] = value.strip()
        length = fields.get("content-length", "0" if expected_status == 204 else "")
        if ("transfer-encoding" in fields or not re.fullmatch(r"[0-9]{1,5}", length)
                or int(length) > 4096 or (expected_status == 204 and int(length) != 0)):
            raise ValueError()
        length = int(length)
        while len(body) < length:
            self._timeout(stream, deadline)
            block = stream.recv(length - len(body))
            if not block:
                raise ValueError()
            body += block
        if len(body) != length:
            raise ValueError()
        self._timeout(stream, deadline)
        return body
