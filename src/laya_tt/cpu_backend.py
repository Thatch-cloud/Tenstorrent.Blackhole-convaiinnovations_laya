"""Pinned CPU/FP32 execution with explicit rejection of native input truncation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any
import warnings

from .admission import PreparedInput, RequestTooLarge
from .reference import checkpoint_files, validate_manifest


class InputWouldTruncate(RequestTooLarge):
    """The upstream encoder would silently discard request content."""


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def strict_sequence(tokenizer, state, question, *, render_options, serialize_state):
    """Construct the expected complete row using pinned upstream text rendering.

    Native caps are 48 tokens per rendered option, 192 head tokens including
    markers, and 512 total tokens including special tokens and repeated state.
    The upstream option-compaction branch reserves at least 16 instruction tokens.
    """
    def tokens(text):
        return list(tokenizer(text.replace(tokenizer.mask_token, " "),
                              add_special_tokens=False)["input_ids"])

    head = tokens(f"{question['t']} question: {question['ins']}")
    options = [tokens(" " + option) for option in render_options(question)]
    if not options:
        raise ValueError("at least one decision option is required")
    if any(len(option) > 48 for option in options):
        raise InputWouldTruncate("rendered option exceeds 48 tokens")
    option_size = sum(1 + len(option) for option in options)
    instruction_budget = 192 - option_size
    if instruction_budget < 16:
        raise InputWouldTruncate("options exceed native head budget")
    if len(head) > instruction_budget:
        raise InputWouldTruncate("instructions exceed remaining native head budget")
    sequence = [tokenizer.cls_token_id, *head, tokenizer.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(sequence))
        sequence.extend([tokenizer.mask_token_id, *option])
    sequence.append(tokenizer.sep_token_id)
    sequence.extend(tokens(serialize_state(state)))
    sequence.append(tokenizer.sep_token_id)
    if len(sequence) > 512:
        raise InputWouldTruncate("complete question row exceeds 512 tokens")
    return sequence, markers


@dataclass(frozen=True)
class CpuPreparedPayload:
    """Owned CPU tensors; only this backend may consume this payload.

    The envelope/maps are immutable. Tensor buffers remain mutable by PyTorch;
    admission and worker retain exclusive ownership and must not expose them.
    """
    owner: object
    question_ids: tuple[str, ...]
    internal: Mapping
    items: tuple
    batch: Mapping
    encoded_tokens: int


class CpuReferenceBackend:
    """Use ``load_cpu_backend`` to obtain a verified pinned implementation.

    ``prepare`` tokenizes only; ``execute`` performs one CPU model call without
    invoking upstream fallback, hooks, automatic truncation, or alternate devices.
    Execute through SerializedWorker: this object is not a concurrent engine.
    """
    def __init__(self, agent, common, torch_module, *, initialization_warnings=()):
        self._agent, self._common, self._torch = agent, common, torch_module
        self._owner = object()
        self.initialization_warnings = tuple(initialization_warnings)

    def prepare(self, request: Mapping) -> PreparedInput:
        request = _thaw(request)
        if request.get("model") != "laya-english":
            raise ValueError("unsupported model")
        questions, state = request["questions"], request["state"]
        ids = list(questions)
        if not 1 <= len(ids) <= 64:
            raise ValueError("question rows must be between 1 and 64")
        for qid in ids:
            self._agent._check_question(qid, questions[qid])
        internal = {qid: self._agent._to_internal(questions[qid]) for qid in ids}
        expected = [strict_sequence(self._agent.tok, state, internal[qid],
                                   render_options=self._common.render_options,
                                   serialize_state=self._common.serialize_state)
                    for qid in ids]
        items = self._agent._encode_state(state, ids, internal, max_len=512, head_max_len=192)
        if len(items) != len(ids):
            raise RuntimeError("upstream encoder changed question row count")
        for item, (sequence, markers) in zip(items, expected):
            if list(item["ids"]) != sequence or list(item["markers"]) != markers:
                raise RuntimeError("upstream encoding differs from complete untruncated input")
        batch = self._common.collate_items([items], self._agent.tok.pad_token_id)
        encoded_tokens = sum(len(item["ids"]) for item in items)
        if int(batch["attention_mask"].sum().item()) != encoded_tokens:
            raise RuntimeError("collated token accounting mismatch")
        payload = CpuPreparedPayload(self._owner, tuple(ids), _freeze(internal),
                                     _freeze(items), MappingProxyType(batch), encoded_tokens)
        return PreparedInput(len(items), encoded_tokens, payload)

    def execute(self, payload: CpuPreparedPayload) -> dict:
        if not isinstance(payload, CpuPreparedPayload) or payload.owner is not self._owner:
            raise ValueError("payload belongs to a different runtime generation")
        torch, agent = self._torch, self._agent
        if agent.device.type != "cpu" or agent.amp_enabled or agent._fast is not None:
            raise RuntimeError("reference runtime changed device or precision mode")
        if any(p.device.type != "cpu" or (p.is_floating_point() and p.dtype != torch.float32)
               for p in agent.model.parameters()):
            raise RuntimeError("reference model is not CPU float32")
        batch = payload.batch
        with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
            logits, act_logits = agent.model(*(batch[name] for name in
                ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
            if not torch.isfinite(logits).all() or not torch.isfinite(act_logits).all():
                raise RuntimeError("reference produced nonfinite logits")
            answers = agent._decode_answers(
                logits.float().cpu().numpy(), torch.softmax(act_logits.float(), -1).cpu().numpy(),
                _thaw(payload.items), list(payload.question_ids), _thaw(payload.internal), offset=0)
        if list(answers) != list(payload.question_ids):
            raise RuntimeError("decoded question identity or order changed")
        return {"answers": answers, "encoded_tokens": payload.encoded_tokens}

    __call__ = execute


def load_cpu_backend(manifest_path: Path, *, root: Path) -> CpuReferenceBackend:
    """Verify exact source/checkpoint lock, then explicitly load CPU float32.

    No network download or accelerator fallback is provided. Initialization that
    rewrites checkpoint files is rejected, requiring deliberate normalization and
    repinning outside this path. Calibration warnings are retained as metadata.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    source, checkpoint = validate_manifest(manifest, root)
    if os.environ.get("LAYA_CPU_AMP", "").lower() in ("bf16", "bfloat16"):
        raise ValueError("CPU reference requires LAYA_CPU_AMP disabled")
    sys.path.insert(0, str(source))
    import torch
    import laya.agent as upstream_agent
    import laya.common as common
    for module, filename in ((upstream_agent, "agent.py"), (common, "common.py")):
        if Path(module.__file__).resolve() != (source / "laya" / filename).resolve():
            raise ValueError("imported Laya differs from pinned source")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = upstream_agent.Agent(str(checkpoint), device="cpu", compile=False, fast=False)
    if agent.device.type != "cpu" or agent.amp_enabled or agent._fast is not None:
        raise ValueError("reference initialization changed device or precision mode")
    if checkpoint_files(checkpoint) != manifest["checkpoint"]["files"]:
        raise ValueError("upstream initialization mutated checkpoint")
    if agent.cfg.get("max_len", 512) != 512 or agent.cfg.get("head_max_len", 192) != 192:
        raise ValueError("checkpoint does not use validated 512/192 native limits")
    agent.model.float().eval()
    from .integrity import verify_loaded_state
    verify_loaded_state(agent.model, checkpoint, expected_sha256=manifest["checkpoint"]["files"]["model.safetensors"])
    return CpuReferenceBackend(agent, common, torch,
                               initialization_warnings=[str(w.message) for w in caught])
