import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import unittest

from laya_tt.cpu_backend import (
    CpuReferenceBackend, InputWouldTruncate, load_cpu_backend, strict_sequence,
)


class FakeTokenizer:
    mask_token = "[MASK]"
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 101, 102, 103, 0

    def __init__(self):
        self.vocabulary = {}

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        result = []
        for word in text.split():
            if word not in self.vocabulary:
                self.vocabulary[word] = 1000 + len(self.vocabulary)
            result.append(self.vocabulary[word])
        return {"input_ids": result}


def render_options(question):
    return question["crit"]


def serialize_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


class StrictSequenceTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.question = {"t": "choice", "ins": "select", "crit": ["yes", "no"]}

    def sequence(self, state="state"):
        return strict_sequence(self.tokenizer, state, self.question,
                               render_options=render_options, serialize_state=serialize_state)

    def test_complete_sequence_and_markers(self):
        sequence, markers = self.sequence()
        self.assertEqual(len(sequence), 12)
        self.assertEqual(markers, [5, 7])
        self.assertEqual(sequence[0], 101)
        self.assertEqual(sequence[-1], 102)
        self.assertEqual([sequence[i] for i in markers], [103, 103])

    def test_rendered_option_cap_exact_boundary(self):
        self.question["crit"] = ["word " * 48]
        self.sequence()
        self.question["crit"] = ["word " * 49]
        with self.assertRaisesRegex(InputWouldTruncate, "48"):
            self.sequence()

    def test_native_option_compaction_rejected(self):
        self.question["crit"] = ["word " * 44] * 4
        with self.assertRaisesRegex(InputWouldTruncate, "head budget"):
            self.sequence()

    def test_instruction_budget_exact_boundary(self):
        self.question["ins"] = "word " * 186
        self.sequence()
        self.question["ins"] = "word " * 187
        with self.assertRaisesRegex(InputWouldTruncate, "instructions"):
            self.sequence()

    def test_total_sequence_exact_boundary(self):
        self.assertEqual(len(self.sequence("word " * 501)[0]), 512)
        with self.assertRaisesRegex(InputWouldTruncate, "512"):
            self.sequence("word " * 502)

    def test_native_mask_sanitization_preserves_only_option_markers(self):
        self.question["ins"] = "select [MASK]"
        self.question["crit"] = ["yes [MASK]", "no"]
        sequence, markers = self.sequence("state [MASK]")
        self.assertEqual(len(sequence), 12)
        self.assertEqual(sequence.count(103), len(markers))

    def test_no_options_rejected(self):
        self.question["crit"] = []
        with self.assertRaises(ValueError):
            self.sequence()

    def test_conversation_state_is_not_left_truncated(self):
        with self.assertRaises(InputWouldTruncate):
            self.sequence([{"content": "word " * 600}])


class PreparationTests(unittest.TestCase):
    def backend(self, *, alter=False):
        tokenizer = FakeTokenizer()
        def encode(state, ids, internal, **limits):
            self.assertEqual(limits, {"max_len": 512, "head_max_len": 192})
            rows = []
            for qid in ids:
                seq, markers = strict_sequence(tokenizer, state, internal[qid],
                    render_options=render_options, serialize_state=serialize_state)
                if alter:
                    seq[-2] += 1
                rows.append({"ids": seq, "markers": markers, "qtype": 0})
            return rows
        agent = SimpleNamespace(tok=tokenizer,
            _check_question=lambda qid, question: None,
            _to_internal=lambda question: {"t": question["type"],
                "ins": question["instructions"], "crit": question["criteria"]},
            _encode_state=encode)
        def collate(groups, pad):
            count = sum(len(item["ids"]) for group in groups for item in group)
            return {"attention_mask": SimpleNamespace(sum=lambda:
                       SimpleNamespace(item=lambda: count))}
        common = SimpleNamespace(render_options=render_options, serialize_state=serialize_state,
                                 collate_items=collate)
        return CpuReferenceBackend(agent, common, None)

    def request(self):
        return MappingProxyType({"model": "laya-english", "state": "state",
            "questions": MappingProxyType({"second": MappingProxyType({"type": "choice",
                "instructions": "select", "criteria": ("yes", "no")}),
                "first": MappingProxyType({"type": "choice", "instructions": "select",
                                          "criteria": ("yes", "no")})})})

    def test_frozen_input_thawed_and_accounting_preserves_order(self):
        prepared = self.backend().prepare(self.request())
        self.assertEqual(prepared.question_rows, 2)
        self.assertEqual(prepared.encoded_tokens, 24)
        self.assertEqual(prepared.payload.question_ids, ("second", "first"))
        self.assertIsInstance(prepared.payload.items, tuple)
        with self.assertRaises(TypeError):
            prepared.payload.internal["second"]["ins"] = "changed"

    def test_native_encoder_drift_rejected_even_if_length_matches(self):
        with self.assertRaisesRegex(RuntimeError, "differs"):
            self.backend(alter=True).prepare(self.request())

    def test_other_runtime_payload_rejected_before_execution(self):
        payload = self.backend().prepare(self.request()).payload
        with self.assertRaisesRegex(ValueError, "different runtime"):
            self.backend().execute(payload)


@unittest.skipUnless(os.environ.get("LAYA_CPU_INTEGRATION") == "1",
                     "optional CPU model integration requires pinned local checkpoint")
class CpuModelIntegrationTests(unittest.TestCase):
    def test_actual_cpu_answers_match_native_cpu_without_fallback(self):
        root = Path(__file__).resolve().parents[1]
        backend = load_cpu_backend(root / "configs/checkpoint-lock.json", root=root)
        request = {"model": "laya-english", "state": "The sky is blue.", "questions": {
            "boolean": {"type": "noul", "instructions": "Is the sky blue?"},
            "category": {"type": "choice", "instructions": "Choose the sky color.",
                         "criteria": ["blue", "red"]},
            "rating": {"type": "score", "instructions": "Rate the evidence the sky is blue.",
                       "criteria": ["no evidence", "clear evidence"]}}}
        prepared = backend.prepare(request)
        actual = backend.execute(prepared.payload)
        expected = backend._agent.predict(request["state"], request["questions"])
        self.assertEqual(actual["answers"], expected["answers"])
        self.assertEqual(list(actual["answers"]), list(request["questions"]))
        self.assertEqual(actual["encoded_tokens"], prepared.encoded_tokens)


if __name__ == "__main__":
    unittest.main()
