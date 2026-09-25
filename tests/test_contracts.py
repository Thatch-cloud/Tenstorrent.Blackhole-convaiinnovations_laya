"""Executable wire-contract checks; no model outputs or authentication claims."""
import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator
from laya_tt.contracts import load_schema

ROOT = Path(__file__).resolve().parents[1]


def validator(name):
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / 'schemas/fixtures/request-all-types.json').read_text())
        self.public = validator('decision-request')
        self.internal = validator('admission-context')
        self.context = {
            'schema_version': '1', 'request_id': 'server-request',
            'execution_attempt_id': 'server-attempt', 'usage_event_id': 'server-usage',
            'tenant_id': 'tenant-a', 'key_id': 'key-a', 'runtime_generation': 'generation-a',
            'policy_revision': 'policy-a', 'reservation_id': 'reservation-a',
            'model': 'laya-english', 'checkpoint_revision': 'a' * 40,
            'issued_at_unix_ms': 1000, 'deadline_unix_ms': 2000,
            'max_question_rows': 3, 'max_encoded_tokens': 1536,
            'request_sha256': 'b' * 64,
        }

    def test_all_schemas_are_valid(self):
        for name in ('decision-request', 'decision-response', 'admission-context', 'execution-receipt'):
            validator(name)

    def test_all_upstream_question_types(self):
        self.public.validate(self.request)
        self.request['questions']['route']['criteria'] = ['shipping']
        self.request['questions']['urgency']['criteria'] = ['Routine']
        self.request['questions']['delivered'].pop('criteria')
        self.public.validate(self.request)

    def test_unrecognized_model_never_autoroutes(self):
        for name in ('english', 'jev-1', 'multilingual', '', '../weights'):
            with self.subTest(model=name):
                self.request['model'] = name
                self.assertFalse(self.public.is_valid(self.request))

    def test_public_boundary_rejects_authority_fields(self):
        for field in (*self.context, 'admission_context', 'api_key', 'authorization', 'tenant'):
            if field == 'model':
                continue
            with self.subTest(field=field):
                request = copy.deepcopy(self.request)
                request[field] = self.context.get(field, 'attacker-controlled')
                self.assertFalse(self.public.is_valid(request))

    def test_question_cannot_smuggle_authority(self):
        for field in ('tenant_id', 'reservation_id', 'deadline_unix_ms', 'hooks'):
            request = copy.deepcopy(self.request)
            request['questions']['route'][field] = 'tenant-b'
            self.assertFalse(self.public.is_valid(request))

    def test_state_is_data_not_authority(self):
        # Arbitrary JSON state can contain these words. It must remain opaque model
        # input; adapters must NEVER merge it into their authenticated envelope.
        self.request['state'] = {'tenant_id': 'tenant-b', 'admission_context': self.context}
        self.public.validate(self.request)

    def test_question_shape_and_resource_bounds(self):
        bad_questions = [
            {}, {'q': {'type': 'chat', 'instructions': 'x'}},
            {'q': {'type': 'choice', 'instructions': 'x', 'criteria': []}},
            {'q': {'type': 'choice', 'instructions': 'x', 'criteria': ['a', 'a']}},
            {'q': {'type': 'score', 'instructions': 'x', 'criteria': [None]}},
            {'q': {'type': 'score', 'instructions': 'x', 'criteria': ['x'] * 65}},
            {'q': {'type': 'noul', 'instructions': 'x', 'criteria': {'yes': 'x'}}},
            {'q': {'type': 'choice', 'instructions': 'x', 'criteria': ['x'], 'labels': {}}},
        ]
        bad_questions.append({f'q{i}': {'type': 'noul', 'instructions': 'x'} for i in range(65)})
        for questions in bad_questions:
            with self.subTest(questions=questions):
                request = {**self.request, 'questions': questions}
                self.assertFalse(self.public.is_valid(request))

    def test_internal_context_requires_all_security_metadata(self):
        self.internal.validate(self.context)
        for field in self.context:
            context = dict(self.context)
            del context[field]
            with self.subTest(missing=field):
                self.assertFalse(self.internal.is_valid(context))
        self.assertFalse(self.internal.is_valid(self.request))
        self.assertFalse(self.public.is_valid(self.context))

    def test_internal_context_rejects_unbounded_or_malformed_reservations(self):
        for field, value in [('max_question_rows', 0), ('max_question_rows', 65),
                             ('max_encoded_tokens', 32769), ('deadline_unix_ms', -1),
                             ('request_sha256', 'not-a-hash'), ('checkpoint_revision', 'main'),
                             ('tenant_id', ''), ('api_key', 'secret')]:
            with self.subTest(field=field):
                self.assertFalse(self.internal.is_valid({**self.context, field: value}))

    def test_schema_validation_does_not_claim_to_authenticate_tenant(self):
        # Both are structurally valid. Only an authenticated issuer and binding
        # verification can reject the spoof; this intentionally exposes that limit.
        self.internal.validate(self.context)
        self.internal.validate({**self.context, 'tenant_id': 'tenant-b'})

    def test_response_usage_forbids_generated_tokens(self):
        usage = validator('decision-response').schema['properties']['usage']
        v = Draft202012Validator(usage)
        observed = {'requests': 1, 'question_rows': 3, 'encoded_tokens': 120,
                    'output_tokens': 0, 'queue_ms': 1, 'execution_ms': 10}
        v.validate(observed)  # Synthetic metering shape, never an inference fixture.
        self.assertFalse(v.is_valid({**observed, 'output_tokens': 1}))
        self.assertFalse(v.is_valid({**observed, 'completion_tokens': 1}))

    def test_unknown_question_result_is_rejected_without_inventing_predictions(self):
        answer_schema = validator('decision-response').schema['properties']['answers']
        self.assertFalse(Draft202012Validator(answer_schema).is_valid({'q': {'type': 'chat'}}))


if __name__ == '__main__':
    unittest.main()
