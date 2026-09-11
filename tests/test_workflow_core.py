from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import workflow_core as core


class WorkflowCoreTests(unittest.TestCase):
    def test_extract_json_from_markdown(self) -> None:
        parsed = core.extract_json('```json\n{"plausible": true}\n```')
        self.assertEqual(parsed, {"plausible": True})

    def test_fail_open_on_malformed_audit(self) -> None:
        retained, status, plausible, confidence = core.interpret_audit_decision(
            "type_audit", {"plausible": False, "confidence_score": "bad"}
        )
        self.assertTrue(retained)
        self.assertEqual(status, "RetainedOnMalformedDecision")
        self.assertFalse(plausible)
        self.assertIsNone(confidence)

    def test_confidence_rule(self) -> None:
        low_confidence = core.interpret_audit_decision(
            "autonomous_audit",
            {"plausible": False, "confidence_score": 0.69},
        )
        high_confidence = core.interpret_audit_decision(
            "autonomous_audit",
            {"plausible": False, "confidence_score": 0.70},
        )
        self.assertTrue(low_confidence[0])
        self.assertFalse(high_confidence[0])

    def test_physical_audit_has_no_confidence_gate(self) -> None:
        decision = core.interpret_audit_decision(
            "physical_audit", {"plausible": False}
        )
        self.assertEqual(decision, (False, "Success", False, None))

    def test_normalize_doi(self) -> None:
        self.assertEqual(
            core.normalize_doi("https://doi.org/10.1000/ABC"),
            "10.1000/abc",
        )


if __name__ == "__main__":
    unittest.main()
