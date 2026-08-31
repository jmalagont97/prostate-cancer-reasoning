from types import SimpleNamespace
from typing import Any, cast

from chimera_agent.tasks.task2.baseline.form_fill import make_form_fill_node


class FakeModel:
    def invoke(self, messages):
        return SimpleNamespace(
            content=(
                '{"case_id":"wrong","task":2,'
                '"treatment_decision":"Active Surveillance",'
                '"confidence":"clear","variable_weights":{},'
                '"reasoning":"PSA 4.2 ng/mL and PI-RADS 3 support the treatment recommendation for this case."}'
            )
        )


def test_form_fill_normalizes_legacy_treatment_label_before_validation():
    node = make_form_fill_node(cast(Any, FakeModel()), max_retries=1)
    result = node(
        {
            "case_id": "T2-001",
            "context": "PSA 4.2 ng/mL; PI-RADS 3; age 65.",
            "tool_outputs": {
                "treatment_decision_support": {
                    "decision": "Active Surveillance",
                    "modality_probabilities": {},
                    "treatment_probabilities": {},
                    "neighbors_union": [],
                },
                "confidence_estimate": {"confidence": "clear"},
                "variable_weights_estimate": {"variable_weights": {}},
            },
        }
    )

    assert result["structured_response"]["treatment_decision"] == "active_surveillance"
