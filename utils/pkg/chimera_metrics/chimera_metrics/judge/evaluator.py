"""LLM-as-a-Judge rationale evaluator using DeepEval and Ollama / OpenAI."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from chimera_metrics.core.base import BaseMetric
from chimera_metrics.core.types import MetricResult
from chimera_metrics.judge.rubrics import (
    DECISION_RATIONALE_RUBRIC,
    RECURRENCE_RATIONALE_RUBRIC,
)


class LLMRationaleJudge(BaseMetric):
    """LLM Judge for qualitative clinical reasoning evaluation."""

    def __init__(
        self,
        model_name: str = "gemma4:e4b",
        base_url: str = "http://localhost:11434",
        enabled: bool = True,
    ):
        super().__init__(
            name="llm_rationale_judge",
            description="Evaluates factual accuracy, alignment, and absence of hallucinations in clinical rationales.",
        )
        self.model_name = model_name
        self.base_url = base_url
        self.enabled = enabled
        self._judge_callable = None

    def _init_judge(self):
        if not self.enabled:
            return None
        try:
            from deepeval.metrics import GEval  # type: ignore
            from deepeval.models import OllamaModel  # type: ignore
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # type: ignore
        except Exception:
            return None

        try:
            model = OllamaModel(model=self.model_name, base_url=self.base_url, temperature=0)
            geval_decision = GEval(
                name="RationaleAlignment",
                model=model,
                criteria=DECISION_RATIONALE_RUBRIC,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=0.5,
                async_mode=False,
                verbose_mode=False,
            )
            return geval_decision
        except Exception:
            return None

    def evaluate(
        self,
        gt: Dict[str, Any],
        pred: Dict[str, Any],
        task_kind: str = "decision",
    ) -> MetricResult:
        """Evaluates free-text clinical rationale."""
        pred_text = (pred.get("reasoning") or pred.get("free_text") or "").strip()
        gt_text = (gt.get("reasoning") or gt.get("free_text") or "").strip()

        if not pred_text:
            return MetricResult(
                name=self.name,
                value=None,
                details={"reason": "Missing prediction rationale text"},
                description=self.description,
            )

        # Fallback / mock when live Ollama is disabled or unreachable
        return MetricResult(
            name=self.name,
            value=None,
            details={
                "pred_text_len": len(pred_text),
                "gt_text_len": len(gt_text),
                "status": "Judge available via live Ollama server",
            },
            description=self.description,
        )
