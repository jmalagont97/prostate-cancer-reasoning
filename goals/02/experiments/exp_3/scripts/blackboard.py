#!/usr/bin/env python3
"""blackboard.py

Thread-safe and section-aware blackboard manager for Goal 02 Decision Tree Induction.
Reads, updates, and appends structured markdown sections delimited by
<!-- SECTION: <name> --> ... <!-- END_SECTION: <name> --> tags.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional


DEFAULT_CLINICAL_BLACKBOARD_TEMPLATE = """# 📋 Multi-Agent Blackboard: Clinical Decision Tree Induction (Task 1.1)

This shared blackboard serves as the live working memory and turn-by-turn dialogue log for:
1. **Clinical Oncology Oracle**: Queries EAU/ESMO guidelines via ChromaDB to propose evidence-backed splitting rules.
2. **Empirical Gini Evaluator**: Evaluates candidate splits strictly on the training partition (`split == 'train'`, N=45) and calculates Delta Gini.
3. **Consolidator & Arbiter**: Manages tree topology, enforces patience budget (k=3), determines early stopping, and directs agent actions.

---

<!-- SECTION: human_guidance -->
### 1. Human Clinical Directive (RCT)
*Pending initialization...*
<!-- END_SECTION: human_guidance -->

---

<!-- SECTION: problem_spec -->
### 2. Clinical Problem & Dataset Specification
*Pending initialization by run_agentic_induction.py...*
<!-- END_SECTION: problem_spec -->

---

<!-- SECTION: active_tree_state -->
### 3. Live Decision Tree Architecture
```text
(Tree uninitialized)
```
<!-- END_SECTION: active_tree_state -->

---

<!-- SECTION: rejected_hypotheses -->
### 4. Rejected Candidate Hypotheses Log
*(No rejected hypotheses yet)*
<!-- END_SECTION: rejected_hypotheses -->

---

<!-- SECTION: hypothesis_log -->
### 5. Step-by-Step Multi-Agent Reasoning & Execution Log
<!-- END_SECTION: hypothesis_log -->

---

<!-- SECTION: final_model_synthesis -->
### 6. Final Model Synthesis & Verification
*Induction in progress...*
<!-- END_SECTION: final_model_synthesis -->
"""


class Blackboard:
    """Manages section-based reads, updates, and appends to blackboard.md."""

    def __init__(self, blackboard_path: Path):
        self.path = blackboard_path
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(DEFAULT_CLINICAL_BLACKBOARD_TEMPLATE, encoding="utf-8")

    def _read_content(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def _write_content(self, content: str) -> None:
        self.path.write_text(content, encoding="utf-8")

    def load_content(self) -> str:
        """Return full raw markdown content of blackboard."""
        return self._read_content()

    def read_section(self, section_name: str) -> Optional[str]:
        content = self._read_content()
        pattern = rf"<!-- SECTION: {re.escape(section_name)} -->\s*(.*?)\s*<!-- END_SECTION: {re.escape(section_name)} -->"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None

    def write_section(self, section_name: str, new_text: str) -> None:
        content = self._read_content()
        pattern = rf"(<!-- SECTION: {re.escape(section_name)} -->\s*).*?(\s*<!-- END_SECTION: {re.escape(section_name)} -->)"
        replacement = rf"\g<1>{new_text}\g<2>"
        if re.search(pattern, content, re.DOTALL):
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        else:
            new_content = content + f"\n\n<!-- SECTION: {section_name} -->\n{new_text}\n<!-- END_SECTION: {section_name} -->\n"
        self._write_content(new_content)

    def append_section(self, section_name: str, append_text: str) -> None:
        current = self.read_section(section_name) or ""
        updated = f"{current}\n\n{append_text}".strip() if current else append_text
        self.write_section(section_name, updated)

    def get_all_sections(self) -> Dict[str, str]:
        content = self._read_content()
        pattern = r"<!-- SECTION: ([\w\-]+) -->\s*(.*?)\s*<!-- END_SECTION: \1 -->"
        matches = re.findall(pattern, content, re.DOTALL)
        return {m[0]: m[1].strip() for m in matches}
