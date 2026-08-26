"""Grand Challenge entrypoint (Task 1 only).

Reads flat /input files, runs the fixed-orchestration agent, writes the two
GC output files to /output. Assumes an Ollama daemon with Gemma 4 E2B is
already serving on localhost inside the container (started by entrypoint.sh).
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from chimera_agent_baseline.run import run_agent
from chimera_agent_baseline.utils import setup_logging

log = logging.getLogger(__name__)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
CONFIG_PATH = Path("/opt/app/configs/config.yaml")


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def run() -> int:
    setup_logging("INFO")

    structured_prompt = load_json_file(location=INPUT_PATH / "structured-prompt.json")
    clinical_data = load_json_file(
        location=INPUT_PATH / "prostate-biopsy-decision-clinical-data.json"
    )
    neural_reps = load_json_file(
        location=INPUT_PATH / "prostate-modality-level-neural-representations.json"
    )

    case_id = structured_prompt.get("case_id", "gc-case")
    structured_prompt.setdefault("case_id", case_id)

    with tempfile.TemporaryDirectory(prefix="chimera-t1-") as tmp:
        tmp_root = Path(tmp)
        case_dir = tmp_root / "task1" / "agent_input" / case_id
        case_dir.mkdir(parents=True)
        write_json_file(location=case_dir / "structured-prompt.json", content=structured_prompt)
        write_json_file(
            location=case_dir / "prostate-biopsy-decision-clinical-data.json", content=clinical_data
        )
        write_json_file(
            location=case_dir / "prostate-modality-level-neural-representations.json",
            content=neural_reps,
        )

        cfg = OmegaConf.load(CONFIG_PATH)
        OmegaConf.update(cfg, "paths.data_root", str(tmp_root))
        OmegaConf.update(cfg, "paths.output_dir", str(tmp_root / "out"))
        OmegaConf.update(cfg, "paths.submodels_dir", "/opt/ml/model")

        log.info("Running CHIMERA task1 fixed-orchestration agent for case %s", case_id)
        asyncio.run(run_agent(cfg))

        produced = tmp_root / "out" / "task1" / case_id
        for name in (
            "prostate-biopsy-decision.json",
            "prostate-biopsy-decision-reasoning.json",
        ):
            src = produced / name
            if not src.is_file():
                raise FileNotFoundError(f"Missing output {src}")
            write_json_file(location=OUTPUT_PATH / name, content=load_json_file(location=src))

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
