from pathlib import Path

import yaml

from social_sim.models import Scenario


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
