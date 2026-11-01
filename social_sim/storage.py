import json
from pathlib import Path
from typing import Any

from pydantic import Field

from social_sim.engine import Simulation
from social_sim.models import Model, WorldState


class Snapshot(Model):
    world: WorldState
    rng_state: list[Any] = Field(min_length=3, max_length=3)


def save_snapshot(simulation: Simulation, path: Path) -> None:
    # Validate nested references even when in-memory collections were changed.
    WorldState.model_validate(simulation.world.model_dump())
    snapshot = Snapshot(world=simulation.world, rng_state=list(simulation.rng.getstate()))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(snapshot.model_dump_json(indent=2), encoding='utf-8')
    temporary.replace(path)


def load_snapshot(path: Path) -> Simulation:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or not isinstance(raw.get('world'), dict):
        raise ValueError('Снимок должен быть объектом с объектом world')
    version_number = raw['world'].get('schema_version')
    if version_number != 3:
        raise ValueError(f'Неподдерживаемая версия снимка {version_number}; требуется 3. Неявная миграция не выполняется.')
    snapshot = Snapshot.model_validate(raw)
    simulation = Simulation(snapshot.world.scenario)
    simulation.world = snapshot.world
    version, internal, gauss = snapshot.rng_state
    try:
        simulation.rng.setstate((version, tuple(internal), gauss))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Некорректное состояние RNG') from exc
    return simulation


def export_run(simulation: Simulation, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_snapshot(simulation, directory / 'snapshot.json')
    (directory / 'events.jsonl').write_text(''.join(e.model_dump_json()+'\n' for e in simulation.world.events), encoding='utf-8')
    (directory / 'traces.jsonl').write_text(''.join(t.model_dump_json()+'\n' for t in simulation.world.traces), encoding='utf-8')
    (directory / 'relationships.json').write_text(json.dumps([r.model_dump() for r in simulation.world.relationships.values()], ensure_ascii=False, indent=2), encoding='utf-8')
