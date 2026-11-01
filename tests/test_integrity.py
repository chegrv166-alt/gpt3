from pathlib import Path

import pytest

from social_sim.authoring import load_scenario
from social_sim.engine import Simulation
from social_sim.models import Scenario, WorldState


def completed():
    s = Simulation(load_scenario(Path('data/fixtures/school.yaml')))
    s.world.tick = 2
    s.act('n1', 'invite', 'n2')
    s.respond('p000001', True)
    s.world.tick = 3
    s.meet(s.world.plans[0])
    return s


def test_personality_nested_traits_cannot_bypass_validation():
    scenario = completed().world.scenario
    t = scenario.people[0].personality.traits
    for key in ('initiative', 'unknown'):
        with pytest.raises((ValueError, TypeError, AttributeError)):
            t[key] = 999
    assert Scenario.model_validate_json(scenario.model_dump_json()) == scenario


@pytest.mark.parametrize('change', ['future', 'mismatch'])
def test_completed_plan_rejects_future_or_mismatched_meeting_tick(change):
    data = completed().world.model_dump()
    WorldState.model_validate(data)
    if change == 'future':
        data['plans'][0]['tick'] = 99
    else:
        data['events'][-1]['tick'] = 2
    with pytest.raises(ValueError):
        WorldState.model_validate(data)


@pytest.mark.parametrize('change', ['participants', 'transition', 'cause'])
def test_plan_history_rejects_wrong_participants_or_transition(change):
    data = completed().world.model_dump()
    if change == 'participants':
        data['events'][1]['participants'] = ['n3', 'n1']
    elif change == 'transition':
        data['plans'][0]['history'].insert(1, data['plans'][0]['history'][0])
    else:
        data['events'][2]['causes'] = []
    with pytest.raises(ValueError):
        WorldState.model_validate(data)


def test_relationship_change_is_measured_from_initial_state():
    s = completed()
    initial = next(r for r in s.world.scenario.relationships if r.source == 'n1' and r.target == 'n2')
    assert s.world.relationships['n1:n2'].closeness > initial.closeness
    assert s.world.relationships['n1:n2'].shared_meetings == 1


def test_invalid_action_preserves_world_and_rng():
    s = Simulation(load_scenario(Path('data/fixtures/school.yaml')))
    s.world.tick = 2
    before, rng = s.world.model_dump_json(), s.rng.getstate()
    with pytest.raises(ValueError):
        s.act('n1', 'invite', 'ghost')
    assert (s.world.model_dump_json(), s.rng.getstate()) == (before, rng)
    s.act('n1', 'invite', 'n2')
    s.respond('p000001', True)
    s.act('n3', 'invite', 'n2')
    before, rng = s.world.model_dump_json(), s.rng.getstate()
    for attempt in [lambda: s.act('n1', 'rest'), lambda: s.respond('p000002', True)]:
        with pytest.raises(ValueError):
            attempt()
        assert (s.world.model_dump_json(), s.rng.getstate()) == (before, rng)


@pytest.mark.parametrize('action,target,event', [('invite','n2','missing'),('rest','n1',None),('ask','n2','missing')])
def test_rejected_unused_action_parameters_preserve_world_and_rng(action,target,event):
    s = Simulation(load_scenario(Path('data/fixtures/school.yaml')))
    s.world.tick = 2
    before, rng = s.world.model_dump_json(), s.rng.getstate()
    with pytest.raises(ValueError):
        s.act('n1',action,target,event)
    assert (s.world.model_dump_json(),s.rng.getstate()) == (before,rng)
