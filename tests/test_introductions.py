from test_group_plans import sim
import pytest
from social_sim.models import WorldState


def test_mediator_introduces_known_people_after_their_decisions():
    s=sim();event=s.introduce('n1','n2','n5',answers={'n2':True,'n5':True})
    assert event.kind=='introduction_completed'
    assert s.world.relationships['n2:n5'].closeness < 20
    assert s.world.relationships['n5:n2'].closeness < 20
    assert len([t for t in s.world.traces if t.selected.action=='accept_introduction'])==2
    WorldState.model_validate(s.world.model_dump())


def test_introduction_can_be_declined_without_forced_friendship():
    s=sim();e=s.introduce('n1','n2','n5',answers={'n2':False,'n5':True})
    assert e.kind=='introduction_declined' and 'n2:n5' not in s.world.relationships


def test_introduction_requires_legitimate_social_opportunity():
    s=sim();before,rng=s.world.model_dump_json(),s.rng.getstate()
    with pytest.raises(ValueError):s.introduce('n1','n2','n8')
    assert (s.world.model_dump_json(),s.rng.getstate())==(before,rng)


def test_hearing_a_name_does_not_create_acquaintance():
    s=sim();s.emit('information_shared',('n1','n2','n5'),'Имена в сообщении')
    assert 'n2:n5' not in s.world.relationships


def test_repeated_introduction_has_no_duplicate_effect():
    s=sim();s.introduce('n1','n2','n5',answers={'n2':True,'n5':True})
    before=s.world.model_dump_json()
    with pytest.raises(ValueError):s.introduce('n1','n2','n5')
    assert s.world.model_dump_json()==before
    s.world.tick=70
    old=s.world.relationships['n2:n5'].model_copy(deep=True)
    s.introduce('n1','n2','n5')
    assert s.world.relationships['n2:n5']==old


def test_noop_player_does_not_accept_introduction_automatically():
    s=sim();s.introduce('n1','player','n5')
    assert 'player:n5' not in s.world.relationships
    assert not any(t.actor=='player' for t in s.world.traces)


def test_autonomous_introduction_appears_in_controlled_fixture():
    s=sim();s.run_days(10)
    assert any(e.kind=='introduction_completed' and not e.outcome.startswith('Внешн') for e in s.world.events)


@pytest.mark.parametrize('damage',['missing_consent','wrong_participant','wrong_outcome'])
def test_corrupt_introduction_history_is_rejected(damage):
    s=sim();s.introduce('n1','n2','n5',answers={'n2':True,'n5':True})
    raw=s.world.model_dump()
    end=raw['events'][-1]
    if damage=='missing_consent':end['causes']=end['causes'][:2]
    elif damage=='wrong_participant':raw['events'][1]['participants']=['n3','n1']
    else:raw['events'][1]['kind']='introduction_rejected'
    with pytest.raises(ValueError):WorldState.model_validate(raw)


def test_social_world_fixture_has_autonomous_introductions_and_group_meetings():
    from pathlib import Path
    from social_sim.authoring import load_scenario
    from social_sim.engine import Simulation
    from social_sim.groups import detect_groups
    s=Simulation(load_scenario(Path('data/fixtures/social_world.yaml')),42)
    assert not detect_groups(s.world)
    s.run_days(30)
    assert any(e.kind=='introduction_completed' for e in s.world.events)
    assert any(e.kind=='meeting_occurred' and len(e.participants)>2 for e in s.world.events)
    assert any(r.origin_event for r in s.world.relationships.values())
    WorldState.model_validate(s.world.model_dump())
