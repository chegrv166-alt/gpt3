import pytest
from pathlib import Path
from cognitive_helpers import episode
from social_sim.authoring import load_scenario
from social_sim.engine import Simulation


def third_ready():
    s=Simulation(load_scenario(Path('data/fixtures/school.yaml')));episode(s)
    root=s.world.states['n3'].interpretations[-1].event_id
    return s,root


def request(s,root):
    return s.interact('n3','support_request','n4',root,defer_reply=True)


def test_support_requires_known_reason_not_hidden_hurt():
    s,root=third_ready()
    with pytest.raises(ValueError):s.interact('n4','support','n3',root)
    x=request(s,root)
    assert root not in s.world.states['n4'].knowledge
    assert x.offered_event in s.world.states['n4'].knowledge
    s.interact('n4','support','n3',root)


def test_support_can_be_declined_without_automatic_penalty():
    s,root=third_ready();request(s,root)
    x=s.interact('n4','support','n3',root,defer_reply=True)
    before=s.world.relationships['n3:n4'].model_copy(deep=True)
    hurt=s.world.states['n3'].hurt.copy()
    s.respond_interaction(x.id,'n3','decline')
    assert s.world.relationships['n3:n4']==before and s.world.states['n3'].hurt==hurt


def test_accepted_support_has_bounded_directed_effect():
    s,root=third_ready();request(s,root)
    x=s.interact('n4','support','n3',root,defer_reply=True)
    old=s.world.relationships['n3:n4'].trust
    reverse=s.world.relationships['n4:n3'].model_copy(deep=True)
    s.respond_interaction(x.id,'n3','accept')
    assert old < s.world.relationships['n3:n4'].trust <= old+2
    assert s.world.relationships['n4:n3']==reverse
    assert s.world.effects[-1].emotion_delta<0


def test_support_by_third_person_does_not_restore_trust_in_other_person():
    s,root=third_ready();request(s,root)
    old=s.world.relationships['n3:n1'].model_copy(deep=True)
    x=s.interact('n4','support','n3',root,defer_reply=True);s.respond_interaction(x.id,'n3','accept')
    assert s.world.relationships['n3:n1']==old


def test_repeated_support_cannot_farm_relationship_gain():
    s,root=third_ready();request(s,root)
    x=s.interact('n4','support','n3',root,defer_reply=True);s.respond_interaction(x.id,'n3','accept')
    s.world.tick+=8
    before,rng=s.world.model_dump_json(),s.rng.getstate()
    with pytest.raises(ValueError):s.interact('n4','support','n3',root)
    assert (s.world.model_dump_json(),s.rng.getstate())==(before,rng)


def test_support_response_is_a_separate_explainable_decision():
    s,root=third_ready();request(s,root)
    x=s.interact('n4','support','n3',root)
    event=next(e for e in s.world.events if e.id==x.response_event)
    t=next(t for t in s.world.traces if t.id==event.trace_id)
    assert t.actor=='n3' and x.offered_event in t.known_events
    assert {'accept','decline','defer'}<={c.action for c in t.candidates}


@pytest.mark.parametrize('emotion',[33.333333,7.000001,99.999999])
def test_bounded_support_effect_rounding_preserves_valid_world(emotion):
    from social_sim.models import WorldState
    s,root=third_ready();request(s,root)
    x=s.interact('n4','support','n3',root,defer_reply=True)
    situation=next(iter(s.world.states['n3'].situations.values()))
    situation.episodes[root].emotion=emotion
    WorldState.model_validate(s.world.model_dump())
    s.respond_interaction(x.id,'n3','accept')
    WorldState.model_validate(s.world.model_dump())
    assert -6<=s.world.effects[-1].emotion_delta<=0
