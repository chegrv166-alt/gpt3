from pathlib import Path
import pytest

from cognitive_helpers import school,episode
from social_sim.models import WorldState
from social_sim.storage import save_snapshot,load_snapshot


def ready():
    s=school(explanation=True);episode(s)
    return s


def test_interaction_has_specific_subject_and_causal_links():
    s=ready();root=s.world.states['n3'].interpretations[-1].event_id
    x=s.interact('n3','discontent','n1',root,defer_reply=True)
    assert x.message.subject_event==root and x.status=='pending'
    assert root in s.world.events[-1].causes
    assert s.world.events[-1].social_message.content=='hurt_report'
    WorldState.model_validate(s.world.model_dump())


def test_new_episode_preserves_pending_response_and_existing_intention(tmp_path):
    s=ready();root=s.world.states['n3'].interpretations[-1].event_id
    x=s.interact('n3','discontent','n1',root,defer_reply=True)
    # Explicitly retain distance while communication remains pending.
    s.world.tick=10;s.act('n3','distance','n1');episode(s,tick=18)
    assert x.status=='pending' and s.world.states['n3'].avoiding('n1')
    save_snapshot(s,tmp_path/'x.json');assert load_snapshot(tmp_path/'x.json').world==s.world


def test_response_to_previous_episode_has_bounded_scope():
    s=ready();root=s.world.states['n3'].interpretations[-1].event_id
    x=s.interact('n3','discontent','n1',root,defer_reply=True)
    later=episode(s,tick=10)
    before=next(iter(s.world.states['n3'].situations.values())).episodes[later.id].emotion
    s.respond_interaction(x.id,'n1','acknowledge')
    assert next(iter(s.world.states['n3'].situations.values())).episodes[later.id].emotion==before


def test_duplicate_delivery_does_not_duplicate_interaction_effect():
    from social_sim.cognition import perceive
    s=ready();root=s.world.states['n3'].interpretations[-1].event_id
    x=s.interact('n3','discontent','n1',root,defer_reply=True)
    event=next(e for e in s.world.events if e.id==x.offered_event)
    before=s.world.model_dump_json()
    perceive(s.world,'n1',event,'n3',event.id)
    assert s.world.model_dump_json()==before


@pytest.mark.parametrize('kind,target,subject',[('discontent','ghost','missing'),('apology','n3','missing'),('support','n3','missing')])
def test_invalid_social_action_preserves_world_and_rng(kind,target,subject):
    s=ready();before,rng=s.world.model_dump_json(),s.rng.getstate()
    with pytest.raises(ValueError):s.interact('n1',kind,target,subject)
    assert (s.world.model_dump_json(),s.rng.getstate())==(before,rng)
