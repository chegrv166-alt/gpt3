from test_social_interactions import ready
from social_sim.cognition import perceive


def test_discontent_message_reports_feeling_without_proving_intent():
    s=ready();root=s.world.states['n3'].interpretations[-1].event_id
    fact=next(e for e in s.world.events if e.id==root).model_dump_json()
    x=s.interact('n3','discontent','n1',root,defer_reply=True)
    k=s.view('n1').state.knowledge[x.offered_event]
    assert k.social_message.content=='hurt_report' and k.source=='n3'
    assert k.social_message.about=='n1'
    assert next(e for e in s.world.events if e.id==root).model_dump_json()==fact
    assert next(e for e in s.world.events if e.id==x.offered_event).claim is None


def test_private_discontent_does_not_reach_group_automatically():
    s=ready();root=s.world.states['n3'].interpretations[-1].event_id
    x=s.interact('n3','discontent','n1',root)
    assert x.offered_event not in s.world.states['n2'].knowledge
    assert x.response_event not in s.world.states['n2'].knowledge


def test_recipient_decision_uses_only_received_information():
    s=ready();root=s.world.states['n3'].interpretations[-1].event_id
    assert not any(k.social_message for k in s.view('n1').state.knowledge.values())
    x=s.interact('n3','discontent','n1',root)
    t=next(t for t in s.world.traces if t.id==next(e for e in s.world.events if e.id==x.response_event).trace_id)
    assert x.offered_event in t.known_events
    assert not hasattr(s.view('n1'),'interactions')


def test_repeated_subjective_message_is_idempotent_for_same_delivery():
    s=ready();root=s.world.states['n3'].interpretations[-1].event_id
    x=s.interact('n3','discontent','n1',root,defer_reply=True)
    e=next(e for e in s.world.events if e.id==x.offered_event)
    before=s.world.model_dump_json();perceive(s.world,'n1',e,'n3',e.id)
    assert s.world.model_dump_json()==before
