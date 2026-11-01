from random import Random
from test_social_interactions import ready
from social_sim.decision import decide


def test_discontent_is_selected_from_alternatives_not_scripted():
    s=ready();trace=decide(s.view('n3'),Random(3),s.world.scenario.weights,'preview')
    assert any(c.action=='discontent' for c in trace.candidates)
    assert any(c.action in ('ask','distance','invite','rest') for c in trace.candidates)
    assert any(decide(s.view('n3'),Random(seed),s.world.scenario.weights,'preview').selected.action=='discontent' for seed in range(30))


def disagreement(s):
    root=s.world.states['n3'].interpretations[-1].event_id
    x=s.interact('n3','discontent','n1',root,defer_reply=True)
    s.respond_interaction(x.id,'n1','disagree')
    return x


def test_recipient_can_disagree_without_establishing_guilt():
    s=ready();confidence=s.world.states['n3'].interpretations[-1].confidence
    disagreement(s)
    assert s.world.states['n3'].interpretations[-1].confidence==confidence
    assert s.world.events[-1].social_response=='disagree'


def test_conflict_can_arise_from_incompatible_interpretations():
    s=ready();x=disagreement(s)
    assert x.status=='completed'
    assert s.world.effects[-1].reason.startswith('Несовпадение')
    assert not s.world.states['n1'].hurt


def test_conflict_does_not_symmetrize_relationship_changes():
    s=ready();before=s.world.relationships['n1:n3'].model_copy(deep=True)
    disagreement(s)
    assert s.world.relationships['n1:n3']==before


def test_minor_disagreement_preserves_deep_relationship_inertia():
    s=ready();r=s.world.relationships['n3:n1'];r.closeness=90;r.trust=90
    disagreement(s)
    assert r.closeness==90 and r.trust>85 and r.tension<15


def test_uninvolved_npc_does_not_know_private_conflict():
    s=ready();x=disagreement(s)
    assert x.offered_event not in s.view('n2').state.knowledge
