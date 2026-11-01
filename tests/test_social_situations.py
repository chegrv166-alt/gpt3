from random import Random

import pytest

from cognitive_helpers import episode, school
from social_sim.cognition import perceive
from social_sim.decision import decide


def test_weak_acquaintance_does_not_imply_expected_invitation():
    s = school(weak=True)
    episode(s)
    assert not s.world.states['n3'].hurt
    assert not s.world.states['n3'].situations


def test_participant_order_preserves_interpretation_when_roles_unchanged():
    a, b = school(), school()
    ea, eb = episode(a, disclose=False), episode(b, disclose=False)
    eb = eb.model_copy(update={'participants': tuple(reversed(eb.participants))})
    perceive(a.world, 'n3', ea, 'n1', ea.id, True)
    perceive(b.world, 'n3', eb, 'n1', eb.id, True)
    assert a.world.states['n3'].hurt == b.world.states['n3'].hurt
    assert a.world.states['n3'].interpretations == b.world.states['n3'].interpretations


def test_same_fact_different_profiles_produce_different_appraisals():
    a, b = school(), school(calm=True)
    episode(a)
    episode(b)
    assert a.world.states['n3'].interpretations[-1].confidence > b.world.states['n3'].interpretations[-1].confidence


def test_evidence_delivery_is_idempotent():
    s = school()
    e = episode(s)
    before = s.world.model_dump_json()
    s.share_information('n1','n3',e.id)
    assert s.world.model_dump_json() == before


def test_bare_confirmation_does_not_prove_absence_of_exclusion():
    s = school()
    episode(s)
    state = s.world.states['n3']
    before = (state.hurt.copy(), state.interpretations[-1].confidence)
    s.act('n3','ask','n1')
    assert (state.hurt, state.interpretations[-1].confidence) == before
    assert s.world.events[-1].claim == 'meeting_confirmed'


def test_relevant_explanation_can_revise_suspicion_without_forcing_forgiveness():
    s = school(explanation=True)
    episode(s)
    state = s.world.states['n3']
    old = state.interpretations[-1]
    trust = s.world.relationships['n3:n1'].trust
    s.act('n3','ask','n1')
    assert state.interpretations[-1].confidence < old.confidence
    assert state.interpretations[0] == old
    assert 0 < state.hurt['n1'] < old.emotion_after
    assert trust <= s.world.relationships['n3:n1'].trust < 100
    assert s.world.events[-1].claim == 'no_prior_group_plan'


def test_interpretation_revision_preserves_previous_trace():
    s = school(explanation=True)
    episode(s)
    trace = decide(s.view('n3'), Random(1), s.world.scenario.weights, 'preview')
    frozen = trace.model_dump_json()
    s.act('n3','ask','n1')
    assert trace.model_dump_json() == frozen
    assert s.world.states['n3'].interpretations[-1].id not in trace.interpretation_ids


def test_uninformed_respondent_does_not_invent_explanation():
    s = school()
    episode(s)
    # Participant knows the meeting but has no authored circumstance to disclose.
    s.act('n3','ask','n1')
    assert not any(e.claim == 'no_prior_group_plan' for e in s.world.events)


def test_declined_question_does_not_automatically_prove_guilt_or_relieve_hurt():
    s = school(explanation=True)
    episode(s)
    s.world.states['n1'].energy = 0
    before = s.world.states['n3'].hurt.copy()
    suspicion = s.world.states['n3'].interpretations[-1].confidence
    s.act('n3','ask','n1')
    assert s.world.events[-1].kind == 'question_declined'
    assert s.world.states['n3'].hurt == before
    assert s.world.states['n3'].interpretations[-1].confidence == suspicion


def test_cooldown_expiry_alone_does_not_repeat_reaction_to_same_situation():
    s = school()
    episode(s)
    s.act('n3','distance','n1')
    s.world.tick += 8
    t = decide(s.view('n3'), Random(2), s.world.scenario.weights, 'preview')
    assert not any(c.action in ('ask','distance') and c.target == 'n1' for c in t.candidates)
    before, rng = s.world.model_dump_json(), s.rng.getstate()
    with pytest.raises(ValueError):
        s.act('n3','distance','n1')
    assert (s.world.model_dump_json(),s.rng.getstate()) == (before,rng)


def test_new_independent_evidence_can_reopen_or_escalate_situation():
    s = school()
    episode(s)
    s.act('n3','distance','n1')
    episode(s, tick=10)
    state = s.world.states['n3']
    assert len(state.situations) == 1
    assert len(state.interpretations) == 2
    t = decide(s.view('n3'), Random(2), s.world.scenario.weights, 'preview')
    assert any(c.action == 'ask' and c.target == 'n1' for c in t.candidates)


def test_noop_player_never_receives_automatic_answers_or_actions():
    s = school()
    s.run_days(10)
    assert not any(t.actor == 'n2' for t in s.world.traces)
    fresh = school()
    fresh.world.tick = 2
    fresh.act('n1','invite','n2')
    with pytest.raises(ValueError, match='Игрок'):
        fresh.respond(fresh.world.plans[-1].id)


def test_player_information_causes_specific_future_npc_decision():
    from social_sim.storage import load_snapshot, save_snapshot
    # The player is n2, a participant, and can disclose a fact others cannot yet know.
    base, changed = school(explanation=True), school(explanation=True)
    episode(base,disclose=False)
    fact = episode(changed,disclose=False)
    assert fact.id not in base.world.states['n3'].knowledge
    information = changed.share_information('n2','n3',fact.id)
    assert information is not None
    assert base.rng.getstate() == changed.rng.getstate()
    old = decide(base.view('n3'),Random(11),base.world.scenario.weights,'preview')
    new = decide(changed.view('n3'),Random(11),changed.world.scenario.weights,f't{len(changed.world.traces)+1:06d}')
    assert new.selected.action == 'ask' and new.selected.target == 'n1'
    assert not any(c.action == 'ask' for c in old.candidates)
    assert fact.id in new.known_events
    knowledge = changed.world.states['n3'].knowledge[fact.id]
    assert knowledge.source == 'n2' and knowledge.via_event == information.id
    assert any(m.event_id == fact.id for m in new.memories)
    question = changed.act('n3','ask','n1',trace=new)
    assert question.causes == (fact.id,)
    assert changed.world.events[-1].claim == 'no_prior_group_plan'


def test_end_to_end_autonomous_reaction_then_reappraisal_changes_next_scores():
    s = school(explanation=True)
    e = episode(s,disclose=False)
    assert e.id not in s.world.states['n3'].knowledge
    s.share_information('n1','n3',e.id)
    trace = decide(s.view('n3'),s.rng,s.world.scenario.weights,f't{len(s.world.traces)+1:06d}')
    state = s.world.states['n3']
    confidence = state.interpretations[-1].confidence
    old = next(c for c in trace.candidates if c.action == 'invite' and c.target == 'n1')
    s.act('n3',trace.selected.action,trace.selected.target,trace=trace)
    assert trace.selected.action == 'ask'
    assert state.interpretations[-1].confidence < confidence
    next_trace = decide(s.view('n3'),Random(0),s.world.scenario.weights,'preview')
    new = next(c for c in next_trace.candidates if c.action == 'invite' and c.target == 'n1')
    assert new.utility > old.utility


def test_unanswered_player_question_becomes_unresolved_without_relief():
    data = school().world.scenario.model_dump()
    for p in data['people']:
        p['is_player'] = p['id'] == 'n1'
    from social_sim.models import Scenario
    from social_sim.engine import Simulation
    s = Simulation(Scenario.model_validate(data))
    episode(s)
    before = s.world.states['n3'].hurt.copy()
    s.act('n3','ask','n1')
    situation = next(iter(s.world.states['n3'].situations.values()))
    assert situation.status == 'unresolved'
    assert s.world.states['n3'].hurt == before
    assert s.world.events[-1].kind == 'question_unanswered'
    assert s.world.states['n3'].interpretations[-1].event_id == s.world.events[-1].id
    assert not any(t.actor == 'n1' and t.selected.action == 'clarify' for t in s.world.traces)
