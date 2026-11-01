from random import Random

from cognitive_helpers import episode, school
from social_sim.decision import decide
from social_sim.memory import relevant_memories, retain_memories
from social_sim.models import Memory


def memory(index=0, significance=80, target='n1', topic='exclusion'):
    return Memory(id=f'm{index}', event_id=f'e{index}', target=target, significance=significance,
                  valence='negative', tick=index, topic=topic, source='n1', confidence=0.8)


def test_relevant_memory_changes_appraisal_or_candidate_score():
    s = school()
    episode(s)
    view = s.view('n3')
    a = decide(view, Random(1), s.world.scenario.weights, 'preview')
    view.state.memories.clear()
    b = decide(view, Random(1), s.world.scenario.weights, 'preview')
    assert [c.utility for c in a.candidates] != [c.utility for c in b.candidates]
    assert a.memories


def test_irrelevant_memory_does_not_change_unrelated_decision():
    s = school()
    episode(s)
    view = s.view('n3')
    a = decide(view, Random(1), s.world.scenario.weights, 'preview')
    view.state.memories.append(memory(99,target='n2',topic='homework'))
    b = decide(view, Random(1), s.world.scenario.weights, 'preview')
    assert a.candidates == b.candidates


def test_repeated_distinct_experiences_differ_from_one_experience():
    s = school()
    e = episode(s)
    once = s.world.states['n3'].interpretations[-1].confidence
    s.share_information('n1','n3',e.id)
    assert s.world.states['n3'].interpretations[-1].confidence == once
    episode(s, tick=10)
    assert s.world.states['n3'].interpretations[-1].confidence > once


def test_significant_memory_survives_routine_events():
    major = memory(0,100)
    routine = [memory(i,10) for i in range(1,40)]
    retained = retain_memories([major,*routine],40)
    assert major in retained and len(retained) == 24


def test_memory_selection_is_bounded_and_deterministic():
    s = school()
    s.world.states['n3'].memories = [memory(i) for i in range(30)]
    a = relevant_memories(s.world.states['n3'],'n1','exclusion',40)
    assert len(a) <= 5
    assert a == relevant_memories(s.world.states['n3'],'n1','exclusion',40)
