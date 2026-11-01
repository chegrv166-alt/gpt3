from random import Random

import pytest

from cognitive_helpers import episode, school
from social_sim.decision import choose, decide
from social_sim.diagnostics import explain, metrics
from social_sim.models import Candidate, Weights


@pytest.mark.parametrize('category,components', [
    ('personality', ['personality','strategy','privacy','openness','repair','autonomy']),
    ('relationship',['relationship','connection']),
    ('emotion',['emotion','memory']), ('need',['need','fatigue','recovery']), ('goal',['goal'])])
def test_zero_weight_removes_matching_utility_components(category, components):
    s = school()
    candidate = Candidate(action='rest',components={key:3.0 for key in components})
    weights = Weights.model_validate({category: 0.0})
    trace = choose(s.view('n3'),[candidate],[],Random(0),weights,'preview')
    assert all(value == 0 for value in trace.selected.components.values())


@pytest.mark.parametrize('policy', ['initiative','tell','invitation','question'])
def test_zero_weights_in_real_policies(policy):
    s = school(explanation=True)
    if policy == 'invitation':
        s.world.tick = 2
        s.act('n1','invite','n2')
        # n2 is player: explicit response retains the same scoring factors.
        data = s.world.scenario.model_dump()
        data['weights'] = dict(need=0.0,personality=0.0,emotion=0.0,relationship=0.0,goal=0.0)
        from social_sim.models import Scenario
        s.world.scenario = Scenario.model_validate(data)
        s.respond('p000001', False)
        traces = [s.world.traces[-1]]
    else:
        episode(s)
        data = s.world.scenario.model_dump()
        data['weights'] = dict(need=0.0,personality=0.0,emotion=0.0,relationship=0.0,goal=0.0)
        from social_sim.models import Scenario
        s.world.scenario = Scenario.model_validate(data)
        if policy == 'question':
            s.act('n3','ask','n1')
            traces = [s.world.traces[-1]]
        else:
            traces = [decide(s.view('n3' if policy=='initiative' else 'n2'),Random(0),s.world.scenario.weights,'preview')]
    for trace in traces:
        for candidate in trace.candidates:
            assert all(value == 0 for key,value in candidate.components.items() if key not in ('effort','external_input'))


def test_explain_links_memory_evidence_and_interpretation_revision():
    s = school(explanation=True)
    episode(s)
    trace = decide(s.view('n3'),s.rng,s.world.scenario.weights,f't{len(s.world.traces)+1:06d}')
    assert trace.selected.action == 'ask'
    question = s.act('n3',trace.selected.action,trace.selected.target,trace=trace)
    data = explain(s.world,question.id)
    assert data['decision']['memories']
    assert data['decision']['interpretation_ids']
    assert data['knowledge_at_decision']
    assert data['interpretations_at_decision'][0]['confidence'] > s.world.states['n3'].interpretations[-1].confidence
    # Current reply cannot retroactively become part of questioner's knowledge.
    assert s.world.events[-1].id not in [k['event_id'] for k in data['knowledge_at_decision']]
    assert data['state_updates']


def test_metrics_separate_initiative_response_and_repeated_causes():
    s = school()
    episode(s)
    s.act('n3','distance','n1')
    m = metrics(s.world)
    assert m['initiatives_by_action']['invite'] == 1
    assert m['responses_by_action']['accept'] == 1
    assert m['social_situations'] == 1
    assert m['situation_reactions'] == 1
    assert m['repeated_reactions_same_cause'] == 0
