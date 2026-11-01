from pathlib import Path

from social_sim.authoring import load_scenario
from social_sim.engine import Simulation
from social_sim.models import Scenario


def school(*, weak=False, calm=False, explanation=False):
    data = load_scenario(Path('data/fixtures/school.yaml')).model_dump()
    data['people'] = [p for p in data['people'] if p['id'] in ('n1','n2','n3')]
    for p in data['people']:
        p['is_player'] = p['id'] == 'n2'
        if p['id'] == 'n3':
            p['personality']['traits']['directness'] = 5
            p['personality']['traits']['reconciliation_readiness'] = 5
            p['personality']['strategies'] = ['ask']
            if calm:
                for key in ('rejection_sensitivity','inclusion_expectation','suspiciousness'):
                    p['personality']['traits'][key] = 0
    data['relationships'] = [r for r in data['relationships'] if r['source'] in ('n1','n2','n3') and r['target'] in ('n1','n2','n3')]
    if weak:
        for r in data['relationships']:
            if r['source'] == 'n3':
                r['closeness'] = r['liking'] = r['trust'] = 0
    if explanation:
        data['meeting_context'] = 'no_prior_group_plan'
    return Simulation(Scenario.model_validate(data))


def episode(s, tick=2, disclose=True):
    s.world.tick = tick
    s.act('n1', 'invite', 'n2')
    plan = s.world.plans[-1]
    s.respond(plan.id, True)
    s.world.tick = tick+1
    s.meet(plan)
    event = s.world.events[-1]
    s.world.tick = tick+4
    if disclose:
        s.share_information('n1', 'n3', event.id)
    return event
