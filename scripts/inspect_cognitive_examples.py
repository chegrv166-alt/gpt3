"""Контролируемые синтетические контрасты; не авторские сцены игры."""
from pathlib import Path
from random import Random
import json
import argparse

from social_sim.authoring import load_scenario
from social_sim.decision import decide
from social_sim.diagnostics import explain, metrics
from social_sim.engine import Simulation
from social_sim.models import Scenario
from social_sim.storage import export_run


def scenario(*, weak: bool = False, calm: bool = False, explanation: bool = False) -> Scenario:
    data = load_scenario(Path('data/fixtures/school.yaml')).model_dump()
    data['description'] = 'Синтетический контролируемый когнитивный контраст, не канон.'
    data['people'] = [p for p in data['people'] if p['id'] in ('n1', 'n2', 'n3')]
    for p in data['people']:
        p['is_player'] = p['id'] == 'n2'
        if p['id'] == 'n3':
            p['personality']['traits']['directness'] = 5
            p['personality']['traits']['reconciliation_readiness'] = 5
            p['personality']['strategies'] = ['ask']
            if calm:
                for key in ('inclusion_expectation', 'rejection_sensitivity', 'suspiciousness'):
                    p['personality']['traits'][key] = 0
    data['relationships'] = [r for r in data['relationships'] if r['source'] in ('n1','n2','n3') and r['target'] in ('n1','n2','n3')]
    if weak:
        for r in data['relationships']:
            if r['source'] == 'n3':
                r['closeness'] = r['liking'] = r['trust'] = 0
    data['meeting_context'] = 'no_prior_group_plan' if explanation else 'unknown'
    return Scenario.model_validate(data)


def episode(sim: Simulation, tick: int) -> str:
    sim.world.tick = tick
    sim.act('n1', 'invite', 'n2')
    plan = sim.world.plans[-1]
    sim.respond(plan.id, True)  # Синтетический внешний ответ игрока n2.
    sim.world.tick = plan.tick
    sim.meet(plan)
    fact = sim.world.events[-1]
    sim.world.tick = tick + 4
    assert fact.id not in sim.world.states['n3'].knowledge
    sim.share_information('n1', 'n3', fact.id)
    return fact.id


def main() -> None:
    parser = argparse.ArgumentParser(description='Синтетические когнитивные контрасты')
    parser.add_argument('--output',type=Path,default=Path('output/cognitive_examples'))
    directory = parser.parse_args().output
    if directory.exists():
        raise ValueError('Результаты уже существуют; выберите новый каталог в скрипте.')
    reports = []
    for name in ('confirmation', 'explanation', 'refusal', 'repeated', 'weak', 'calm'):
        sim = Simulation(scenario(weak=name=='weak', calm=name=='calm', explanation=name=='explanation'), 42)
        fact_id = episode(sim, 2)
        before = decide(sim.view('n3'), Random(11), sim.world.scenario.weights, 'preview')
        if name == 'refusal':
            sim.world.states['n1'].energy = 0
        trace = decide(sim.view('n3'), sim.rng, sim.world.scenario.weights, f't{len(sim.world.traces)+1:06d}')
        event = sim.execute_decision(trace)
        if name == 'repeated':
            episode(sim,10)
        export_run(sim, directory/name)
        after = decide(sim.view('n3'), Random(11), sim.world.scenario.weights, 'preview')
        report = {'name': name, 'fact_id': fact_id, 'reaction_id': event.id,
                  'before': before.model_dump(), 'after': after.model_dump(),
                  'explanation': explain(sim.world,event.id), 'metrics': metrics(sim.world)}
        reports.append(report)
        (directory/name/'explanation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        state=sim.world.states['n3']
        print(name, 'action', trace.selected.action, 'hurt',state.hurt,
              'confidences',[r.confidence for r in state.interpretations],
              'events',[e.kind for e in sim.world.events[-3:]])
    (directory/'summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')



if __name__ == '__main__':
    main()
