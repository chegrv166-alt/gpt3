"""Синтетические контрасты. Ответы на поддержку/извинение выбирает политика NPC."""
import argparse
import json
from pathlib import Path
from random import Random

from scripts.inspect_cognitive_examples import scenario, episode
from social_sim.authoring import load_scenario
from social_sim.decision import decide
from social_sim.diagnostics import explain, metrics
from social_sim.engine import Simulation
from social_sim.models import Scenario
from social_sim.storage import export_run


def build(name: str) -> Simulation:
    data=(load_scenario(Path('data/fixtures/school.yaml')) if name=='third_support' else scenario()).model_dump()
    data['description']='Синтетический контролируемый контраст восстановления; не канон.'
    for person in data['people']:
        traits=person['personality']['traits']
        if person['id']=='n3':
            if name=='rejected':traits.update(openness=0,reconciliation_readiness=0,autonomy=5,grudge_persistence=5)
            elif name=='deferred':traits.update(openness=0,reconciliation_readiness=0,autonomy=0,grudge_persistence=0,stability=5)
            elif name=='third_support':traits.update(openness=5,reconciliation_readiness=5)
        if person['id']=='n1' and name in ('weak','deep'):
            traits.update(directness=5,honesty_priority=5,tact=0,reconciliation_readiness=0)
    if name in ('weak','deep'):
        for rel in data['relationships']:
            if rel['source']=='n3' and rel['target']=='n1':
                rel.update(closeness=30 if name=='weak' else 90, trust=35 if name=='weak' else 90,liking=40 if name=='weak' else 70)
    return Simulation(Scenario.model_validate(data),42)


def main() -> None:
    parser=argparse.ArgumentParser(description='Контрасты поддержки, недовольства и извинения')
    parser.add_argument('--output',type=Path,default=Path('output/repair_examples'))
    directory=parser.parse_args().output
    if directory.exists():raise ValueError('Каталог уже существует: выберите новый --output')
    rows=[]
    for name in ('accepted','rejected','deferred','third_support','weak','deep'):
        sim=build(name)
        root=episode(sim,2)
        before=decide(sim.view('n3'),Random(7),sim.world.scenario.weights,'preview')
        if name=='third_support':
            report=sim.interact('n3','support_request','n4',root)
            assert root not in sim.world.states['n4'].knowledge
            attempt=sim.interact('n4','support','n3',root)
        else:
            report=sim.interact('n3','discontent','n1',root,defer_reply=True)
            if name in ('weak','deep'):
                sim.respond_interaction(report.id,'n1')
                attempt=report
            else:
                sim.respond_interaction(report.id,'n1','acknowledge')
                sim.world.tick=7
                attempt=sim.interact('n1','apology','n3',root)
        after=decide(sim.view('n3'),Random(7),sim.world.scenario.weights,'preview')
        export_run(sim,directory/name)
        effects=[e.model_dump() for e in sim.world.effects]
        report_data={'name':name,'root':root,'report':report.model_dump(),'attempt':attempt.model_dump(),
                     'before':before.model_dump(),'after':after.model_dump(),'effects':effects,
                     'explanation':explain(sim.world,attempt.offered_event),'metrics':metrics(sim.world)}
        (directory/name/'explanation.json').write_text(json.dumps(report_data,ensure_ascii=False,indent=2),encoding='utf-8')
        rows.append(report_data)
        response=next(e for e in sim.world.events if e.id==attempt.response_event) if attempt.response_event else None
        print(name,attempt.status,response.social_response if response else 'pending',
              'hurt',sim.world.states['n3'].hurt,'trust',sim.world.relationships['n3:n1'].trust,
              'effects',[(e['actor'],e['target'],e['trust_delta'],e['emotion_delta']) for e in effects])
    (directory/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
