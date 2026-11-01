from dataclasses import dataclass
from random import Random
from itertools import combinations

from social_sim.memory import UTILITY_GAIN, memory_signal, relevant_memories
from social_sim.models import Candidate, DecisionTrace, NPCState, Person, Relationship, Weights, SituationAtDecision


@dataclass(frozen=True)
class ActorView:
    person: Person
    state: NPCState
    relationships: tuple[Relationship, ...]
    tick: int
    can_plan: bool
    event_boundary: int


def choose(actor: ActorView, candidates: list[Candidate], constraints: list[str], rng: Random, weights: Weights, trace_id: str, randomize: bool = True) -> DecisionTrace:
    aliases = {'strategy': 'personality', 'autonomy': 'personality', 'privacy': 'personality',
               'openness': 'personality', 'repair': 'personality', 'connection': 'relationship',
               'fatigue': 'need', 'recovery': 'need', 'memory': 'emotion'}
    candidates = [c.model_copy(update={'components': {
        key: value * float(getattr(weights, aliases.get(key, key.split('.')[0]), 1))
        for key, value in c.components.items()
    }}) for c in candidates]
    best = max(c.utility for c in candidates)
    pool = tuple(i for i, c in enumerate(candidates) if c.utility >= best - weights.selection_band)
    draw = rng.random() if randomize else None
    masses = [max(0.01, candidates[i].utility - best + weights.selection_band) for i in pool]
    cursor = (draw or 0) * sum(masses)
    selected = candidates[pool[-1]]
    for i, mass in zip(pool, masses):
        cursor -= mass
        if cursor < 0:
            selected = candidates[i]
            break
    return DecisionTrace(
        id=trace_id, actor=actor.person.id, tick=actor.tick, event_boundary=actor.event_boundary,
        known_events=tuple(sorted(actor.state.knowledge)),
        interpretation_events=tuple(i.event_id for i in actor.state.interpretations),
        interpretation_ids=tuple(s.current_revision for s in actor.state.situations.values()),
        memories=tuple(m for m in actor.state.memories if any(m.id in c.memory_ids for c in candidates)),
        situations=tuple(SituationAtDecision(id=s.id, revision_id=s.current_revision, emotion=s.emotion,
                         status=s.status, generation=s.generation, distance_active=s.distance_active)
                         for s in actor.state.situations.values()),
        goals=actor.person.personality.goals, candidates=tuple(candidates), constraints=tuple(constraints),
        selected=selected, random_draw=draw, selection_pool=pool,
        explanation=f'Выбрано {selected.action}; вклад факторов: {selected.components}. ' + ' '.join(constraints),
    )


def decide(view: ActorView, rng: Random, weights: Weights, trace_id: str) -> DecisionTrace:
    p, state = view.person.personality, view.state
    t = p.traits
    autonomy_goal = sum(g.priority for g in p.goals if g.kind == 'preserve_autonomy')
    candidates = [Candidate(action='rest', components={'recovery': (100-state.energy)/10, 'personality': t['autonomy']/2, 'goal': autonomy_goal * 4})]
    constraints = []
    connect_goal = sum(g.priority for g in p.goals if g.kind == 'connect')
    for rel in view.relationships:
        target = rel.target
        hurt = state.hurt.get(target, 0)
        avoiding = state.avoiding(target)
        memories = relevant_memories(state, target, 'exclusion', view.tick)
        signal = memory_signal(memories) * UTILITY_GAIN
        memory_ids = tuple(m.id for m in memories)
        situation = state.actionable_situation(target)
        if view.can_plan and state.energy >= 20 and not avoiding and view.tick - state.cooldowns.get(f'invite:{target}', -100) >= 8:
            candidates.append(Candidate(action='invite', target=target, memory_ids=memory_ids, components={
                'memory': -signal,
                'need': state.loneliness / 10,
                'personality': (t['initiative'] + t['sociability'] - t['autonomy']) / 2,
                'goal': (connect_goal * 3 - autonomy_goal * 2),
                'strategy': 4 if 'invite' in p.strategies else 0,
                'relationship': (rel.liking + rel.closeness - rel.tension) / 40,
                'emotion': -hurt / 8, 'effort': -3,
            }))
        else:
            constraints.append(f'invite:{target}: время, энергия, дистанция или cooldown')
        if situation is not None and state.energy >= 10:
            for action in ('ask', 'distance'):
                if view.tick - state.cooldowns.get(f'{action}:{target}', -100) < 8:
                    constraints.append(f'{action}:{target}: cooldown')
                    continue
                fit = (t['directness'] + t['reconciliation_readiness']) if action == 'ask' else (t['grudge_persistence'] + 5 - t['directness'])
                candidates.append(Candidate(action=action, target=target, situation_id=situation.id, memory_ids=memory_ids, components={
                    'memory': signal if action == 'distance' else signal/2,
                    'emotion': hurt / 6,
                    'personality': fit / 2,
                    'strategy': 4 if action in p.strategies else 0,
                    'effort': -2,
                }))
        elif hurt > 10:
            constraints.append(f'reaction:{target}: нет нового эпизода или энергии')
        if not avoiding and state.energy >= 10 and t['openness'] >= 2:
            facts = [k for k in state.knowledge.values() if k.kind == 'meeting_occurred' and view.person.id in k.participants and target not in k.participants and view.tick-k.acquired_tick <= 8 and f'tell:{target}:{k.event_id}' not in state.cooldowns]
            if facts:
                k = facts[-1]
                candidates.append(Candidate(action='tell', target=target, event_id=k.event_id, components={
                    'personality': (t['openness'] + t['sociability']) / 2,
                    'connection': rel.trust / 25, 'effort': -2,
                }))
            else:
                constraints.append(f'tell:{target}: нет непереданного наблюдённого события')
        else:
            constraints.append(f'tell:{target}: дистанция, энергия или порог открытости')
    contacts=sorted((r for r in view.relationships if not state.avoiding(r.target)),key=lambda r:r.target)[:6]
    if state.energy>=20:
        for pair in combinations(contacts,2):
            targets=tuple(r.target for r in pair)
            known_pair=any(k.kind in ('introduction_completed','introduction_already_known') and k.participants[0]==view.person.id
                           and set(k.participants[1:])==set(targets) for k in state.knowledge.values())
            key='introduce:'+':'.join(targets)
            if not known_pair and view.tick-state.cooldowns.get(key,-1000)>=32:
                candidates.append(Candidate(action='introduce',targets=targets,components={
                    'need':state.loneliness/15,'personality':t['initiative']+t['sociability']+t['novelty']-t['autonomy'],
                    'goal':connect_goal*2-autonomy_goal*8,'effort':-3}))
            groupkey='invite_group:'+':'.join(targets)
            if view.can_plan and view.tick-state.cooldowns.get(groupkey,-1000)>=8 and all(view.tick-state.cooldowns.get(f'invite:{p}',-1000)>=8 for p in targets):
                candidates.append(Candidate(action='invite_group',targets=targets,components={
                    'need':state.loneliness/10,'personality':t['sociability']*1.5-t['autonomy']/2,
                    'goal':connect_goal*3-autonomy_goal*4,'relationship':sum(r.liking+r.closeness-r.tension for r in pair)/80,
                    'emotion':-sum(state.hurt.get(r.target,0) for r in pair)/8,
                    'strategy':4 if 'invite' in p.strategies else 0,'effort':-4}))
    else:
        constraints.append('Групповая инициатива/знакомство: недостаточно энергии')
    return choose(view, candidates, constraints, rng, weights, trace_id)
