from dataclasses import dataclass
from random import Random

from social_sim.models import Candidate, DecisionTrace, NPCState, Person, Relationship, Weights


@dataclass(frozen=True)
class ActorView:
    person: Person
    state: NPCState
    relationships: tuple[Relationship, ...]
    tick: int
    can_plan: bool


def choose(actor: ActorView, candidates: list[Candidate], constraints: list[str], rng: Random, weights: Weights, trace_id: str, randomize: bool = True) -> DecisionTrace:
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
        id=trace_id, actor=actor.person.id, tick=actor.tick,
        known_events=tuple(sorted(actor.state.knowledge)),
        interpretation_events=tuple(i.event_id for i in actor.state.interpretations),
        goals=actor.person.personality.goals, candidates=tuple(candidates), constraints=tuple(constraints),
        selected=selected, random_draw=draw, selection_pool=pool,
        explanation=f'Выбрано {selected.action}; вклад факторов: {selected.components}. ' + ' '.join(constraints),
    )


def decide(view: ActorView, rng: Random, weights: Weights, trace_id: str) -> DecisionTrace:
    p, state = view.person.personality, view.state
    t = p.traits
    autonomy_goal = sum(g.priority for g in p.goals if g.kind == 'preserve_autonomy')
    candidates = [Candidate(action='rest', components={'recovery': (100-state.energy)/10 + t['autonomy']/2, 'goal': autonomy_goal * 4 * weights.goal})]
    constraints = []
    connect_goal = sum(g.priority for g in p.goals if g.kind == 'connect')
    for rel in view.relationships:
        target = rel.target
        hurt = state.hurt.get(target, 0)
        avoiding = state.avoiding_until.get(target, 0) > view.tick
        if view.can_plan and state.energy >= 20 and not avoiding and view.tick - state.cooldowns.get(f'invite:{target}', -100) >= 8:
            candidates.append(Candidate(action='invite', target=target, components={
                'need': state.loneliness / 10 * weights.need,
                'personality': (t['initiative'] + t['sociability'] - t['autonomy']) / 2 * weights.personality,
                'goal': (connect_goal * 3 - autonomy_goal * 2) * weights.goal,
                'strategy': 4 if 'invite' in p.strategies else 0,
                'relationship': (rel.liking + rel.closeness - rel.tension) / 40 * weights.relationship,
                'emotion': -hurt / 8 * weights.emotion, 'effort': -3,
            }))
        else:
            constraints.append(f'invite:{target}: время, энергия, дистанция или cooldown')
        if hurt > 10 and state.energy >= 10:
            for action in ('ask', 'distance'):
                if view.tick - state.cooldowns.get(f'{action}:{target}', -100) < 8:
                    continue
                fit = (t['directness'] + t['reconciliation_readiness']) if action == 'ask' else (t['grudge_persistence'] + 5 - t['directness'])
                candidates.append(Candidate(action=action, target=target, components={
                    'emotion': hurt / 6 * weights.emotion,
                    'personality': fit / 2 * weights.personality,
                    'strategy': 4 if action in p.strategies else 0,
                    'effort': -2,
                }))
        if not avoiding and state.energy >= 10 and t['openness'] >= 2:
            facts = [k for k in state.knowledge.values() if k.kind == 'meeting_occurred' and view.person.id in k.participants and target not in k.participants and view.tick-k.acquired_tick <= 8 and f'tell:{target}:{k.event_id}' not in state.cooldowns]
            if facts:
                k = facts[-1]
                candidates.append(Candidate(action='tell', target=target, event_id=k.event_id, components={
                    'personality': (t['openness'] + t['sociability']) / 2,
                    'connection': rel.trust / 25, 'effort': -2,
                }))
    return choose(view, candidates, constraints, rng, weights, trace_id)
