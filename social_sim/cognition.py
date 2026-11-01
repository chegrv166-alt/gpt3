from social_sim.models import Interpretation, KnowledgeRecord, Memory, WorldEvent, WorldState


def bounded(value: float) -> float:
    return round(max(0, min(100, value)), 6)


def perceive(world: WorldState, actor: str, event: WorldEvent, source: str, via: str, message: bool = False) -> None:
    state = world.states[actor]
    if event.id in state.knowledge:
        return
    confidence = 0.8 if message else 1.0
    state.knowledge[event.id] = KnowledgeRecord(
        event_id=event.id, kind=event.kind, participants=event.participants,
        source=source, acquired_tick=world.tick, method='message' if message else 'observation',
        confidence=confidence, via_event=via,
    )
    if event.kind != 'meeting_occurred' or actor in event.participants:
        return
    # Missing an arbitrary stranger's meeting is not a violated expectation.
    if not all(f'{actor}:{p}' in world.relationships for p in event.participants):
        return
    target = event.participants[0]
    rel = world.relationships[f'{actor}:{target}']
    person = next(p for p in world.scenario.people if p.id == actor)
    traits = person.personality.traits
    beliefs = person.personality.beliefs
    strength = max((b.strength for b in beliefs if b.condition == 'friends_met_without_me'), default=0)
    expectation = (traits['inclusion_expectation'] + traits['rejection_sensitivity'] + traits['suspiciousness']) / 15
    concern = confidence * strength * expectation * (0.5 + rel.closeness / 200)
    excluded = concern >= 0.3
    state.interpretations.append(Interpretation(
        event_id=event.id, kind='excluded' if excluded else 'neutral', target=target,
        confidence=round(concern, 6), belief_ids=tuple(b.id for b in beliefs),
        reason='Встреча знакомых без меня допускает исключение; намерение неизвестно.' if excluded else 'Отдельная встреча не доказывает отвержение.',
    ))
    if excluded:
        impact = concern * 20 / (1 + rel.closeness / 50)
        state.hurt[target] = bounded(state.hurt.get(target, 0) + concern * 55)
        rel.tension = bounded(rel.tension + impact)
        rel.trust = bounded(rel.trust - impact / 2)
        state.memories.append(Memory(event_id=event.id, target=target, significance=concern * 100, valence='negative'))
        state.memories[:] = state.memories[-24:]
