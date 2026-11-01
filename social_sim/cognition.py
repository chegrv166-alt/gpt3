from social_sim.memory import APPRAISAL_GAIN, memory_signal, relevant_memories, retain_memories
from social_sim.models import Interpretation, KnowledgeRecord, Memory, SocialSituation, WorldEvent, WorldState


def bounded(value: float) -> float:
    return round(max(0, min(100, value)), 6)


def remember(world: WorldState, actor: str, event_id: str, target: str, topic: str, significance: float, negative: bool) -> None:
    state = world.states[actor]
    record = state.knowledge[event_id]
    memory = Memory(id=f'm:{actor}:{event_id}:{target}:{topic}', event_id=event_id, target=target,
                    significance=bounded(significance), valence='negative' if negative else 'positive',
                    tick=world.tick, topic=topic, source=record.source, confidence=record.confidence)
    state.memories = retain_memories([*state.memories, memory], world.tick)


def perceive(world: WorldState, actor: str, event: WorldEvent, source: str, via: str, message: bool = False) -> None:
    state = world.states[actor]
    if event.id in state.knowledge:
        return
    confidence = 0.8 if message else 1.0
    state.knowledge[event.id] = KnowledgeRecord(
        event_id=event.id, kind=event.kind, participants=event.participants,
        source=source, acquired_tick=world.tick, method='message' if message else 'observation',
        confidence=confidence, via_event=via, organizer=event.organizer,
        context=event.context if not message else 'unknown',
    )
    if event.kind in ('question_answered', 'question_declined', 'question_unanswered'):
        revise_from_answer(world, actor, event)
        return
    if event.kind != 'meeting_occurred' or actor in event.participants:
        return
    known = [world.relationships[f'{actor}:{p}'] for p in event.participants if f'{actor}:{p}' in world.relationships]
    eligible = [r for r in known if r.closeness >= 30 and r.liking >= 30 or relevant_memories(state, r.target, 'exclusion', world.tick)]
    if len(known) != len(event.participants) or not eligible:
        return
    # Explicit organizer, otherwise strongest expectation; never tuple position.
    rel = next((r for r in eligible if r.target == event.organizer), None)
    if rel is None:
        rel = sorted(eligible, key=lambda r: (-r.closeness, -r.liking, r.target))[0]
    target = rel.target
    person = next(p for p in world.scenario.people if p.id == actor)
    traits, beliefs = person.personality.traits, person.personality.beliefs
    memories = relevant_memories(state, target, 'exclusion', world.tick)
    memory_effect = APPRAISAL_GAIN * memory_signal(memories)
    strength = max((b.strength for b in beliefs), default=0)
    expectation = (traits['inclusion_expectation'] + traits['rejection_sensitivity'] + traits['suspiciousness']) / 15
    concern = round(max(0, min(0.95, confidence * strength * expectation * (0.5 + rel.closeness / 200) + memory_effect)), 6)
    excluded = concern >= 0.3
    sid = f's:{actor}:{target}:exclusion'
    previous = state.situations.get(sid)
    evidence = (*previous.evidence, event.id) if previous else (event.id,)
    emotion = bounded((previous.emotion if previous else 0) + (concern*55 if excluded else 0))
    impact = concern * 20 / (1 + rel.closeness / 50) if excluded else 0
    revision = Interpretation(
        id=f'i:{actor}:{len(state.interpretations)+1}', situation_id=sid, tick=world.tick,
        evidence=evidence, memory_ids=tuple(m.id for m in memories), used_memories=memories, memory_effect=memory_effect,
        emotion_after=emotion, trust_delta=-impact/2, tension_delta=impact,
        event_id=event.id, kind='excluded' if excluded else 'neutral', target=target,
        confidence=concern, belief_ids=tuple(b.id for b in beliefs),
        reason='Ожидание приглашения: близость или прошлый опыт; отдельная встреча допускает исключение, но не доказывает умысел.',
    )
    state.interpretations.append(revision)
    state.situations[sid] = SocialSituation(
        id=sid, observer=actor, target=target, related=tuple(sorted(event.participants)),
        expectation='Ожидание приглашения от близкого знакомого', evidence=evidence,
        current_revision=revision.id, updated_tick=world.tick, emotion=emotion,
        status='active' if excluded else 'completed', generation=previous.generation+1 if previous else 1,
        reacted_generation=previous.reacted_generation if previous else 0,
        reactions=previous.reactions[:] if previous else [], processed_claims=previous.processed_claims if previous else (),
    )
    rel.tension = bounded(rel.tension + impact)
    rel.trust = bounded(rel.trust - impact / 2)
    if excluded:
        remember(world, actor, event.id, target, 'exclusion', concern*100, True)


def revise_from_answer(world: WorldState, actor: str, event: WorldEvent) -> None:
    state = world.states[actor]
    situation = next((s for s in state.situations.values() if event.subject_event in s.evidence), None)
    if situation is None or event.participants[0] == actor:
        return
    old = next(i for i in state.interpretations if i.id == situation.current_revision)
    source = event.participants[0]
    rel = world.relationships.get(f'{actor}:{source}')
    trust = rel.trust / 100 if rel else 0
    claim_key = f'{event.subject_event}:{event.claim}'
    relevant = event.claim == 'no_prior_group_plan' and claim_key not in situation.processed_claims
    confidence = round(old.confidence * (1 - 0.6*trust), 6) if relevant else old.confidence
    relief = situation.emotion * 0.35*trust if relevant else 0
    situation.emotion = bounded(situation.emotion - relief)
    situation.evidence = (*situation.evidence, event.id)
    situation.updated_tick = world.tick
    situation.processed_claims = tuple(sorted(set((*situation.processed_claims, claim_key))))
    # Relief is partial and applies only to the relationship under discussion.
    relationship = world.relationships[f'{actor}:{situation.target}']
    gain = (old.confidence-confidence)*2
    relationship.trust = bounded(relationship.trust + gain)
    relationship.tension = bounded(relationship.tension - gain)
    revision = Interpretation(
        id=f'i:{actor}:{len(state.interpretations)+1}', situation_id=situation.id,
        tick=world.tick, evidence=situation.evidence, emotion_after=situation.emotion,
        trust_delta=gain, tension_delta=-gain, event_id=event.id, target=situation.target,
        kind='excluded' if confidence >= 0.3 else 'neutral', confidence=confidence,
        belief_ids=old.belief_ids,
        reason='Сообщение об отсутствии общего плана оценено с доверием к источнику; умысел остаётся неизвестным.' if relevant else 'Подтверждение, повтор, отказ или молчание не опровергают и не доказывают исключение.',
    )
    state.interpretations.append(revision)
    situation.current_revision = revision.id
    situation.status = 'completed' if confidence < 0.3 else 'unresolved'
    if situation.status == 'completed':
        situation.distance_active = False
    if relevant:
        remember(world, actor, event.id, situation.target, 'exclusion', 40*trust, False)
