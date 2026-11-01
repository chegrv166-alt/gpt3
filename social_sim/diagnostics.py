from collections import Counter

from social_sim.models import WorldState


def metrics(world: WorldState) -> dict[str, object]:
    counts = Counter(e.kind for e in world.events)
    npc_ids = {p.id for p in world.scenario.people if not p.is_player}
    initiative_kinds = {'invite', 'ask', 'distance', 'tell'}
    initiatives = Counter(t.selected.action for t in world.traces if t.selected.action in initiative_kinds)
    responses = Counter(t.selected.action for t in world.traces if t.selected.action in ('accept', 'decline', 'clarify'))
    active = {t.actor for t in world.traces if t.selected.action in initiative_kinds}
    situations = [s for state in world.states.values() for s in state.situations.values()]
    revisions = [i for state in world.states.values() for i in state.interpretations]
    reactions = [r for situation in situations for r in situation.reactions]
    cause_counts = Counter((s.id, r.generation) for s in situations for r in s.reactions)
    initial = {f'{r.source}:{r.target}': r for r in world.scenario.relationships}
    changed = sum(key not in initial or rel != initial[key] for key, rel in world.relationships.items())
    meaningful = [e for e in world.events if e.kind not in ('rested', 'meeting_occurred', 'invitation_expired', 'question_unanswered')]
    return {
        'schema_version': 2, 'elapsed_slots': world.tick, 'event_counts': dict(sorted(counts.items())),
        'social_initiatives': sum(initiatives.values()), 'invitation_initiatives': counts['invitation_sent'],
        'initiatives_by_action': dict(initiatives), 'responses_by_action': dict(responses),
        'social_situations': len(situations), 'situation_reactions': len(reactions),
        'repeated_reactions_same_cause': sum(max(0, count-1) for count in cause_counts.values()),
        'interpretation_revisions': len(revisions) - len(situations),
        'revision_grounds': dict(Counter(i.reason for i in revisions)),
        'decisions_using_memory': sum(bool(t.memories) for t in world.traces),
        'completed_situations': sum(s.status == 'completed' for s in situations),
        'unresolved_situations': sum(s.status == 'unresolved' for s in situations),
        'situations_by_status': dict(Counter(s.status for s in situations)), 'accepted': counts['invitation_accepted'],
        'declined': counts['invitation_declined'], 'meetings': counts['meeting_occurred'],
        'changed_directed_relationships': changed,
        'new_directed_acquaintances': len(set(world.relationships)-set(initial)),
        'social_action_types': sorted({t.selected.action for t in world.traces if t.selected.action != 'rest'}),
        'inactive_npc_fraction': len(npc_ids-active)/len(npc_ids) if npc_ids else 0,
        'explained_decision_fraction': sum(e.trace_id is not None for e in meaningful)/len(meaningful) if meaningful else None,
        'interpretations': dict(Counter(i.kind for s in world.states.values() for i in s.interpretations)),
        'not_implemented': ['конфликты', 'извинения и примирения', 'динамические компании', 'авторские сцены'],
    }


def explain(world: WorldState, event_id: str) -> dict[str, object]:
    events = {e.id: e for e in world.events}
    traces = {t.id: t for t in world.traces}
    if event_id not in events:
        raise ValueError('Событие не найдено')
    event = events[event_id]
    trace = traces.get(event.trace_id or '')
    state = world.states[trace.actor] if trace else None
    revisions = {i.id: i for i in state.interpretations} if state else {}
    return {
        'event': event.model_dump(),
        'decision': traces[event.trace_id].model_dump() if event.trace_id else None,
        'knowledge_at_decision': [state.knowledge[e].model_dump() for e in trace.known_events] if trace and state else [],
        'interpretations_at_decision': [revisions[rid].model_dump() for rid in trace.interpretation_ids] if trace else [],
        'state_updates': [
            {'observer': actor, **i.model_dump()} for actor, current_state in world.states.items()
            for i in current_state.interpretations if i.event_id == event_id or i.event_id in {e.id for e in world.events if event_id in e.causes}
        ],
        'causes': [
            {'event': events[c].model_dump(), 'decision': traces[cause_trace].model_dump() if (cause_trace := events[c].trace_id) else None}
            for c in event.causes
        ],
        'interpretations': [
            {'observer': actor, **i.model_dump()} for actor, s in world.states.items()
            for i in s.interpretations if i.event_id == event_id
        ],
        'audience': 'Диагностика разработчика; не является знаниями игрока.',
    }
