from collections import Counter

from social_sim.models import WorldState


def metrics(world: WorldState) -> dict[str, object]:
    counts = Counter(e.kind for e in world.events)
    npc_ids = {p.id for p in world.scenario.people if not p.is_player}
    active = {t.actor for t in world.traces if t.selected.action not in ('rest', 'decline')}
    initial = {f'{r.source}:{r.target}': r for r in world.scenario.relationships}
    changed = sum(key not in initial or rel != initial[key] for key, rel in world.relationships.items())
    meaningful = [e for e in world.events if e.kind not in ('rested', 'meeting_occurred', 'invitation_expired', 'question_unanswered')]
    return {
        'schema_version': 1, 'elapsed_slots': world.tick, 'event_counts': dict(sorted(counts.items())),
        'social_initiatives': counts['invitation_sent'], 'accepted': counts['invitation_accepted'],
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
    return {
        'event': event.model_dump(),
        'decision': traces[event.trace_id].model_dump() if event.trace_id else None,
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
