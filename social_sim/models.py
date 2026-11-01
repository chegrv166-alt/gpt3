from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TraitValue = Annotated[int, Field(strict=True, ge=0, le=5)]
Scale = Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)]
TRAITS = frozenset('belonging closeness autonomy recognition stability novelty initiative sociability openness directness tact conformity rejection_sensitivity humiliation_sensitivity suspiciousness impulsivity grudge_persistence reconciliation_readiness inclusion_expectation contact_expectation reciprocity_expectation loyalty_priority honesty_priority privacy_priority'.split())
ActionKind = Literal['rest', 'invite', 'ask', 'distance', 'tell', 'accept', 'decline', 'clarify']


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)


class Frozen(Model):
    model_config = ConfigDict(extra='forbid', frozen=True)


class BeliefRule(Frozen):
    id: str
    condition: Literal['friends_met_without_me']
    interpretation: Literal['excluded']
    strength: Annotated[float, Field(strict=True, ge=0, le=1)]
    description: str


class Goal(Frozen):
    kind: Literal['connect', 'preserve_autonomy']
    priority: Annotated[float, Field(strict=True, ge=0, le=1)]


class Personality(Frozen):
    traits: dict[str, TraitValue]
    beliefs: tuple[BeliefRule, ...] = ()
    strategies: tuple[Literal['ask', 'distance', 'invite'], ...] = ()
    goals: tuple[Goal, ...] = ()

    @model_validator(mode='after')
    def complete_traits(self) -> Self:
        if self.traits.keys() != TRAITS:
            raise ValueError('Нужны ровно 24 характеристики: ' + ', '.join(sorted(TRAITS)))
        return self


class Person(Frozen):
    id: Annotated[str, Field(pattern=r'^[a-z][a-z0-9_]*$')]
    name: str
    biography: str
    synthetic: Literal[True]
    is_player: bool = False
    personality: Personality


class Relationship(Model):
    source: str
    target: str
    liking: Scale = 40
    trust: Scale = 50
    closeness: Scale = 20
    tension: Scale = 0
    romantic_interest: Scale = 0
    perceived_reciprocity: Scale = 50
    shared_meetings: Annotated[int, Field(ge=0)] = 0


class Weights(Frozen):
    need: Annotated[float, Field(strict=True, ge=0, le=5)] = 1
    personality: Annotated[float, Field(strict=True, ge=0, le=5)] = 1
    goal: Annotated[float, Field(strict=True, ge=0, le=5)] = 1
    relationship: Annotated[float, Field(strict=True, ge=0, le=5)] = 1
    emotion: Annotated[float, Field(strict=True, ge=0, le=5)] = 1
    selection_band: Annotated[float, Field(strict=True, gt=0, le=20)] = 3


class Scenario(Frozen):
    schema_version: Literal[1]
    description: str
    people: tuple[Person, ...]
    relationships: tuple[Relationship, ...]
    weights: Weights = Field(default_factory=Weights)

    @model_validator(mode='after')
    def references(self) -> Self:
        ids = [p.id for p in self.people]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError('ID персонажей должны быть уникальны')
        if sum(p.is_player for p in self.people) != 1:
            raise ValueError('Требуется ровно один игрок')
        pairs = set()
        for r in self.relationships:
            if r.source not in ids or r.target not in ids or r.source == r.target:
                raise ValueError('Некорректная ссылка отношений')
            if (r.source, r.target) in pairs:
                raise ValueError('Повтор отношений')
            pairs.add((r.source, r.target))
        return self


class WorldEvent(Frozen):
    id: str
    tick: Annotated[int, Field(ge=0)]
    kind: str
    participants: tuple[str, ...]
    location: str
    outcome: str
    witnesses: tuple[str, ...] = ()
    visibility: Literal['participants', 'witnesses'] = 'participants'
    causes: tuple[str, ...] = ()
    trace_id: str | None = None


class KnowledgeRecord(Frozen):
    event_id: str
    kind: str
    participants: tuple[str, ...]
    source: str
    acquired_tick: int
    method: Literal['observation', 'message']
    confidence: Annotated[float, Field(strict=True, ge=0, le=1)]
    via_event: str


class Interpretation(Frozen):
    event_id: str
    kind: Literal['excluded', 'neutral']
    target: str
    confidence: Annotated[float, Field(strict=True, ge=0, le=1)]
    belief_ids: tuple[str, ...]
    reason: str


class Memory(Frozen):
    event_id: str
    target: str
    significance: Scale
    valence: Literal['positive', 'negative']


class NPCState(Model):
    last_action_tick: Annotated[int, Field(ge=-1)] = -1
    energy: Scale = 80
    loneliness: Scale = 35
    location: str = 'home'
    hurt: dict[str, Scale] = Field(default_factory=dict)
    knowledge: dict[str, KnowledgeRecord] = Field(default_factory=dict)
    interpretations: list[Interpretation] = Field(default_factory=list)
    memories: list[Memory] = Field(default_factory=list)
    cooldowns: dict[str, int] = Field(default_factory=dict)
    avoiding_until: dict[str, int] = Field(default_factory=dict)


class SocialPlan(Model):
    id: str
    organizer: str
    invitee: str
    tick: Annotated[int, Field(ge=0)]
    location: str = 'park'
    status: Literal['pending', 'accepted', 'declined', 'completed', 'cancelled'] = 'pending'
    history: list[str] = Field(default_factory=list)


class Candidate(Frozen):
    action: ActionKind
    target: str | None = None
    event_id: str | None = None
    components: dict[str, float]

    @property
    def utility(self) -> float:
        return sum(self.components.values())


class DecisionTrace(Frozen):
    id: str
    actor: str
    tick: int
    known_events: tuple[str, ...]
    interpretation_events: tuple[str, ...]
    goals: tuple[Goal, ...]
    candidates: tuple[Candidate, ...]
    constraints: tuple[str, ...]
    selected: Candidate
    random_draw: float | None
    selection_pool: tuple[int, ...]
    explanation: str


class WorldState(Model):
    schema_version: Literal[1] = 1
    tick: Annotated[int, Field(ge=0)] = 0
    scenario: Scenario
    states: dict[str, NPCState]
    relationships: dict[str, Relationship]
    plans: list[SocialPlan] = Field(default_factory=list)
    events: list[WorldEvent] = Field(default_factory=list)
    traces: list[DecisionTrace] = Field(default_factory=list)

    @model_validator(mode='after')
    def integrity(self) -> Self:
        ids = {p.id for p in self.scenario.people}
        if set(self.states) != ids:
            raise ValueError('Некорректный набор состояний NPC')
        for key, rel in self.relationships.items():
            if key != f'{rel.source}:{rel.target}' or rel.source not in ids or rel.target not in ids or rel.source == rel.target:
                raise ValueError('Некорректные отношения')
        event_ids: set[str] = set()
        trace_ids = {t.id for t in self.traces}
        if len(trace_ids) != len(self.traces):
            raise ValueError('Повтор trace ID')
        previous_tick = -1
        for index, e in enumerate(self.events, 1):
            if e.id != f'e{index:06d}' or e.tick < previous_tick or e.tick > self.tick:
                raise ValueError('Неверный порядок событий')
            if not set(e.causes) <= event_ids or not set(e.participants + e.witnesses) <= ids:
                raise ValueError('Некорректная ссылка события')
            if e.trace_id is not None and e.trace_id not in trace_ids:
                raise ValueError('Неизвестный trace')
            event_ids.add(e.id)
            previous_tick = e.tick
        for actor, state in self.states.items():
            if not set(state.hurt) <= ids or not set(state.avoiding_until) <= ids:
                raise ValueError('Некорректная ссылка состояния')
            for key, k in state.knowledge.items():
                if key != k.event_id or key not in event_ids or k.via_event not in event_ids or k.source not in ids or not set(k.participants) <= ids:
                    raise ValueError('Некорректная ссылка знания')
            records: list[Memory | Interpretation] = [*state.memories, *state.interpretations]
            for item in records:
                if item.event_id not in state.knowledge or item.target not in ids:
                    raise ValueError('Память без знания')
        events_by_id = {e.id: e for e in self.events}
        for index, t in enumerate(self.traces, 1):
            if t.id != f't{index:06d}' or t.actor not in ids or not set(t.known_events) <= event_ids:
                raise ValueError('Некорректный trace')
            if not set(t.interpretation_events) <= set(t.known_events) or t.tick > self.tick or t.tick < 0:
                raise ValueError('Неизвестная интерпретация или неверное время trace')
            if t.selected not in t.candidates or not t.selection_pool or any(i < 0 or i >= len(t.candidates) for i in t.selection_pool):
                raise ValueError('Некорректные альтернативы trace')
            for candidate in t.candidates:
                if candidate.target is not None and (candidate.target not in ids or candidate.target == t.actor):
                    raise ValueError('Неизвестный адресат trace')
                if candidate.event_id is not None and candidate.event_id not in t.known_events:
                    raise ValueError('Решение использует неизвестный факт')
            for known in t.known_events:
                record = self.states[t.actor].knowledge.get(known)
                if record is None or record.acquired_tick > t.tick or events_by_id[known].tick > t.tick:
                    raise ValueError('Решение использует недоступное знание')
        occupied: set[tuple[int, str]] = set()
        for index, p in enumerate(self.plans, 1):
            if p.id != f'p{index:06d}' or p.organizer not in ids or p.invitee not in ids or p.organizer == p.invitee or p.tick % 4 < 2 or not set(p.history) <= event_ids:
                raise ValueError('Некорректный план')
            if not p.history:
                raise ValueError('План без приглашения')
            invitation = events_by_id[p.history[0]]
            if invitation.kind != 'invitation_sent' or invitation.participants != (p.organizer, p.invitee) or invitation.tick >= p.tick:
                raise ValueError('План не соответствует приглашению')
            if p.status in ('pending', 'accepted') and p.tick < self.tick:
                raise ValueError('Просроченное обязательство')
            last_kind = {'pending': 'invitation_sent', 'accepted': 'invitation_accepted', 'declined': 'invitation_declined', 'completed': 'meeting_occurred', 'cancelled': 'invitation_expired'}[p.status]
            if events_by_id[p.history[-1]].kind != last_kind:
                raise ValueError('Статус плана не соответствует истории')
            if p.status in ('pending', 'accepted', 'completed'):
                participants = [p.organizer] + ([p.invitee] if p.status != 'pending' else [])
                for actor in participants:
                    if (p.tick, actor) in occupied:
                        raise ValueError('Несовместимые обязательства')
                    occupied.add((p.tick, actor))
        return self
