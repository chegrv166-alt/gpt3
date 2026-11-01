from __future__ import annotations

from typing import Annotated, Literal, Self, cast

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


class Traits(Frozen):
    belonging: TraitValue
    closeness: TraitValue
    autonomy: TraitValue
    recognition: TraitValue
    stability: TraitValue
    novelty: TraitValue
    initiative: TraitValue
    sociability: TraitValue
    openness: TraitValue
    directness: TraitValue
    tact: TraitValue
    conformity: TraitValue
    rejection_sensitivity: TraitValue
    humiliation_sensitivity: TraitValue
    suspiciousness: TraitValue
    impulsivity: TraitValue
    grudge_persistence: TraitValue
    reconciliation_readiness: TraitValue
    inclusion_expectation: TraitValue
    contact_expectation: TraitValue
    reciprocity_expectation: TraitValue
    loyalty_priority: TraitValue
    honesty_priority: TraitValue
    privacy_priority: TraitValue

    def __getitem__(self, key: str) -> int:
        if key not in TRAITS:
            raise KeyError(key)
        return cast(int, getattr(self, key))

    def keys(self) -> frozenset[str]:
        return TRAITS

    def __len__(self) -> int:
        return len(TRAITS)


class Personality(Frozen):
    traits: Traits
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
    meeting_context: Literal['unknown', 'no_prior_group_plan'] = 'unknown'

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
    organizer: str | None = None
    context: Literal['unknown', 'no_prior_group_plan'] = 'unknown'
    claim: Literal['meeting_confirmed', 'no_prior_group_plan', 'refused', 'unanswered'] | None = None
    subject_event: str | None = None
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
    organizer: str | None = None
    context: Literal['unknown', 'no_prior_group_plan'] = 'unknown'


class Interpretation(Frozen):
    id: str
    situation_id: str
    tick: int
    evidence: tuple[str, ...]
    memory_ids: tuple[str, ...] = ()
    used_memories: tuple[Memory, ...] = ()
    memory_effect: float = 0
    emotion_after: Scale
    trust_delta: float = 0
    tension_delta: float = 0
    event_id: str
    kind: Literal['excluded', 'neutral']
    target: str
    confidence: Annotated[float, Field(strict=True, ge=0, le=1)]
    belief_ids: tuple[str, ...]
    reason: str


class Memory(Frozen):
    id: str
    tick: Annotated[int, Field(ge=0)]
    topic: str
    source: str
    confidence: Annotated[float, Field(strict=True, ge=0, le=1)]
    event_id: str
    target: str
    significance: Scale
    valence: Literal['positive', 'negative']


class Reaction(Frozen):
    event_id: str
    revision_id: str
    generation: int
    action: Literal['ask', 'distance']


class SocialSituation(Model):
    id: str
    observer: str
    target: str
    related: tuple[str, ...]
    expectation: str
    evidence: tuple[str, ...]
    current_revision: str
    updated_tick: int
    status: Literal['active', 'discussing', 'completed', 'unresolved'] = 'active'
    emotion: Scale = 0
    generation: Annotated[int, Field(ge=1)] = 1
    reacted_generation: Annotated[int, Field(ge=0)] = 0
    reactions: list[Reaction] = Field(default_factory=list)
    distance_active: bool = False
    processed_claims: tuple[str, ...] = ()


class NPCState(Model):
    last_action_tick: Annotated[int, Field(ge=-1)] = -1
    energy: Scale = 80
    loneliness: Scale = 35
    location: str = 'home'
    situations: dict[str, SocialSituation] = Field(default_factory=dict)
    knowledge: dict[str, KnowledgeRecord] = Field(default_factory=dict)
    interpretations: list[Interpretation] = Field(default_factory=list)
    memories: list[Memory] = Field(default_factory=list)
    cooldowns: dict[str, int] = Field(default_factory=dict)

    @property
    def hurt(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for situation in self.situations.values():
            if situation.emotion > 0:
                result[situation.target] = min(100, result.get(situation.target, 0) + situation.emotion)
        return result

    def avoiding(self, target: str) -> bool:
        return any(s.target == target and s.distance_active for s in self.situations.values())

    def actionable_situation(self, target: str) -> SocialSituation | None:
        return next((s for s in self.situations.values() if s.target == target and s.emotion > 10
                     and s.status != 'completed' and s.generation > s.reacted_generation), None)



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
    situation_id: str | None = None
    memory_ids: tuple[str, ...] = ()
    components: dict[str, float]

    @property
    def utility(self) -> float:
        return sum(self.components.values())


class SituationAtDecision(Frozen):
    id: str
    revision_id: str
    emotion: Scale
    status: Literal['active', 'discussing', 'completed', 'unresolved']
    generation: int
    distance_active: bool


class DecisionTrace(Frozen):
    event_boundary: Annotated[int, Field(ge=0)]
    id: str
    actor: str
    tick: int
    known_events: tuple[str, ...]
    interpretation_events: tuple[str, ...]
    interpretation_ids: tuple[str, ...] = ()
    memories: tuple[Memory, ...] = ()
    situations: tuple[SituationAtDecision, ...] = ()
    goals: tuple[Goal, ...]
    candidates: tuple[Candidate, ...]
    constraints: tuple[str, ...]
    selected: Candidate
    random_draw: float | None
    selection_pool: tuple[int, ...]
    explanation: str


class WorldState(Model):
    schema_version: Literal[2] = 2
    tick: Annotated[int, Field(ge=0)] = 0
    scenario: Scenario
    states: dict[str, NPCState]
    relationships: dict[str, Relationship]
    plans: list[SocialPlan] = Field(default_factory=list)
    events: list[WorldEvent] = Field(default_factory=list)
    traces: list[DecisionTrace] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)

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
            if e.organizer is not None and e.organizer not in e.participants:
                raise ValueError('Некорректная роль организатора')
            if e.subject_event is not None and e.subject_event not in event_ids:
                raise ValueError('Неизвестный предмет разговора')
            if not set(e.causes) <= event_ids or not set(e.participants + e.witnesses) <= ids:
                raise ValueError('Некорректная ссылка события')
            if e.trace_id is not None and e.trace_id not in trace_ids:
                raise ValueError('Неизвестный trace')
            event_ids.add(e.id)
            previous_tick = e.tick
        events_by_id = {e.id: e for e in self.events}
        event_order = {e.id: n for n, e in enumerate(self.events,1)}
        trace_events = {e.trace_id: e for e in self.events if e.trace_id is not None}
        for actor, state in self.states.items():
            for key, k in state.knowledge.items():
                if key != k.event_id or key not in event_ids or k.via_event not in event_ids or k.source not in ids:
                    raise ValueError('Некорректная ссылка знания')
                fact, via = events_by_id[key], events_by_id[k.via_event]
                if k.kind != fact.kind or k.participants != fact.participants or k.organizer != fact.organizer:
                    raise ValueError('Знание не соответствует факту')
                if not fact.tick <= k.acquired_tick <= self.tick or via.tick > k.acquired_tick:
                    raise ValueError('Знание из будущего')
                if k.method == 'observation':
                    if actor not in fact.participants + fact.witnesses or k.context != fact.context or k.via_event != key:
                        raise ValueError('Недоступное наблюдение')
                else:
                    if actor not in via.participants or k.source not in via.participants or key not in via.causes or k.context != 'unknown':
                        raise ValueError('Нет канала передачи знания')
                    sender = self.states[k.source].knowledge.get(key)
                    if sender is None or sender.acquired_tick > k.acquired_tick:
                        raise ValueError('Отправитель не знал факт')
            def check_memory(memory: Memory, at: int) -> None:
                record = state.knowledge.get(memory.event_id)
                if (record is None or memory.target not in ids or memory.source not in ids
                    or memory.source != record.source or not record.acquired_tick <= memory.tick <= at
                    or memory.id != f'm:{actor}:{memory.event_id}:{memory.target}:{memory.topic}'):
                    raise ValueError('Некорректная память или происхождение')
            if len(state.memories) > 24 or len({m.id for m in state.memories}) != len(state.memories):
                raise ValueError('Нарушено ограничение памяти')
            for memory in state.memories:
                check_memory(memory, self.tick)
            revisions = {i.id: i for i in state.interpretations}
            if len(revisions) != len(state.interpretations):
                raise ValueError('Повтор версии интерпретации')
            for n, revision in enumerate(state.interpretations, 1):
                if (revision.id != f'i:{actor}:{n}' or revision.target not in ids
                    or revision.situation_id not in state.situations
                    or revision.event_id not in revision.evidence or not revision.evidence
                    or len(set(revision.evidence)) != len(revision.evidence)
                    or not set(revision.evidence) <= state.knowledge.keys()
                    or not 0 <= revision.tick <= self.tick):
                    raise ValueError('Некорректная версия интерпретации')
                if any(state.knowledge[e].acquired_tick > revision.tick for e in revision.evidence):
                    raise ValueError('Интерпретация использует будущее знание')
                if revision.memory_ids != tuple(m.id for m in revision.used_memories):
                    raise ValueError('Неизвестная память интерпретации')
                for memory in revision.used_memories:
                    check_memory(memory, revision.tick)
            for key, situation in state.situations.items():
                current = revisions.get(situation.current_revision)
                if (key != situation.id or situation.id != f's:{actor}:{situation.target}:exclusion'
                    or situation.observer != actor or situation.target not in ids or situation.target == actor
                    or not set(situation.related) <= ids or situation.target not in situation.related
                    or current is None or current.situation_id != key or current.target != situation.target
                    or current.evidence != situation.evidence or situation.updated_tick != current.tick
                    or situation.reacted_generation > situation.generation):
                    raise ValueError('Некорректная социальная ситуация')
                generations: set[int] = set()
                for reaction in situation.reactions:
                    prior = revisions.get(reaction.revision_id)
                    event = events_by_id.get(reaction.event_id)
                    if (prior is None or prior.situation_id != key or event is None
                        or event.kind != ('question_asked' if reaction.action == 'ask' else 'contact_avoided')
                        or event.participants[0] != actor or prior.tick > event.tick
                        or reaction.generation in generations or not 1 <= reaction.generation <= situation.generation):
                        raise ValueError('Некорректная реакция на ситуацию')
                    generations.add(reaction.generation)
                if situation.status == 'discussing' and not any(r.event_id in self.pending_questions for r in situation.reactions):
                    raise ValueError('Обсуждение без ожидающего вопроса')
                if situation.reacted_generation != max(generations, default=0):
                    raise ValueError('Реакция не соответствует поколению ситуации')
            for index, t in enumerate(self.traces, 1):
                if t.actor != actor:
                    continue
                emitted = trace_events.get(t.id)
                if emitted is None or event_order[emitted.id] != t.event_boundary + 1 or emitted.tick != t.tick or emitted.participants[0] != actor:
                    raise ValueError('Неверная граница времени решения')
                if t.id != f't{index:06d}' or not set(t.known_events) <= state.knowledge.keys() or not 0 <= t.tick <= self.tick:
                    raise ValueError('Некорректный trace')
                if not set(t.interpretation_events) <= set(t.known_events) or not set(t.interpretation_ids) <= revisions.keys():
                    raise ValueError('Неизвестная интерпретация trace')
                for rid in t.interpretation_ids:
                    if revisions[rid].tick > t.tick or not set(revisions[rid].evidence) <= set(t.known_events):
                        raise ValueError('Будущая интерпретация trace')
                if t.selected not in t.candidates or not t.selection_pool or any(i < 0 or i >= len(t.candidates) for i in t.selection_pool):
                    raise ValueError('Некорректные альтернативы trace')
                for memory in t.memories:
                    check_memory(memory, t.tick)
                for snapshot in t.situations:
                    if snapshot.revision_id not in t.interpretation_ids or revisions[snapshot.revision_id].situation_id != snapshot.id:
                        raise ValueError('Некорректный исторический снимок ситуации')
                for candidate in t.candidates:
                    if candidate.target is not None and (candidate.target not in ids or candidate.target == actor):
                        raise ValueError('Неизвестный адресат trace')
                    if candidate.event_id is not None and candidate.event_id not in t.known_events:
                        raise ValueError('Решение использует неизвестный факт')
                    if not set(candidate.memory_ids) <= {m.id for m in t.memories}:
                        raise ValueError('Неизвестная память trace')
                    if candidate.situation_id is not None and candidate.situation_id not in {snap.id for snap in t.situations}:
                        raise ValueError('Неизвестная ситуация trace')
                for known in t.known_events:
                    if state.knowledge[known].acquired_tick > t.tick or event_order[state.knowledge[known].via_event] > t.event_boundary:
                        raise ValueError('Решение использует недоступное знание')
        for question_id in self.pending_questions:
            question = events_by_id.get(question_id)
            if question is None or question.kind != 'question_asked' or len(question.participants) != 2:
                raise ValueError('Некорректный отложенный вопрос')
            state = self.states[question.participants[0]]
            if not any(s.status == 'discussing' and any(r.event_id == question_id for r in s.reactions) for s in state.situations.values()):
                raise ValueError('Нет обсуждаемой ситуации для вопроса')
        if len(set(self.pending_questions)) != len(self.pending_questions):
            raise ValueError('Повтор отложенного вопроса')
        if any(t.actor not in ids for t in self.traces):
            raise ValueError('Неизвестный автор trace')
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
            history = [events_by_id[e] for e in p.history]
            expected = {
                'pending': ['invitation_sent'],
                'accepted': ['invitation_sent', 'invitation_accepted'],
                'declined': ['invitation_sent', 'invitation_declined'],
                'completed': ['invitation_sent', 'invitation_accepted', 'meeting_occurred'],
                'cancelled': ['invitation_sent', 'invitation_expired'],
            }[p.status]
            if [e.kind for e in history] != expected or len(set(p.history)) != len(p.history):
                raise ValueError('Невозможная последовательность плана')
            for position, event in enumerate(history):
                if set(event.participants) != {p.organizer, p.invitee} or len(event.participants) != 2:
                    raise ValueError('Участники плана не совпадают')
                if position and (event.tick < history[position-1].tick or event.tick > p.tick):
                    raise ValueError('Неверное время перехода плана')
                if position and event.causes != tuple(p.history[:position]):
                    raise ValueError('Неверные причины перехода плана')
            if p.status in ('completed', 'cancelled') and (history[-1].tick != p.tick or p.tick > self.tick):
                raise ValueError('Время исполнения плана не совпадает с событием')
            if p.status in ('pending', 'accepted', 'completed'):
                participants = [p.organizer] + ([p.invitee] if p.status != 'pending' else [])
                for actor in participants:
                    if (p.tick, actor) in occupied:
                        raise ValueError('Несовместимые обязательства')
                    occupied.add((p.tick, actor))
        return self
