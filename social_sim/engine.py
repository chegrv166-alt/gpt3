from __future__ import annotations

from random import Random

from social_sim.cognition import bounded, perceive
from social_sim.decision import ActorView, choose, decide
from social_sim.models import (
    ActionKind, Candidate, DecisionTrace, Memory, NPCState, Relationship,
    Scenario, SocialPlan, WorldEvent, WorldState,
)


class Simulation:
    def __init__(self, scenario: Scenario, seed: int = 42) -> None:
        self.rng = Random(seed)
        self.world = WorldState(
            scenario=scenario, states={p.id: NPCState() for p in scenario.people},
            relationships={f'{r.source}:{r.target}': r.model_copy(deep=True) for r in scenario.relationships},
        )

    def next_free_slot(self) -> int:
        tick = self.world.tick + 1
        while tick % 4 < 2:
            tick += 1
        return tick

    def busy(self, actor: str, tick: int) -> bool:
        if tick % 4 < 2:
            return True
        return any(p.tick == tick and (
            p.organizer == actor and p.status in ('pending', 'accepted', 'completed')
            or p.invitee == actor and p.status in ('accepted', 'completed')
        ) for p in self.world.plans)

    def view(self, actor: str) -> ActorView:
        person = next(p for p in self.world.scenario.people if p.id == actor)
        # The selector never receives other people's states, future decisions or the global log.
        return ActorView(person, self.world.states[actor].model_copy(deep=True),
                         tuple(r.model_copy(deep=True) for r in self.world.relationships.values() if r.source == actor),
                         self.world.tick, not self.busy(actor, self.next_free_slot()))

    def emit(self, kind: str, participants: tuple[str, ...], outcome: str, *, causes: tuple[str, ...] = (), trace: DecisionTrace | None = None, location: str = 'message') -> WorldEvent:
        event = WorldEvent(id=f'e{len(self.world.events)+1:06d}', tick=self.world.tick,
                           kind=kind, participants=participants, outcome=outcome,
                           location=location, causes=causes, trace_id=trace.id if trace else None)
        self.world.events.append(event)
        for actor in participants:
            perceive(self.world, actor, event, actor, event.id)
        return event

    def recover(self) -> None:
        for person in self.world.scenario.people:
            state = self.world.states[person.id]
            state.energy = bounded(state.energy + 8)
            state.loneliness = bounded(state.loneliness + person.personality.traits['belonging'] * 0.9)
            # Emotion settles, but no relationship scale decays with time.
            for target, hurt in state.hurt.items():
                state.hurt[target] = bounded(hurt * (0.92 + person.personality.traits['grudge_persistence'] * 0.01))

    def act(self, actor: str, action: ActionKind, target: str | None = None, event_id: str | None = None, trace: DecisionTrace | None = None) -> WorldEvent:
        if actor not in self.world.states or (target is not None and target not in self.world.states):
            raise ValueError('Неизвестный участник')
        state = self.world.states[actor]
        if action not in ('rest', 'invite', 'ask', 'distance', 'tell'):
            raise ValueError('Ответ на приглашение проходит через respond')
        if self.busy(actor, self.world.tick) or state.last_action_tick == self.world.tick:
            raise ValueError('Участник занят')
        if action != 'rest' and (target is None or target == actor or f'{actor}:{target}' not in self.world.relationships):
            raise ValueError('Нужно знакомство с другим участником')
        if action != 'rest' and state.energy < (20 if action == 'invite' else 10):
            raise ValueError('Недостаточно энергии')
        if action in ('invite', 'ask', 'distance') and self.world.tick - state.cooldowns.get(f'{action}:{target}', -100) < 8:
            raise ValueError('Повтор действия: cooldown')
        if action == 'invite' and (self.busy(actor, self.next_free_slot()) or state.avoiding_until.get(target or '', 0) > self.world.tick):
            raise ValueError('Несовместимый план или дистанция')
        if action in ('ask', 'distance') and state.hurt.get(target or '', 0) <= 10:
            raise ValueError('Нет неприятной ситуации для реакции')
        if action == 'tell':
            if event_id is None or event_id not in state.knowledge:
                raise ValueError('Нельзя сообщить неизвестный факт')
            if f'tell:{target}:{event_id}' in state.cooldowns:
                raise ValueError('Факт уже передан этому адресату')
        if trace is not None and (
            trace.id != f't{len(self.world.traces)+1:06d}' or trace.actor != actor or trace.tick != self.world.tick
            or trace.selected.action != action or trace.selected.target != target or trace.selected.event_id != event_id
            or trace.selected not in trace.candidates or set(trace.known_events) != set(state.knowledge)
        ):
            raise ValueError('Trace не соответствует действию или текущему знанию')
        # All preconditions above precede any mutation (including RNG and trace insertion).
        if trace is None:
            candidate = Candidate(action=action, target=target, event_id=event_id, components={'external_input': 1})
            trace = choose(self.view(actor), [candidate], ['Внешнее действие через общие проверки мира'], self.rng,
                           self.world.scenario.weights, f't{len(self.world.traces)+1:06d}', randomize=False)
        self.world.traces.append(trace)
        state.last_action_tick = self.world.tick
        if action == 'rest':
            state.energy = bounded(state.energy + 12)
            return self.emit('rested', (actor,), 'Отдых', trace=trace, location=state.location)
        assert target is not None
        state.energy = bounded(state.energy - (12 if action == 'invite' else 6))
        state.cooldowns[f'{action}:{target}'] = self.world.tick
        if action == 'invite':
            event = self.emit('invitation_sent', (actor, target), f'Встреча в слоте {self.next_free_slot()}', trace=trace)
            self.world.plans.append(SocialPlan(id=f'p{len(self.world.plans)+1:06d}', organizer=actor,
                                             invitee=target, tick=self.next_free_slot(), history=[event.id]))
            return event
        if action == 'tell':
            assert event_id is not None
            event = self.emit('information_shared', (actor, target), 'Сообщение о наблюдавшейся встрече', causes=(event_id,), trace=trace)
            state.cooldowns[f'tell:{target}:{event_id}'] = self.world.tick
            fact = next(e for e in self.world.events if e.id == event_id)
            perceive(self.world, target, fact, actor, event.id, message=True)
            return event
        causes = tuple(i.event_id for i in state.interpretations if i.target == target and i.kind == 'excluded')[-1:]
        if action == 'distance':
            state.avoiding_until[target] = self.world.tick + 8
            return self.emit('contact_avoided', (actor,), 'Временная дистанция; чужие мысли неизвестны', causes=causes, trace=trace)
        event = self.emit('question_asked', (actor, target), 'Вопрос о встрече без приглашения', causes=causes, trace=trace)
        self.answer_question(actor, target, event, causes)
        return event

    def answer_question(self, asker: str, respondent: str, question: WorldEvent, causes: tuple[str, ...]) -> None:
        view = self.view(respondent)
        if view.person.is_player:
            self.emit('question_unanswered', (asker, respondent), 'Игрок не ответил', causes=(question.id,))
            return
        state = self.world.states[respondent]
        available = not self.busy(respondent, self.world.tick) and state.energy >= 10
        informed = bool(causes) and all(c in state.knowledge for c in causes)
        t = view.person.personality.traits
        rel = self.world.relationships.get(f'{respondent}:{asker}')
        candidates = [Candidate(action='decline', target=asker, components={
            'privacy': t['privacy_priority'], 'emotion': state.hurt.get(asker, 0)/5,
        })]
        if available and informed:
            candidates.append(Candidate(action='clarify', target=asker, components={
                'openness': t['openness'], 'relationship': rel.trust/20 if rel else 0,
                'repair': t['reconciliation_readiness'], 'effort': -2,
            }))
        trace = choose(view, candidates, [] if available and informed else ['Ответ: нет доступного знания, времени или энергии'],
                       self.rng, self.world.scenario.weights, f't{len(self.world.traces)+1:06d}')
        self.world.traces.append(trace)
        clarified = trace.selected.action == 'clarify'
        self.emit('question_answered' if clarified else 'question_declined', (respondent, asker),
                  'Подтверждено участие во встрече; намеренное исключение не подтверждено.' if clarified else 'Обсуждение не состоялось.',
                  causes=(question.id,), trace=trace)
        if clarified:
            state.energy = bounded(state.energy - 4)
            trust = self.world.relationships[f'{asker}:{respondent}'].trust
            observer = self.world.states[asker]
            observer.hurt[respondent] = bounded(observer.hurt[respondent] * (1-trust/200))

    def respond(self, plan_id: str, accept: bool | None = None) -> WorldEvent:
        plan = next((p for p in self.world.plans if p.id == plan_id), None)
        if plan is None or plan.status != 'pending' or plan.tick < self.world.tick:
            raise ValueError('Нет актуального приглашения')
        view = self.view(plan.invitee)
        state = self.world.states[plan.invitee]
        rel = self.world.relationships.get(f'{plan.invitee}:{plan.organizer}')
        available = not self.busy(plan.invitee, plan.tick) and state.energy >= 15 and state.avoiding_until.get(plan.organizer, 0) <= self.world.tick
        if accept is True and not available:
            raise ValueError('Участник недоступен')
        t = view.person.personality.traits
        candidates = [Candidate(action='decline', target=plan.organizer, components={
            'autonomy': t['autonomy'] * 1.5, 'fatigue': (100-state.energy)/10,
            'emotion': state.hurt.get(plan.organizer, 0)/5,
        })]
        if available:
            candidates.append(Candidate(action='accept', target=plan.organizer, components={
                'need': state.loneliness/10, 'personality': t['sociability'],
                'relationship': (rel.liking + rel.trust - rel.tension)/20 if rel else 0,
                'effort': -3,
            }))
        constraints = [] if available else ['accept: несовместимое время, дистанция или энергия']
        if accept is not None:
            candidates = [c for c in candidates if c.action == ('accept' if accept else 'decline')]
            constraints.append('Внешний ответ; предусловия проверены')
        trace = choose(view, candidates, constraints, self.rng, self.world.scenario.weights,
                       f't{len(self.world.traces)+1:06d}', randomize=accept is None)
        self.world.traces.append(trace)
        accepted = trace.selected.action == 'accept'
        event = self.emit('invitation_accepted' if accepted else 'invitation_declined',
                          (plan.invitee, plan.organizer), 'Согласие' if accepted else 'Отказ',
                          causes=(plan.history[0],), trace=trace)
        plan.status = 'accepted' if accepted else 'declined'
        plan.history.append(event.id)
        return event

    def share_information(self, sender: str, receiver: str, event_id: str) -> WorldEvent | None:
        if sender not in self.world.states or receiver not in self.world.states or event_id not in self.world.states[sender].knowledge:
            raise ValueError('Участник не знает факт')
        if event_id in self.world.states[receiver].knowledge:
            return None
        return self.act(sender, 'tell', receiver, event_id)

    def meet(self, plan: SocialPlan) -> None:
        event = self.emit('meeting_occurred', (plan.organizer, plan.invitee), 'Совместная прогулка',
                          causes=tuple(plan.history), location=plan.location)
        plan.status = 'completed'
        plan.history.append(event.id)
        for actor, other in ((plan.organizer, plan.invitee), (plan.invitee, plan.organizer)):
            state = self.world.states[actor]
            state.location = plan.location
            state.energy = bounded(state.energy - 15)
            state.loneliness = bounded(state.loneliness - 30)
            key = f'{actor}:{other}'
            if key not in self.world.relationships:
                self.world.relationships[key] = Relationship(source=actor, target=other)
            rel = self.world.relationships[key]
            rel.shared_meetings += 1
            rel.liking = bounded(rel.liking + 2)
            rel.trust = bounded(rel.trust + 1)
            rel.closeness = bounded(rel.closeness + 3 * (1-rel.closeness/120))
            rel.perceived_reciprocity = bounded(rel.perceived_reciprocity + 2)
            state.memories.append(Memory(event_id=event.id, target=other, significance=30, valence='positive'))
            state.memories[:] = state.memories[-24:]

    def step(self) -> None:
        self.recover()
        for state in self.world.states.values():
            state.location = 'school' if self.world.tick % 4 < 2 else 'home'
        if self.world.tick % 4 >= 2:
            people = {p.id: p for p in self.world.scenario.people}
            for plan in self.world.plans:
                if plan.status == 'pending' and not people[plan.invitee].is_player:
                    self.respond(plan.id)
                if plan.tick <= self.world.tick:
                    if plan.status == 'accepted':
                        self.meet(plan)
                    elif plan.status == 'pending':
                        event = self.emit('invitation_expired', (plan.organizer, plan.invitee), 'Ответ не получен', causes=tuple(plan.history))
                        plan.status = 'cancelled'
                        plan.history.append(event.id)
            actors = sorted(p.id for p in self.world.scenario.people if not p.is_player)
            self.rng.shuffle(actors)
            for actor in actors:
                if self.busy(actor, self.world.tick) or self.world.states[actor].last_action_tick == self.world.tick:
                    continue
                trace = decide(self.view(actor), self.rng, self.world.scenario.weights, f't{len(self.world.traces)+1:06d}')
                selected = trace.selected
                self.act(actor, selected.action, selected.target, selected.event_id, trace)
        self.world.tick += 1

    def run_days(self, days: int) -> None:
        if days < 0:
            raise ValueError('Число дней должно быть неотрицательным')
        for _ in range(days * 4):
            self.step()
