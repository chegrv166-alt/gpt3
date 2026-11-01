from __future__ import annotations

from random import Random

from social_sim.groups import update_groups
from social_sim.cognition import bounded, perceive, remember
from social_sim.decision import ActorView, choose, decide
from social_sim.models import (
    ActionKind, Candidate, DecisionTrace, NPCState, Relationship, Reaction,
    Scenario, PlanReply, SocialPlan, WorldEvent, WorldState,
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
            or actor in p.accepted and p.status in ('accepted', 'completed')
        ) for p in self.world.plans)

    def view(self, actor: str) -> ActorView:
        person = next(p for p in self.world.scenario.people if p.id == actor)
        # The selector never receives other people's states, future decisions or the global log.
        return ActorView(person, self.world.states[actor].model_copy(deep=True),
                         tuple(r.model_copy(deep=True) for r in self.world.relationships.values() if r.source == actor),
                         self.world.tick, not self.busy(actor, self.next_free_slot()), len(self.world.events))

    def emit(self, kind: str, participants: tuple[str, ...], outcome: str, *, causes: tuple[str, ...] = (), trace: DecisionTrace | None = None, location: str = 'message', organizer: str | None = None,
             context: str = 'unknown', claim: str | None = None, subject_event: str | None = None) -> WorldEvent:
        event = WorldEvent(id=f'e{len(self.world.events)+1:06d}', tick=self.world.tick,
                           kind=kind, participants=participants, outcome=outcome,
                           location=location, causes=causes, trace_id=trace.id if trace else None,
                           organizer=organizer, context=context, claim=claim, subject_event=subject_event)
        self.world.events.append(event)
        for actor in participants:
            perceive(self.world, actor, event, participants[0], event.id)
        return event

    def recover(self) -> None:
        for person in self.world.scenario.people:
            state = self.world.states[person.id]
            state.energy = bounded(state.energy + 8)
            state.loneliness = bounded(state.loneliness + person.personality.traits['belonging'] * 0.9)
            # Emotion settles, but no relationship scale decays with time.
            for situation in state.situations.values():
                for assessment in situation.episodes.values():
                    assessment.emotion = bounded(assessment.emotion * (0.92 + person.personality.traits['grudge_persistence'] * 0.01))
                if situation.emotion <= 10:
                    situation.distance_active = False

    def act(self, actor: str, action: ActionKind, target: str | None = None, event_id: str | None = None, trace: DecisionTrace | None = None, *, defer_reply: bool = False) -> WorldEvent:
        if actor not in self.world.states or (target is not None and target not in self.world.states):
            raise ValueError('Неизвестный участник')
        state = self.world.states[actor]
        if (action != 'tell' and event_id is not None) or (action == 'rest' and target is not None) or (defer_reply and action != 'ask'):
            raise ValueError('Лишние параметры действия')
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
        if action == 'invite' and (self.busy(actor, self.next_free_slot()) or state.avoiding(target or '')):
            raise ValueError('Несовместимый план или дистанция')
        if action in ('ask', 'distance') and state.actionable_situation(target or '') is None:
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
            situation = state.actionable_situation(target or '') if action in ('ask', 'distance') else None
            candidate = Candidate(action=action, target=target, event_id=event_id, situation_id=situation.id if situation else None, components={'external_input': 1})
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
                                             replies={target:PlanReply(history=[event.id])}, tick=self.next_free_slot(), history=[event.id]))
            return event
        if action == 'tell':
            assert event_id is not None
            event = self.emit('information_shared', (actor, target), 'Сообщение о наблюдавшейся встрече', causes=(event_id,), trace=trace)
            state.cooldowns[f'tell:{target}:{event_id}'] = self.world.tick
            fact = next(e for e in self.world.events if e.id == event_id)
            perceive(self.world, target, fact, actor, event.id, message=True)
            return event
        situation = state.actionable_situation(target)
        assert situation is not None
        revision = next(i for i in state.interpretations if i.id == situation.current_revision)
        # React to the latest meeting, not to a previous reply about it.
        meeting_id = next(e for e in reversed(situation.evidence) if state.knowledge[e].kind == 'meeting_occurred')
        causes = (meeting_id,)
        event = self.emit('contact_avoided' if action == 'distance' else 'question_asked',
                          (actor,) if action == 'distance' else (actor, target),
                          'Сохраняемая дистанция' if action == 'distance' else 'Вопрос о встрече без приглашения',
                          causes=causes, trace=trace, subject_event=meeting_id)
        situation.reactions.append(Reaction(event_id=event.id, revision_id=revision.id,
                                           generation=situation.generation, action=action))
        situation.reacted_generation = situation.generation
        situation.distance_active = action == 'distance'
        pending=any(r.event_id in self.world.pending_questions for r in situation.reactions)
        situation.status = 'discussing' if action=='ask' or pending else 'unresolved'
        if action == 'ask':
            if defer_reply:
                self.world.pending_questions.append(event.id)
            else:
                self.answer_question(actor, target, event, causes)
        return event

    def answer_question(self, asker: str, respondent: str, question: WorldEvent, causes: tuple[str, ...]) -> None:
        view = self.view(respondent)
        if view.person.is_player:
            self.emit('question_unanswered', (respondent, asker), 'Игрок не ответил', causes=(question.id,), claim='unanswered', subject_event=question.subject_event)
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
        claim = 'refused'
        if clarified:
            claim = 'no_prior_group_plan' if all(state.knowledge[c].context == 'no_prior_group_plan' for c in causes) else 'meeting_confirmed'
        self.emit('question_answered' if clarified else 'question_declined', (respondent, asker),
                  'Сообщение об отсутствии общего плана' if claim == 'no_prior_group_plan' else
                  'Подтверждён только факт встречи' if clarified else 'Обсуждение не состоялось',
                  causes=(question.id,), trace=trace, claim=claim, subject_event=question.subject_event)
        if clarified:
            state.energy = bounded(state.energy - 4)

    def respond(self, plan_id: str, accept: bool | None = None, *, invitee: str | None = None) -> WorldEvent:
        if accept is not None and type(accept) is not bool:
            raise ValueError('Ответ должен быть bool или None')
        plan = next((p for p in self.world.plans if p.id == plan_id), None)
        if plan is None or plan.status in ('completed','cancelled') or plan.tick < self.world.tick:
            raise ValueError('Нет актуального приглашения')
        invitee = invitee or plan.invitee
        if invitee not in plan.replies or plan.replies[invitee].status != 'pending':
            raise ValueError('Приглашение уже обработано или адресат неизвестен')
        reply=plan.replies[invitee]
        view = self.view(invitee)
        if view.person.is_player and accept is None:
            raise ValueError('Игрок должен сам выбрать ответ')
        state = self.world.states[invitee]
        rel = self.world.relationships.get(f'{invitee}:{plan.organizer}')
        available = not self.busy(invitee, plan.tick) and state.energy >= 15 and not state.avoiding(plan.organizer)
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
                          (invitee, plan.organizer), 'Согласие' if accepted else 'Отказ',
                          causes=(reply.history[0],), trace=trace)
        reply.status = 'accepted' if accepted else 'declined'
        reply.history.append(event.id)
        plan.status = 'accepted' if plan.accepted else 'pending' if any(r.status=='pending' for r in plan.replies.values()) else 'declined'
        plan.history.append(event.id)
        return event

    def share_information(self, sender: str, receiver: str, event_id: str) -> WorldEvent | None:
        if sender not in self.world.states or receiver not in self.world.states or event_id not in self.world.states[sender].knowledge:
            raise ValueError('Участник не знает факт')
        if event_id in self.world.states[receiver].knowledge:
            return None
        return self.act(sender, 'tell', receiver, event_id)

    def meet(self, plan: SocialPlan) -> None:
        if plan not in self.world.plans or plan.status in ('completed','cancelled') or plan.tick != self.world.tick:
            raise ValueError('Встреча не соответствует принятому плану и времени')
        for person, reply in plan.replies.items():
            if reply.status == 'pending':
                event=self.emit('invitation_expired',(plan.organizer,person),'Ответ не получен',causes=(reply.history[0],))
                reply.status='expired';reply.history.append(event.id);plan.history.append(event.id)
        if not plan.accepted:
            event=self.emit('plan_cancelled',(plan.organizer,),'Никто не согласился',causes=tuple(plan.history))
            plan.status='cancelled';plan.history.append(event.id)
            return
        members=(plan.organizer,*plan.accepted)
        event = self.emit('meeting_occurred', members, 'Совместная прогулка',
                          causes=tuple(plan.history), location=plan.location, organizer=plan.organizer, context=self.world.scenario.meeting_context)
        plan.status = 'completed';plan.history.append(event.id)
        for actor in members:
            state=self.world.states[actor]
            state.location=plan.location;state.energy=bounded(state.energy-15);state.loneliness=bounded(state.loneliness-30)
            for other in members:
                if actor==other:continue
                key=f'{actor}:{other}'
                if key not in self.world.relationships:
                    continue  # Co-presence is a fact, not consent to a personal introduction.
                rel=self.world.relationships[key]
                rel.shared_meetings+=1
                rel.liking=bounded(rel.liking+2);rel.trust=bounded(rel.trust+1)
                rel.closeness=bounded(rel.closeness+3*(1-rel.closeness/120))
                rel.perceived_reciprocity=bounded(rel.perceived_reciprocity+2)
                remember(self.world,actor,event.id,other,'companionship',30,False)

    def prepare_social(self, actor: str, action: ActionKind, targets: tuple[str,...], trace: DecisionTrace | None, cooldown: int) -> DecisionTrace:
        if (actor not in self.world.states or not targets or len(set(targets))!=len(targets) or actor in targets
            or any(f'{actor}:{p}' not in self.world.relationships for p in targets)):
            raise ValueError('Нужно собственное знакомство со всеми участниками')
        state=self.world.states[actor]
        key=f'{action}:'+':'.join(sorted(targets))
        if (self.busy(actor,self.world.tick) or state.last_action_tick==self.world.tick or state.energy<20
            or self.world.tick-state.cooldowns.get(key,-1000)<cooldown or any(state.avoiding(p) for p in targets)):
            raise ValueError('Нет времени, энергии или действует cooldown/дистанция')
        if trace is not None and (trace.id!=f't{len(self.world.traces)+1:06d}' or trace.actor!=actor or trace.tick!=self.world.tick
            or trace.selected.action!=action or trace.selected.targets!=targets or set(trace.known_events)!=set(state.knowledge)):
            raise ValueError('Trace не соответствует социальной инициативе')
        if trace is None:
            trace=choose(self.view(actor),[Candidate(action=action,targets=targets,components={'external_input':1})],[],self.rng,
                         self.world.scenario.weights,f't{len(self.world.traces)+1:06d}',randomize=False)
        self.world.traces.append(trace)
        state.last_action_tick=self.world.tick;state.energy=bounded(state.energy-12);state.cooldowns[key]=self.world.tick
        return trace

    def invite_group(self, actor: str, invitees: list[str], *, trace: DecisionTrace | None = None) -> SocialPlan:
        targets=tuple(sorted(invitees))
        if not 1<=len(targets)<=4 or self.busy(actor,self.next_free_slot()):
            raise ValueError('Несовместимый план или размер группы')
        state=self.world.states.get(actor)
        if state is None or any(self.world.tick-state.cooldowns.get(f'invite:{p}',-1000)<8 for p in targets):
            raise ValueError('Приглашение адресату на cooldown')
        trace=self.prepare_social(actor,'invite_group',targets,trace,8)
        root=self.emit('group_invitation_proposed',(actor,),'Намерение собрать знакомых',trace=trace) if len(targets)>1 else None
        history=[] if root is None else [root.id]
        replies={}
        for person in targets:
            event=self.emit('invitation_sent',(actor,person),f'Встреча в слоте {self.next_free_slot()}',causes=(root.id,) if root else (),trace=trace if root is None else None)
            history.append(event.id);replies[person]=PlanReply(history=[event.id]);state.cooldowns[f'invite:{person}']=self.world.tick
        plan=SocialPlan(id=f'p{len(self.world.plans)+1:06d}',organizer=actor,replies=replies,tick=self.next_free_slot(),history=history)
        self.world.plans.append(plan)
        return plan

    def introduce(self, mediator: str, first: str, second: str, *, answers: dict[str,bool] | None = None, trace: DecisionTrace | None = None) -> WorldEvent:
        targets=tuple(sorted((first,second)))
        if answers is not None and (not set(answers)<=set(targets) or any(type(v) is not bool for v in answers.values())):
            raise ValueError('Некорректный внешний ответ')
        if answers is not None:
            for person,accepted in answers.items():
                state=self.world.states.get(person)
                if accepted and (state is None or self.busy(person,self.world.tick) or state.energy<10 or state.avoiding(mediator)):
                    raise ValueError('Явное согласие несовместимо с доступностью')
        trace=self.prepare_social(mediator,'introduce',targets,trace,32)
        proposal=self.emit('introduction_proposed',(mediator,*targets),'Предложение знакомства через общего знакомого',trace=trace)
        if all(f'{a}:{b}' in self.world.relationships for a,b in (targets,tuple(reversed(targets)))):
            return self.emit('introduction_already_known',(mediator,*targets),'Участники уже знакомы',causes=(proposal.id,))
        agreed=[];responses=[]
        for person in targets:
            view=self.view(person);t=view.person.personality.traits
            explicit=answers.get(person) if answers is not None else None
            if view.person.is_player and explicit is None:
                e=self.emit('introduction_unanswered',(person,mediator),'Игрок не ответил',causes=(proposal.id,));responses.append(e.id);continue
            rel=self.world.relationships.get(f'{person}:{mediator}')
            available=not self.busy(person,self.world.tick) and view.state.energy>=10 and not view.state.avoiding(mediator)
            candidates=[Candidate(action='decline_introduction',target=mediator,components={'personality':t['autonomy'],'effort':1})]
            if available:
                candidates.append(Candidate(action='accept_introduction',target=mediator,components={'personality':t['novelty']+t['sociability'],'relationship':rel.trust/20 if rel else 0,'need':view.state.loneliness/15,'effort':-2}))
            if explicit is not None:candidates=[c for c in candidates if c.action==('accept_introduction' if explicit else 'decline_introduction')]
            response=choose(view,candidates,[] if available else ['Знакомство: нет доступности'],self.rng,self.world.scenario.weights,
                            f't{len(self.world.traces)+1:06d}',randomize=explicit is None)
            self.world.traces.append(response)
            accepted=response.selected.action=='accept_introduction'
            e=self.emit('introduction_accepted' if accepted else 'introduction_rejected',(person,mediator),'Согласие' if accepted else 'Отказ',causes=(proposal.id,),trace=response)
            responses.append(e.id)
            if accepted:agreed.append(person)
        success=len(agreed)==2
        event=self.emit('introduction_completed' if success else 'introduction_declined',(mediator,*targets),
                        'Личное знакомство' if success else 'Знакомство не состоялось',causes=(proposal.id,*responses))
        if success:
            for source,target in (targets,tuple(reversed(targets))):
                key=f'{source}:{target}'
                if key not in self.world.relationships:self.world.relationships[key]=Relationship(source=source,target=target,liking=30,trust=30,closeness=5,origin_event=event.id)
        return event

    def execute_decision(self, trace: DecisionTrace) -> WorldEvent:
        selected=trace.selected
        if selected.action=='introduce':
            return self.introduce(trace.actor,*selected.targets,trace=trace)
        if selected.action=='invite_group':
            plan=self.invite_group(trace.actor,list(selected.targets),trace=trace)
            return next(e for e in self.world.events if e.id==plan.history[0])
        return self.act(trace.actor,selected.action,selected.target,selected.event_id,trace)

    def step(self) -> None:
        self.recover()
        for state in self.world.states.values():
            state.location = 'school' if self.world.tick % 4 < 2 else 'home'
        if self.world.tick % 4 >= 2:
            for question_id in self.world.pending_questions[:]:
                question = next(e for e in self.world.events if e.id == question_id)
                self.answer_question(question.participants[0], question.participants[1], question, question.causes)
                self.world.pending_questions.remove(question_id)
            people = {p.id: p for p in self.world.scenario.people}
            for plan in self.world.plans:
                if plan.status in ('completed','cancelled'):continue
                for person,reply in plan.replies.items():
                    if reply.status=='pending' and not people[person].is_player:self.respond(plan.id,invitee=person)
                if plan.tick<=self.world.tick:self.meet(plan)
            actors = sorted(p.id for p in self.world.scenario.people if not p.is_player)
            self.rng.shuffle(actors)
            for actor in actors:
                if self.busy(actor, self.world.tick) or self.world.states[actor].last_action_tick == self.world.tick:
                    continue
                trace = decide(self.view(actor), self.rng, self.world.scenario.weights, f't{len(self.world.traces)+1:06d}')
                self.execute_decision(trace)
        self.world.tick += 1
        if self.world.tick % 4 == 0:update_groups(self.world)

    def run_days(self, days: int) -> None:
        if days < 0:
            raise ValueError('Число дней должно быть неотрицательным')
        for _ in range(days * 4):
            self.step()
