"""Типизированные частные обращения поверх событий и собственных знаний NPC."""
from __future__ import annotations

from typing import TYPE_CHECKING

from social_sim.cognition import bounded, remember
from social_sim.memory import relevant_memories
from social_sim.models import (
    Candidate, DecisionTrace, InteractionEffect, Interpretation, KnowledgeRecord,
    SocialInteraction, SocialMessage, WorldState,
)

if TYPE_CHECKING:
    from social_sim.decision import ActorView
    from social_sim.engine import Simulation

OFFER_COST = 8
REPLY_COST = 4
REPLY_LIMIT = 2
EXPIRY_SLOTS = 16


def received_report(view: ActorView, target: str, subject: str, *, apology: bool = False) -> KnowledgeRecord | None:
    return next((k for k in reversed(list(view.state.knowledge.values()))
                 if k.social_message is not None and k.social_message.kind in ('discontent','support_request')
                 and k.social_message.subject_event == subject and k.participants[0] == target
                 and view.person.id in k.participants and k.source == target
                 and (not apology or k.social_message.about == view.person.id)), None)


def known_reason(view: ActorView, message: SocialMessage, target: str) -> str | None:
    if message.kind in ('discontent','support_request'):
        situation = next((s for s in view.state.situations.values() if message.subject_event in s.episodes
                          and s.target == message.about and s.episodes[message.subject_event].emotion > 10), None)
        if situation is None or message.subject_event not in view.state.knowledge:
            return None
        if message.kind == 'discontent' and target != message.about:
            return None
        return message.subject_event
    report = received_report(view,target,message.subject_event,apology=message.kind=='apology')
    if report is None or report.social_message is None or report.social_message.about != message.about:
        return None
    if message.content == 'acknowledge_act':
        fact = view.state.knowledge.get(message.subject_event)
        if fact is None or view.person.id not in fact.participants:
            return None
    return report.event_id


def can_retry(view: ActorView, kind: str, target: str, subject: str) -> bool:
    offers = [k for k in view.state.knowledge.values() if k.kind == 'social_interaction_offered'
              and k.participants == (view.person.id,target) and k.social_message is not None
              and k.social_message.kind == kind and k.social_message.subject_event == subject]
    if not offers:
        return True
    last = max(offers,key=lambda k:k.acquired_tick)
    if not any(k.reply_to==last.event_id for k in view.state.knowledge.values()):
        return False
    # A reply is not a new reason. A real later shared experience can justify another attempt.
    return any(k.kind == 'meeting_occurred' and {view.person.id,target} <= set(k.participants)
               and k.acquired_tick > last.acquired_tick for k in view.state.knowledge.values())


def offer(sim: Simulation, actor: str, target: str, message: SocialMessage, *,
          defer_reply: bool = False, trace: DecisionTrace | None = None) -> SocialInteraction:
    from social_sim.decision import choose
    if actor not in sim.world.states or target not in sim.world.states or actor == target or f'{actor}:{target}' not in sim.world.relationships:
        raise ValueError('Обращение требует знакомства с адресатом')
    view = sim.view(actor)
    reason = known_reason(view,message,target)
    if message.about not in sim.world.states or reason is None:
        raise ValueError('Нет известного основания для обращения')
    if (sim.busy(actor,sim.world.tick) or view.state.last_action_tick == sim.world.tick or view.state.energy < OFFER_COST
        or not can_retry(view,message.kind,target,message.subject_event)
        or any(x.status=='pending' and x.initiator==actor and x.recipient==target and x.message.kind==message.kind
               and x.message.subject_event==message.subject_event for x in sim.world.interactions.values())):
        raise ValueError('Нет времени/энергии или нового основания для повторного обращения')
    if trace is not None and (trace.id!=f't{len(sim.world.traces)+1:06d}' or trace.actor!=actor
        or trace.tick!=sim.world.tick or trace.selected.social_message!=message or trace.selected.target!=target
        or trace.selected.action!=message.kind or trace.event_boundary!=len(sim.world.events)
        or set(trace.known_events)!=set(view.state.knowledge)):
        raise ValueError('Trace не соответствует обращению')
    if trace is None:
        trace=choose(view,[Candidate(action=message.kind,target=target,event_id=reason,social_message=message,
                                    components={'external_input':1})],[],sim.rng,sim.world.scenario.weights,
                     f't{len(sim.world.traces)+1:06d}',randomize=False)
    sim.world.traces.append(trace)
    state=sim.world.states[actor]
    state.last_action_tick=sim.world.tick
    state.energy=bounded(state.energy-OFFER_COST)
    event=sim.emit('social_interaction_offered',(actor,target),'Частное типизированное обращение',
                   causes=(reason,),trace=trace,subject_event=message.subject_event,social_message=message)
    interaction=SocialInteraction(id=f'x{len(sim.world.interactions)+1:06d}',initiator=actor,recipient=target,
                                  message=message,created_tick=sim.world.tick,expires_tick=sim.world.tick+EXPIRY_SLOTS,
                                  offered_event=event.id)
    sim.world.interactions[interaction.id]=interaction
    if message.kind in ('support','apology'):
        remember(sim.world,target,event.id,actor,'repair_attempt',20,False)
    if not defer_reply and not sim.view(target).person.is_player and response_available(sim,target):
        respond(sim,interaction.id,target)
    return interaction


def response_available(sim: Simulation, actor: str) -> bool:
    state=sim.world.states[actor]
    return not sim.busy(actor,sim.world.tick) and state.energy>=REPLY_COST and (state.reply_tick!=sim.world.tick or state.reply_count<REPLY_LIMIT)


def response_candidates(view: ActorView, message: SocialMessage, sender: str) -> list[Candidate]:
    t=view.person.personality.traits
    rel=next((r for r in view.relationships if r.target==sender),None)
    trust=rel.trust if rel else 0
    emotion=sum(s.episodes[message.subject_event].emotion for s in view.state.situations.values() if message.subject_event in s.episodes)
    if message.kind in ('discontent','support_request'):
        choices=[Candidate(action='acknowledge',target=sender,components={'personality':t['tact']+t['reconciliation_readiness'],
                    'relationship':trust/25,'effort':-2}),
                 Candidate(action='decline',target=sender,components={'personality':t['autonomy']+t['privacy_priority'],'effort':-1})]
        if message.kind=='discontent':
            choices.append(Candidate(action='disagree',target=sender,components={'personality':t['directness']+t['honesty_priority']-t['tact'],
                            'relationship':(100-trust)/30,'effort':-2}))
        return choices
    memories=relevant_memories(view.state,sender,'exclusion',view.tick)
    return [Candidate(action='accept',target=sender,memory_ids=tuple(m.id for m in memories),components={
                'personality':t['openness']+t['reconciliation_readiness'],'relationship':trust/15,
                'emotion':-emotion/25,'memory':-sum(m.significance*m.confidence/100 for m in memories if m.valence=='negative'), 'effort':-1}),
            Candidate(action='decline',target=sender,components={'personality':t['autonomy']+t['grudge_persistence']/2,
                'relationship':(100-trust)/30,'emotion':emotion/25,'effort':-1}),
            Candidate(action='defer',target=sender,components={'personality':t['stability']/2,
                'emotion':emotion/12,'effort':-1})]


def respond(sim: Simulation, interaction_id: str, actor: str, response: str | None = None) -> None:
    from social_sim.decision import choose
    x=sim.world.interactions.get(interaction_id)
    if x is None or x.recipient!=actor or x.status!='pending' or sim.world.tick>=x.expires_tick:
        raise ValueError('Нет ожидающего обращения для этого адресата')
    view=sim.view(actor)
    if view.person.is_player and response is None:
        raise ValueError('Игрок отвечает только явным вводом')
    if not response_available(sim,actor):
        raise ValueError('Нет времени, энергии или бюджета коротких ответов')
    if x.offered_event not in view.state.knowledge:
        raise ValueError('Адресат не получил обращение')
    candidates=response_candidates(view,x.message,x.initiator)
    if response is not None:
        candidates=[c for c in candidates if c.action==response]
        if not candidates:raise ValueError('Недопустимый вид ответа')
    trace=choose(view,candidates,['Короткий ответ стоит 4 энергии; максимум два в слот'],sim.rng,
                 sim.world.scenario.weights,f't{len(sim.world.traces)+1:06d}',randomize=response is None)
    sim.world.traces.append(trace)
    state=sim.world.states[actor]
    if state.reply_tick!=sim.world.tick:state.reply_tick=sim.world.tick;state.reply_count=0
    state.reply_count+=1;state.energy=bounded(state.energy-REPLY_COST)
    selected=trace.selected.action
    event=sim.emit('social_interaction_answered',(actor,x.initiator),'Самостоятельный ответ на обращение',
                   causes=(x.offered_event,),trace=trace,subject_event=x.message.subject_event,social_response=selected)
    x.response_event=event.id
    x.status='declined' if selected=='decline' else 'deferred' if selected=='defer' else 'completed'
    apply_effect(sim,x,event.id,selected)


def apply_effect(sim: Simulation, x: SocialInteraction, event_id: str, response: str) -> None:
    if response in ('decline','defer') or x.message.kind=='support_request':return
    actor=x.initiator if x.message.kind=='discontent' else x.recipient
    target=x.recipient if x.message.kind=='discontent' else x.initiator
    state=sim.world.states[actor]
    situation=next((s for s in state.situations.values() if x.message.subject_event in s.episodes),None)
    if situation is None:return
    assessment=situation.episodes[x.message.subject_event]
    relationship=sim.world.relationships[f'{actor}:{target}']
    before=(relationship.trust,relationship.tension,assessment.emotion)
    token=f'{actor}:{target}:{x.message.kind}:{x.message.subject_event}' if x.message.kind in ('support','apology') else None
    if token is not None and any(e.token==token for e in sim.world.effects):return
    if x.message.kind=='discontent':
        if response=='disagree':
            relationship.tension=bounded(relationship.tension+1/(1+relationship.closeness/50))
            reason='Несовпадение заявленной интерпретации и ответа; умысел не установлен'
        else:
            assessment.emotion=bounded(assessment.emotion*(1-0.05*relationship.trust/100))
            reason='Признано переживание; вина и умысел не устанавливаются'
    else:
        cap=5 if x.message.kind=='apology' else 6
        assessment.emotion=bounded(assessment.emotion-min(cap,assessment.emotion*0.2))
        relationship.trust=bounded(relationship.trust+min(2,(100-relationship.trust)/50))
        if x.message.kind=='apology':relationship.tension=bounded(relationship.tension-min(3,relationship.tension*0.2))
        reason='Принята поддержка конкретного эпизода' if x.message.kind=='support' else 'Принято извинение; доверие восстановлено частично, контакт не навязывается'
        remember(sim.world,actor,event_id,target,'exclusion',30,False)
    effect=InteractionEffect(interaction_id=x.id,event_id=event_id,actor=actor,target=target,subject_event=x.message.subject_event,
        trust_delta=round(relationship.trust-before[0],6),tension_delta=round(relationship.tension-before[1],6),emotion_delta=round(assessment.emotion-before[2],6),reason=reason,token=token)
    sim.world.effects.append(effect)
    old=next(i for i in state.interpretations if i.id==situation.current_revision)
    situation.evidence=(*situation.evidence,event_id);situation.updated_tick=sim.world.tick
    confidence=max(e.confidence for e in situation.episodes.values())
    revision=Interpretation(id=f'i:{actor}:{len(state.interpretations)+1}',situation_id=situation.id,tick=sim.world.tick,
        evidence=situation.evidence,event_id=event_id,target=situation.target,emotion_after=situation.emotion,
        trust_delta=effect.trust_delta if target==situation.target else 0,tension_delta=effect.tension_delta if target==situation.target else 0,
        kind='excluded' if confidence>=0.3 else 'neutral',confidence=confidence,belief_ids=old.belief_ids,reason=reason)
    state.interpretations.append(revision);situation.current_revision=revision.id


def process_pending(sim: Simulation) -> None:
    for x in list(sim.world.interactions.values()):
        if x.status!='pending':continue
        if sim.world.tick>=x.expires_tick:
            event=sim.emit('social_interaction_expired',(x.recipient,x.initiator),'Срок ответа истёк',
                           causes=(x.offered_event,),subject_event=x.message.subject_event)
            x.status='expired';x.response_event=event.id
        elif not sim.view(x.recipient).person.is_player and response_available(sim,x.recipient):
            respond(sim,x.id,x.recipient)


def validate_interactions(world: WorldState) -> None:
    # Reference and lifecycle checks run before loading or saving a world.
    ids=set(world.states)
    events={e.id:e for e in world.events}
    traces={t.id:t for t in world.traces}
    for n,(key,x) in enumerate(world.interactions.items(),1):
        if key!=x.id or key!=f'x{n:06d}' or x.initiator not in ids or x.recipient not in ids or x.initiator==x.recipient:
            raise ValueError('Некорректное обращение')
        if x.message.about not in ids or x.message.subject_event not in events or x.offered_event not in events:
            raise ValueError('Неизвестный предмет обращения')
        offered=events[x.offered_event]
        if (x.message.subject_event not in events or events[x.message.subject_event].kind!='meeting_occurred'
            or events[x.message.subject_event].tick>x.created_tick or offered.subject_event!=x.message.subject_event):
            raise ValueError('Неверный предмет или время обращения')
        if offered.social_message!=x.message or offered.participants!=(x.initiator,x.recipient) or offered.kind!='social_interaction_offered' or offered.tick!=x.created_tick or x.expires_tick!=x.created_tick+EXPIRY_SLOTS:
            raise ValueError('Обращение не соответствует сообщению')
        if x.status=='pending' and world.tick>x.expires_tick:raise ValueError('Просроченное обращение')
        offer_trace=traces.get(offered.trace_id or '')
        if offer_trace is None or offer_trace.selected.action!=x.message.kind or offer_trace.selected.social_message!=x.message or offer_trace.selected.target!=x.recipient or offer_trace.selected.event_id not in offered.causes:
            raise ValueError('Trace обращения не соответствует сообщению')
        if (x.status=='pending')!=(x.response_event is None):raise ValueError('Несогласованный статус ответа')
        if x.response_event is not None:
            reply=events.get(x.response_event)
            if reply is None or reply.participants!=(x.recipient,x.initiator) or reply.causes!=(x.offered_event,) or reply.subject_event!=x.message.subject_event:
                raise ValueError('Некорректный ответ обращения')
            if x.status=='expired':
                if reply.kind!='social_interaction_expired' or reply.tick<x.expires_tick or reply.social_response is not None:
                    raise ValueError('Неверное истечение обращения')
            else:
                allowed={'acknowledge','decline'} if x.message.kind=='support_request' else {'acknowledge','disagree','decline'} if x.message.kind=='discontent' else {'accept','decline','defer'}
                if reply.kind!='social_interaction_answered' or reply.social_response not in allowed or not x.created_tick<=reply.tick<x.expires_tick:
                    raise ValueError('Несовместимый ответ')
                status='declined' if reply.social_response=='decline' else 'deferred' if reply.social_response=='defer' else 'completed'
                if x.status!=status:raise ValueError('Статус не соответствует ответу')
                reply_trace=traces.get(reply.trace_id or '')
                if reply_trace is None or reply_trace.selected.action!=reply.social_response or reply_trace.selected.target!=x.initiator:
                    raise ValueError('Trace ответа не соответствует содержанию')
    for state in world.states.values():
        if not -1<=state.reply_tick<=world.tick:raise ValueError('Бюджет ответа из будущего')
    tokens=set()
    for effect in world.effects:
        owner=world.interactions.get(effect.interaction_id)
        if owner is None or effect.event_id!=owner.response_event or effect.subject_event!=owner.message.subject_event or effect.actor not in (owner.initiator,owner.recipient) or effect.target not in (owner.initiator,owner.recipient):
            raise ValueError('Некорректная причина эффекта')
        expected_actor=owner.initiator if owner.message.kind=='discontent' else owner.recipient
        expected_target=owner.recipient if owner.message.kind=='discontent' else owner.initiator
        response=events[effect.event_id].social_response
        if effect.actor!=expected_actor or effect.target!=expected_target or owner.status!='completed' or owner.message.kind=='support_request' or response in ('decline','defer'):
            raise ValueError('Эффект применён не к тому направлению или исходу')
        expected_token=f'{effect.actor}:{effect.target}:{owner.message.kind}:{effect.subject_event}' if owner.message.kind in ('support','apology') else None
        if effect.token!=expected_token:raise ValueError('Некорректный токен эффекта')
        if effect.token:
            if effect.token in tokens:raise ValueError('Повторный эффект по одному поводу')
            tokens.add(effect.token)


def initiative_candidates(view: ActorView) -> list[Candidate]:
    if view.state.energy<OFFER_COST:return []
    state=view.state;t=view.person.personality.traits
    contacts={r.target:r for r in view.relationships}
    connect=sum(g.priority for g in view.person.personality.goals if g.kind=='connect')
    candidates=[]
    for situation in state.situations.values():
        for root,assessment in situation.episodes.items():
            if assessment.emotion<=10:continue
            for kind in ('discontent','support_request'):
                targets=(situation.target,) if kind=='discontent' else tuple(p for p in contacts if p!=situation.target)
                for target in targets:
                    if not can_retry(view,kind,target,root) or (kind=='support_request' and t['openness']<2):continue
                    message=SocialMessage.model_validate(dict(kind=kind,subject_event=root,about=situation.target,content='hurt_report'))
                    fit: float=t['directness']+t['recognition'] if kind=='discontent' else t['openness']+t['closeness']
                    candidates.append(Candidate(action=message.kind,target=target,event_id=root,social_message=message,
                        components={'emotion':assessment.emotion/8,'personality':fit,'goal':connect*2,'effort':-3}))
    seen=set()
    for record in state.knowledge.values():
        report=record.social_message
        if report is None or report.kind not in ('discontent','support_request') or record.participants[0]==view.person.id:
            continue
        target=record.participants[0]
        if target not in contacts:continue
        for kind in ('support','apology'):
            if kind=='apology' and report.about!=view.person.id:continue
            key=(kind,target,report.subject_event)
            if key in seen or not can_retry(view,kind,target,report.subject_event):continue
            seen.add(key)
            message=SocialMessage.model_validate(dict(kind=kind,subject_event=report.subject_event,about=report.about,
                 content='support_offer' if kind=='support' else 'regret_hurt'))
            reason=known_reason(view,message,target)
            if reason is None:continue
            fit=t['tact']+t['reconciliation_readiness']+(t['honesty_priority']/2 if kind=='apology' else t['openness']/2)
            candidates.append(Candidate(action=message.kind,target=target,event_id=reason,social_message=message,
                components={'personality':fit,'relationship':contacts[target].trust/25,'goal':connect*2,'effort':-3}))
    return candidates
