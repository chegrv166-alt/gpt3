"""Диагностические максимальные клики с подтверждением во времени."""
from collections import Counter
from itertools import combinations

from social_sim.models import GroupChange, GroupRecord, WorldState


def active_edges(world: WorldState) -> dict[tuple[str,str], tuple[str,...]]:
    settings=world.scenario.group_settings
    evidence: dict[tuple[str,str],list[str]]={}
    for event in world.events:
        if event.kind=='meeting_occurred' and world.tick-settings.window_slots<=event.tick<=world.tick:
            for a,b in combinations(sorted(event.participants),2):
                evidence.setdefault((a,b),[]).append(event.id)
    result={}
    for pair,events in evidence.items():
        a,b=pair
        ab,ba=world.relationships.get(f'{a}:{b}'),world.relationships.get(f'{b}:{a}')
        if ab and ba and min(ab.closeness,ba.closeness)>=settings.closeness and len(events)>=settings.meetings:
            result[pair]=tuple(events)
    return result


def detect_groups(world: WorldState) -> tuple[tuple[str,...],...]:
    edges=active_edges(world)
    neighbors:dict[str,set[str]]={}
    for a,b in edges:
        neighbors.setdefault(a,set()).add(b);neighbors.setdefault(b,set()).add(a)
    result=[]
    def visit(r:set[str], p:set[str], x:set[str]) -> None:
        if not p and not x:
            if len(r)>=3:result.append(tuple(sorted(r)))
            return
        for node in sorted(p.copy()):
            visit(r|{node},p&neighbors[node],x&neighbors[node])
            p.remove(node);x.add(node)
    visit(set(),set(neighbors),set())
    return tuple(sorted(result))


def update_groups(world: WorldState) -> None:
    tracker=world.groups
    if world.tick<=tracker.last_tick:return
    tracker.last_tick=world.tick
    settings=world.scenario.group_settings
    detected=detect_groups(world)
    edges=active_edges(world)
    used:set[str]=set()
    seen_keys=set()
    def record(group: GroupRecord, kind: str, previous: tuple[str,...]) -> None:
        evidence=tuple(sorted({e for pair in combinations(group.members,2) for e in edges.get(pair,())}))
        tracker.history.append(GroupChange(tick=world.tick,group_id=group.id,kind=kind,members=group.members,
            previous_members=previous,evidence=evidence,reason=f'Взаимная близость >= {settings.closeness:g}; встреч >= {settings.meetings} за {settings.window_slots} слотов; подтверждений {settings.confirmations}'))
    reserved={g.id:g.members for g in tracker.groups if g.members in detected}
    for members in detected:
        candidates=[g for g in tracker.groups if g.id not in used and (g.id not in reserved or reserved[g.id]==members) and len(set(g.members)&set(members))>=2]
        candidates.sort(key=lambda g:(g.members!=members,-len(set(g.members)&set(members)),
            -len(set(g.members)&set(members))/len(set(g.members)|set(members)),g.id))
        if not candidates:
            key=':'.join(members);seen_keys.add(key)
            tracker.candidates[key]=tracker.candidates.get(key,0)+1
            if tracker.candidates[key]>=settings.confirmations:
                g=GroupRecord(id=f'g{len(tracker.groups)+1:04d}',members=members)
                tracker.groups.append(g);used.add(g.id);record(g,'formed',())
                tracker.candidates.pop(key)
        else:
            g=candidates[0];used.add(g.id);g.missing_checks=0
            if g.members==members and g.active:
                g.proposed_members=();g.confirmations=0
            else:
                g.confirmations=g.confirmations+1 if g.proposed_members==members else 1
                g.proposed_members=members
                if g.confirmations>=settings.confirmations:
                    previous=g.members;g.members=members;g.active=True;g.proposed_members=();g.confirmations=0
                    record(g,'changed',previous)
    tracker.candidates={key:value for key,value in tracker.candidates.items() if key in seen_keys}
    for g in tracker.groups:
        if g.id not in used and g.active:
            g.missing_checks+=1
            g.proposed_members=();g.confirmations=0
            if g.missing_checks>=settings.confirmations:
                g.active=False;record(g,'inactive',g.members)
