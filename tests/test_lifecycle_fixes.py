import json
import subprocess
import sys

import pytest

from cognitive_helpers import school, episode
from social_sim.models import WorldState
from social_sim.storage import save_snapshot, load_snapshot


def pending():
    s = school(explanation=True)
    episode(s)
    second = episode(s,tick=10,disclose=False)
    s.act('n3','ask','n1',defer_reply=True)
    return s,second


def test_new_episode_during_pending_question_keeps_world_valid():
    s,e = pending()
    s.share_information('n2','n3',e.id)
    WorldState.model_validate(s.world.model_dump())
    assert len(s.world.pending_questions)==1


def test_snapshot_after_new_evidence_during_discussion_resumes_identically(tmp_path):
    s,e = pending()
    s.share_information('n2','n3',e.id)
    save_snapshot(s,tmp_path/'state.json')
    b=load_snapshot(tmp_path/'state.json')
    for x in (s,b): x.run_days(10)
    assert s.world==b.world and s.rng.getstate()==b.rng.getstate()


def test_new_negative_episode_preserves_active_distance():
    s=school()
    episode(s)
    s.act('n3','distance','n1')
    episode(s,tick=10)
    assert s.world.states['n3'].avoiding('n1')


def test_reply_to_old_episode_does_not_resolve_new_episode_implicitly():
    s,e=pending()
    s.share_information('n2','n3',e.id)
    confidence=s.world.states['n3'].interpretations[-1].confidence
    s.step()
    assert s.world.states['n3'].interpretations[-1].confidence >= confidence
    assert next(iter(s.world.states['n3'].situations.values())).status != 'completed'


def test_distance_can_end_for_documented_reason():
    s=school();episode(s);s.act('n3','distance','n1')
    for _ in range(300): s.recover()
    assert not s.world.states['n3'].avoiding('n1')


def test_repeated_delivery_during_discussion_is_idempotent():
    s,e=pending();s.share_information('n2','n3',e.id)
    before=s.world.model_dump_json()
    s.share_information('n2','n3',e.id)
    assert s.world.model_dump_json()==before


@pytest.mark.parametrize('value',[None,[], 'str',1,True,{}])
@pytest.mark.parametrize('field',['root','world'])
def test_snapshot_wrong_shape_is_a_clean_error(tmp_path,value,field):
    path=tmp_path/'bad.json'
    path.write_text(json.dumps(value if field=='root' else {'world':value}))
    with pytest.raises(ValueError): load_snapshot(path)
    out=tmp_path/'out'
    p=subprocess.run([sys.executable,'-m','social_sim','resume','--snapshot',str(path),'--days','1','--output',str(out)],capture_output=True,text=True)
    assert p.returncode==2 and 'Traceback' not in p.stderr and not out.exists()


def test_new_reaction_does_not_orphan_previous_pending_question():
    s,e=pending();s.share_information('n2','n3',e.id)
    s.world.tick=18;s.act('n3','distance','n1')
    assert s.world.states['n3'].avoiding('n1')
    WorldState.model_validate(s.world.model_dump())
    assert s.world.pending_questions
