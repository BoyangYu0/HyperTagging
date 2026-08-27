import hashlib, os, stat, sys
from pathlib import Path
import pytest
sys.path.insert(0,'.')
from scripts.reconstruction_hpo._v13_core import VerificationError,jcs,load_spec,validate_state_integrity

ROOT=Path(__file__).resolve().parents[1]; SPEC=load_spec(ROOT)
AUTH={k:False for k in ('submission_authorized','execution_authorized','scheduler_authorized','payload_access_authorized','scientific_execution_authorized','root_final_go')}
DENIAL={k:False for k in ('sealed_test_used','stress_used','restricted_raw_used','restricted_source_used','train_loss_used')}
TARGETS=('checkpoint','model_state','event_manifest','normalized_tensor_manifest','batch_plan','runtime_manifest')

def receipt(tmp_path,phase='before'):
    tmp_path.mkdir(parents=True,exist_ok=True)
    targets={}
    for name in TARGETS:
        path=tmp_path/f'{name}.bin'; path.write_bytes(name.encode()); path.chmod(0o444); st=path.stat()
        targets[name]={'absolute_path':str(path.resolve()),'st_dev':st.st_dev,'st_ino':st.st_ino,'owner_uid':st.st_uid,'mode':stat.S_IMODE(st.st_mode),'byte_length':st.st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    out={'schema':'state-integrity-receipt-v12','version':1,'receipt_id':'synthetic','block_id':'q32_A','phase':phase,'targets':targets,'digest':'','authorization':dict(AUTH),'usage_denials':dict(DENIAL)}
    out['digest']=hashlib.sha256(jcs({k:v for k,v in out.items() if k!='digest'})).hexdigest(); return out

def test_state_integrity_reopens_all_six_targets_and_accepts_before_after(tmp_path):
    before=receipt(tmp_path/'before','before'); after=receipt(tmp_path/'after','after')
    validate_state_integrity(SPEC,before); validate_state_integrity(SPEC,after)

@pytest.mark.parametrize('mutation',('runtime_missing','content','mode','digest','phase'))
def test_state_integrity_exact_negative_mutations(tmp_path,mutation):
    value=receipt(tmp_path)
    if mutation=='runtime_missing': value['targets'].pop('runtime_manifest')
    elif mutation=='content': Path(value['targets']['checkpoint']['absolute_path']).chmod(0o644); Path(value['targets']['checkpoint']['absolute_path']).write_bytes(b'changed'); Path(value['targets']['checkpoint']['absolute_path']).chmod(0o444)
    elif mutation=='mode': value['targets']['checkpoint']['mode']=0o666
    elif mutation=='digest': value['digest']='0'*64
    else: value['phase']='terminal'
    with pytest.raises(VerificationError): validate_state_integrity(SPEC,value)
