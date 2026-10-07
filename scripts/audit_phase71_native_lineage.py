"""Audit native data, tensor intervention execution and context exposure on CPU."""
import argparse, datetime, hashlib, json
from pathlib import Path
import torch
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(2)
    p = json.loads((a.source/'configs/reconstruction/ht_reconstruction_phase71_20261006.json').read_text())
    index = json.loads((a.source/p['data_binding']['dataset_index']).read_text())
    result = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'sealed_test_accessed': False, 'shards': [], 'arms': {}}
    for shard in index['shards']:
        path = Path(shard['path'])
        actual = digest(path)
        expected = shard['source_digest']
        assert expected, shard.keys()
        assert actual == expected
        result['shards'].append({'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size})
    for arm, job in [('aux_teacher_050','16850025'),('aux_teacher_100','16850026')]:
        run = a.source/f'artifacts/runs/ht-reconstruction-phase71-20261006/{arm}/{job}'
        native = json.loads((run/'result.json').read_text())
        src = torch.load(native['pretraining_refinement']['checkpoint'], map_location='cpu', weights_only=False)
        tracks = {}
        for track,b in native['checkpoints'].items():
            cp = torch.load(b['path'], map_location='cpu', weights_only=False)
            old,new = src['encoder_state_dict'],cp['encoder_state_dict']
            assert old.keys() == new.keys()
            changed = [k for k in old if not torch.equal(old[k],new[k])]
            if cp['step'] > 2188: assert changed
            tracks[track] = {'step': cp['step'], 'encoder_keys': len(old), 'changed_encoder_keys': changed, 'encoder_l2_change': sum((old[k].float()-new[k].float()).square().sum().item() for k in old)**0.5}
        rows = [json.loads(x) for x in (run/'training/metrics.jsonl').read_text().splitlines()]
        training = [r for r in rows if 'sampled_teacher_count' in r]
        assert len(training)==4376 and [r['step'] for r in training]==list(range(1,4377))
        keys = ['truth_target_count','representable_target_count','unrepresentable_target_count','sampled_teacher_count','sampled_predicted_count','skipped_event_level_count','fallback_teacher_count','recovery_loss_count','rollout_call_count','model_forward_count']
        sums = {k:sum(r[k] for r in training) for k in keys}
        assert sums['sampled_teacher_count']+sums['sampled_predicted_count']==280064
        from scripts.run_reconstruction_phase71 import audit_context_execution
        context_execution = audit_context_execution(run/'training/metrics.jsonl', arm)
        result['arms'][arm] = {'context_execution':context_execution, 'tracks': tracks, 'training_records':len(training), 'native_log_records':len(rows), 'context_totals':sums, 'all_finite_steps':all(r['loss_finite'] and r['model_finite'] and r['optimizer_finite'] for r in training)}
    a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
