#!/usr/bin/env python3
"""Authenticate completed weights without relabelling failed cohort checks."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.run_reconstruction_phase35 import finite_checkpoint,sha256,atomic_json,replay_slot_audit,first_twenty_gate
from scripts.run_reconstruction_phase57_full_decay import verify_evaluation_contract
from scripts.run_reconstruction_phase57 import CHECKPOINT_TRACKS
from hypertagging.evaluation.checkpoint_pair import validate_checkpoint_pair
ARMS={'pretraining_balance_control':'16647930','lower_late_pid_pretraining':'16647931'}
STATUS='TRAINED_COHORT_DEVIATION_DIAGNOSTIC_ONLY'


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
 source=args.source_root.resolve(strict=True);out=args.output_root.resolve();observed=[]
 for arm,job in ARMS.items():
  run=source/f'artifacts/runs/ht-reconstruction-phase57-20260922/{arm}/{job}'
  original=source/f'artifacts/slurm/reconstruction-phase57/jobs/{job}/attempt-00/receipt.json';receipt=json.loads(original.read_text());assert receipt['status']=='failed' and receipt['terminal_stage']=='trainer' and receipt['artifacts']['training_result'] is None and receipt['artifacts']['full_decay_gates'] is None
  contract,runtime=verify_evaluation_contract(run/'provenance/submitted-contract.json',source)
  cohort=json.loads(Path(runtime['cohort']).read_text());ref=json.loads((run/'pretraining/refinement-result.json').read_text());assert ref['status']=='COMPLETED' and ref['step']==2188 and sha256(Path(ref['checkpoint']))==ref['checkpoint_sha256']
  checkpoints={k:finite_checkpoint(run/'training'/n) for k,n in CHECKPOINT_TRACKS.items()};final=torch.load(run/'training/checkpoint.pt',map_location='cpu',weights_only=False)
  assert final['step']==4376 and all(v['all_checkpoint_tensors_finite'] for v in checkpoints.values())
  selection=final['validation_selection'];actual=selection['event_uids'];strict=cohort['event_uids'];expected=cohort['checkpoint_selection_event_uids'];observed.append(actual)
  assert len(actual)==len(set(actual))==1000 and selection['rollout_event_uids']==actual and len(set(actual)&set(expected))==734 and set(strict)<=set(actual)
  lineage={}
  for key,v in checkpoints.items():
   d=torch.load(v['path'],map_location='cpu',weights_only=False);assert d['validation_selection']['event_uids']==actual
   report=validate_checkpoint_pair(ref['checkpoint'],v['path'],require_exact_frozen_encoder=v['step']<=2188);assert report.compatible and report.configured_pretraining_path_match;lineage[key]=report.as_dict()
  recorded=final['data_order_contract']['balanced_level_replay'];preregistered=contract['config']['balanced_level_replay_contract'];assert recorded=={k:v for k,v in preregistered.items() if k!='planned_schedule'}
  replay=replay_slot_audit(run/'training/metrics.jsonl',config=contract['config'],trainer_contract={**recorded,'planned_schedule':preregistered['planned_schedule']})
  result={'status':STATUS,'scientific_classification':'ALL_STRICT_EVENTS_USED_FOR_CHECKPOINT_SELECTION','independent_validation':False,'selection_expected_overlap':734,'strict_selection_overlap':100,'actual_selection_event_uids':actual,'original_receipt_sha256':sha256(original),'original_receipt_status':'failed','contract_sha256':contract['contract_sha256'],'optimizer_steps':4376,'elapsed_seconds':None,'pretraining_refinement':ref,'checkpoints':checkpoints,'balanced_level_replay':replay,'first_20_optimizer_steps':first_twenty_gate(run/'training/metrics.jsonl'),'checkpoint_lineage':lineage,'sealed_test_accessed':False,'promotion_authorized':False}
  path=out/arm/'diagnostic-training-evidence.json'
  if path.exists():raise FileExistsError(path)
  atomic_json(path,result)
 assert observed[0]==observed[1]
 print('PASS: complete weights authenticated; all strict events selection-contaminated, diagnostic only')

if __name__=='__main__':main()
