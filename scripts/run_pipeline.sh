#!/bin/bash
# Master orchestrator for mis P-experimental-completion.
# Runs Phase 1 (W2, W5, W3-attempt) → Phase 2 BC retries → flags Phase 3 for Claude.
# Updates reports/pipeline_state.json at each transition.
# Logs everything to reports/pipeline_orchestrator.log.
#
# Idempotent: re-launching skips completed tasks (checks output files).
# Resumable: state file tracks last-completed task.
#
# Designed for unattended multi-day operation on M4. Per-task timeouts prevent
# runaway processes (per the v1 BC retrain disaster).
set -u
cd /Users/huanbui/Desktop/adaptive-physics-attack-ad
PYTHON=.venv/bin/python
STATE=reports/pipeline_state.json
LOG=reports/pipeline_orchestrator.log

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $LOG
}

update_state() {
  # update_state <phase_key> <task_key> <field> <value>
  $PYTHON -c "
import json
d = json.load(open('$STATE'))
d['last_updated'] = '$(date -u +%Y-%m-%dT%H:%M:%S)'
d['phases']['$1']['tasks']['$2']['$3'] = '$4'
d['current_phase'] = '$1'
d['current_task'] = '$2'
json.dump(d, open('$STATE','w'), indent=2)
"
}

update_phase_state() {
  # update_phase_state <phase_key> <field> <value>
  $PYTHON -c "
import json
d = json.load(open('$STATE'))
d['last_updated'] = '$(date -u +%Y-%m-%dT%H:%M:%S)'
d['phases']['$1']['$2'] = '$3'
d['current_phase'] = '$1'
json.dump(d, open('$STATE','w'), indent=2)
"
}

escalate() {
  # escalate <severity> <message>
  $PYTHON -c "
import json
d = json.load(open('$STATE'))
d['escalations'].append({'time': '$(date -u +%Y-%m-%dT%H:%M:%S)', 'severity': '$1', 'message': '$2'})
json.dump(d, open('$STATE','w'), indent=2)
"
  log "[ESCALATE:$1] $2"
}

# ----------------------------------------------------------------------------
# PHASE 1 Task 1.1: W2 ε sweep (may already be running)
# ----------------------------------------------------------------------------
log "Pipeline start"

if [ ! -f reports/phase1_w2_eps_sweep/wadi_seed3_eps0.50.json ]; then
  log "Phase 1.1 W2 ε sweep — launching"
  bash /tmp/phase1_w2_eps_sweep.sh 2>&1 | tee -a reports/phase1_w2_eps_sweep.log
  if [ -f reports/phase1_w2_eps_sweep/wadi_seed3_eps0.50.json ]; then
    update_state phase_1 1.1_w2_eps_sweep status completed
    log "Phase 1.1 W2 completed"
  else
    escalate ERROR "Phase 1.1 W2 incomplete; expected reports/phase1_w2_eps_sweep/wadi_seed3_eps0.50.json"
    exit 1
  fi
else
  log "Phase 1.1 W2 already complete (output detected)"
  update_state phase_1 1.1_w2_eps_sweep status completed
fi

# ----------------------------------------------------------------------------
# PHASE 1 Task 1.2: W5 FGSM
# ----------------------------------------------------------------------------
log "Phase 1.2 W5 FGSM — launching"
update_state phase_1 1.2_w5_fgsm status running
if bash /tmp/phase1_w5_fgsm_sweep.sh 2>&1 | tee -a reports/phase1_w5_fgsm_sweep.log; then
  update_state phase_1 1.2_w5_fgsm status completed
  log "Phase 1.2 W5 completed"
else
  escalate WARNING "Phase 1.2 W5 had failures; check reports/phase1_w5_fgsm_sweep.log"
  update_state phase_1 1.2_w5_fgsm status completed_with_warnings
fi

# ----------------------------------------------------------------------------
# PHASE 1 Task 1.3: W3 TopoGDN-SWaT defended-PPO
# ----------------------------------------------------------------------------
log "Phase 1.3 W3 TopoGDN-SWaT — checking prerequisites"
if [ ! -f scripts/ppo_train_topogdn.py ]; then
  escalate INFO "Phase 1.3 prerequisite missing: scripts/ppo_train_topogdn.py — flagging for Claude Code interactive session"
  update_state phase_1 1.3_w3_topogdn_swat status needs_claude_code
  log "Phase 1.3 skipped pending Claude Code code-writing session; proceeding to Phase 2"
else
  log "Phase 1.3 W3 — attempting (up to 2 retries per spec)"
  RETRY=0
  while [ $RETRY -lt 2 ]; do
    log "Phase 1.3 attempt $((RETRY+1))/2"
    if timeout 14400 $PYTHON scripts/ppo_train_topogdn.py --seed 0 --dataset swat 2>&1 | tee -a reports/phase1_w3_topogdn_swat.log; then
      update_state phase_1 1.3_w3_topogdn_swat status completed
      log "Phase 1.3 W3 completed on attempt $((RETRY+1))"
      break
    else
      RETRY=$((RETRY+1))
      log "Phase 1.3 W3 attempt $RETRY failed (likely PH C++ crash)"
    fi
  done
  if [ $RETRY -ge 2 ]; then
    escalate INFO "Phase 1.3 W3 failed after 2 retries (PH C++ extension); proceeding to Phase 2 per spec"
    update_state phase_1 1.3_w3_topogdn_swat status failed_ph_crash
  fi
fi

# ----------------------------------------------------------------------------
# PHASE 1 BACKBRIEF FLAG
# ----------------------------------------------------------------------------
update_phase_state phase_1 status awaiting_backbrief
log "Phase 1 compute complete; backbrief data ready in reports/phase1_w2_eps_sweep/ + reports/phase1_w5_fgsm/"

# ----------------------------------------------------------------------------
# PHASE 2: Q6 non-lean BC retry
# ----------------------------------------------------------------------------
log "Phase 2 launching — non-lean BC retry on 8 failed seeds"
update_state phase_2 2.1_non_lean_bc_retry status running
if bash /tmp/phase2_bc_nonlean_retry.sh 2>&1 | tee -a reports/phase2_bc_nonlean_retry.log; then
  update_state phase_2 2.1_non_lean_bc_retry status completed
  log "Phase 2 BC retry completed"
else
  escalate WARNING "Phase 2 BC retry had failures; check log"
  update_state phase_2 2.1_non_lean_bc_retry status completed_with_warnings
fi

# Aggregate pass rate for escalation decision
PASS_COUNT=$($PYTHON -c "
import json, glob
pass_count = 0
for fn in glob.glob('reports/phase2_bc_nonlean/bc_pretrain_*_report.json'):
    try:
        d = json.load(open(fn))
        # The history field has per-epoch val_cat_acc; max over history is best-val
        if d.get('history') and 'val_cat_acc' in d['history']:
            if max(d['history']['val_cat_acc']) >= 0.95:
                pass_count += 1
    except: pass
print(pass_count)
")
log "Phase 2 result: $PASS_COUNT / 8 seeds passed at non-lean config"
if [ "$PASS_COUNT" -ge 7 ]; then
  escalate INFO "Phase 2: $PASS_COUNT/8 passed — POPULATE FULL Q6 COLUMN"
else
  escalate INFO "Phase 2: $PASS_COUNT/8 passed (below 7/8 threshold) — Q6 unmeasurable across seeds; document in Appendix C"
fi
update_phase_state phase_2 status awaiting_backbrief

# ----------------------------------------------------------------------------
# PHASE 3 BLOCK: needs Claude Code interactive session
# ----------------------------------------------------------------------------
escalate CRITICAL "Phase 3 (PH C++ extension fix + multi-seed TopoGDN) requires Claude Code interactive session. Pipeline pausing here."
update_phase_state phase_3 status needs_claude_code
log "Pipeline pausing at Phase 3 boundary"
log "Phase 4 (PH-regularization defense) also blocked on Phase 3"
log "Total compute time elapsed: see timestamps"
log "Pipeline orchestrator exit OK"
exit 0
