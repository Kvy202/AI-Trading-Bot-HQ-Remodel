# Operations Runbook — AWS (paper mode)

Audience: the operator. Everything here assumes **PAPER mode**.
Live trading is out of scope until after 25 June and requires the approval flow in
SAFETY_CONTROLS.md.

---

## 1. Box facts

| Item | Value |
|---|---|
| Host | AWS EC2, Ubuntu, project at `~/bot` |
| Venv | `~/bot/.venv` (use `.venv/bin/python`, never system python) |
| Services | `bot-writer.service` (signals), `bot-executor.service` (trades), `bot-proxy.service`, grouped under `trading-bot.target` |
| Service config | `WorkingDirectory=~/bot`, `EnvironmentFile=~/bot/.env`, `Environment=PYTHONPATH=~/bot` |
| **Mode switch** | The executor's mode is set by the **systemd ExecStart CLI flag** (`--paper` / `--live`), which **overrides `.env`**. Editing `.env` alone does NOT change mode. Check with: `systemctl cat bot-executor \| grep ExecStart` — expect `--paper`. |
| Key logs | `logs/live_executor.out|.err`, `logs/live_writer.out|.err`, `logs/trades_closed_*.csv`, `logs/live_signals.csv`, heartbeats `logs/heartbeat.json` + `logs/live_writer_heartbeat.json` |
| V2 state | `logs/v2_risk_state.json`, pause file `run/V2_PAUSE`, markers `logs/deploy_markers.csv` + `logs/DEPLOY_MARKERS.txt` |

---

## 2. Standard deploy sequence (pull → test → env → restart)

```bash
ssh ubuntu@<EC2-IP>
cd ~/bot

# 0. Snapshot current state (rollback anchor)
git rev-parse --short HEAD            # note this sha
cp .env /tmp/env.backup.$(date +%Y%m%d_%H%M%S)

# 1. Stop the executor only (writer keeps producing signals; they drain on restart)
sudo systemctl stop bot-executor

# 2. Pull the branch
git fetch origin
git checkout bot-v2-architecture
git pull --ff-only origin bot-v2-architecture

# 3. Test BEFORE starting (one-time: .venv/bin/python -m pip install pytest)
.venv/bin/python -m pytest tests/ -q          # V2 suite — must be all green
.venv/bin/python tools/test_fixes_123.py      # must end: RESULT: PASS
.venv/bin/python sanity_test.py               # offline model load + forward pass

# 4. Set env — APPEND V2 lines to .env, never overwrite the file.
#    First deploy: leave all V2_* unset (or =0) → shadow mode, V1-identical.
nano .env        # see .env.v2.example for the documented flags

# 5. Record the deploy
.venv/bin/python tools/v2_deploy_marker.py --note "deploy bot-v2-architecture (flags off)"

# 6. Start + verify
sudo systemctl start bot-executor
systemctl status bot-executor --no-pager
tail -n 30 logs/live_executor.out     # expect START mode=PAPER and a v2_risk status line
```

**Writer note:** these deploys do not change the writer; do not restart `bot-writer`
unless the deploy touched writer code (this branch does not).

---

## 3. Enabling / disabling V2 risk flags

Edit `~/bot/.env` (values documented in `.env.v2.example`), then:

```bash
sudo systemctl restart bot-executor
tail -n 20 logs/live_executor.out     # expect: v2_risk: enabled time_stop_min=240 ...
```

Suggested first enablement (paper), one flag at a time:

```bash
# Day 1: time-stop only (240 min mirrors run.json MAX_HOLD_BARS=48 × 5m bars)
V2_TIME_STOP_MIN=240
# Day 2+: add daily guards
V2_MAX_SL_PER_DAY=6
V2_DAILY_LOSS_LIMIT_USDT=1.5
V2_DAILY_DD_LIMIT_USDT=2.0
```

Disable: set the flag to `0` (or comment it out) and restart. Master off-switch without
removing lines: `V2_RISK_DISABLED=1` + restart.

**Instant pause (no restart, no .env change):**
```bash
touch ~/bot/run/V2_PAUSE      # blocks NEW entries within one poll (~3 s); exits keep working
rm ~/bot/run/V2_PAUSE         # resume
```

---

## 4. Health checks

```bash
cd ~/bot
systemctl status bot-writer bot-executor --no-pager
.venv/bin/python tools/bot_health_check.py          # 8-section diagnostic
tail -n 50 logs/live_executor.out                   # TRADE/SKIP lines; v2_risk status
tail -n 20 logs/live_executor.err                   # should be quiet
cat logs/heartbeat.json                             # executor heartbeat (age < ~2 min)
cat logs/live_writer_heartbeat.json                 # writer heartbeat (age < ~60 s)
cat logs/v2_risk_state.json 2>/dev/null             # V2 counters (exists once enabled)
grep -c "reason=v2_" logs/live_executor.out         # how often V2 gates fired
grep -c "EXIT_TIME" logs/live_executor.out          # time-stop exits
```

What healthy looks like: both services `active (running)`; heartbeats fresh; `.err` quiet;
signals flowing (`idle: no new signals` is fine off-hours); v2 state day = today (UTC).

---

## 5. Daily evidence routine (order matters)

Run export **before** archive — archiving moves dated CSVs the exporter reads.

```bash
cd ~/bot
.venv/bin/python tools/v2_evidence_export.py            # all days missing from the index
.venv/bin/python tools/v2_dashboard.py                  # → reports/dashboard.html
.venv/bin/python tools/v2_log_archive.py                # DRY-RUN: review the plan
.venv/bin/python tools/v2_log_archive.py --apply        # only if the plan looked right

# Pull artifacts to the laptop for the report:
#   (run from the laptop)
scp -r ubuntu@<EC2-IP>:~/bot/reports/evidence ./reports/
scp ubuntu@<EC2-IP>:~/bot/reports/dashboard.html ./reports/
```

---

## 6. Rollback ladder (fastest first)

| Level | Action | Scope | Commands |
|---|---|---|---|
| L0 | Pause file | Blocks new entries in seconds; positions exit normally | `touch ~/bot/run/V2_PAUSE` |
| L1 | Flags off | Disables all V2 risk behavior; code stays | `sed -i 's/^V2_/#V2_/' ~/bot/.env && sudo systemctl restart bot-executor` |
| L2 | Revert the wiring commit | Removes V2 hooks from the executor; new files remain (inert) | `cd ~/bot && git revert <wiring-commit-sha> && sudo systemctl restart bot-executor` |
| L3 | Checkout last-good sha | Full code rollback to the pre-deploy snapshot from step 0 | `cd ~/bot && git checkout <last-good-sha> && sudo systemctl restart bot-executor` |

After any rollback: run section 4 health checks and add a deploy marker noting the rollback.

---

## 7. Incident playbook

| Symptom | First moves |
|---|---|
| `LOOP_ERROR` repeating in `.err` | Read the traceback. If it mentions `v2`, apply L1 (flags off) — but note every v2 call is try/except-guarded, so a v2 traceback here would itself be a bug to file. Otherwise treat as V1 incident (usually exchange/network). |
| `FATAL` / OOD alarm from writer | The OOD guard forced `allow=0` — the bot is refusing to trade on bad inputs, which is correct. Run `.venv/bin/python tools/diagnose_features.py`. Do NOT restart-loop; fix the feature/scaler mismatch first. |
| `SIDE_BIAS` / `BIAS_LOCK` lines | Working as designed: entries suspended while signals are ≥95 % one-sided. Run `tools/diagnose_bias.py`. Do not disable the guard to "make it trade". |
| Stale heartbeat (> 5 min) | `systemctl status` the service; check disk space `df -h`; check OOM `dmesg \| tail`. Restart the affected service only. |
| Stuck position (no exits firing) | Check writer is emitting that symbol (`tail logs/live_signals.csv`). Time-stop/TP/SL are signal-driven — no signals for a symbol means no exits for it (known limitation). If urgent on paper: `sudo systemctl restart bot-executor` (restart-close handles past-TP/SL positions). |
| Executor won't start after deploy | `tail -n 50 logs/live_executor.err`; check the scaler-dim assert / artifact mismatch messages; if related to the deploy, go to L2/L3. |
| V2 counters look wrong | `cat logs/v2_risk_state.json`; counters rebuild from `logs/trades_closed_$(date -u +%Y%m%d).csv` on restart — restarting the executor re-derives them from the authoritative CSV. |

---

## 8. Live/paper switch warning

Changing `.env` `LIVE_MODE`/`EXEC_PAPER` does **not** switch modes while the systemd unit
passes `--paper` or `--live` (CLI overrides env). To actually switch you must edit the
unit's `ExecStart` (`sudo systemctl edit --full bot-executor`), `sudo systemctl
daemon-reload`, and restart. **Do not do this before 25 June.** Going live additionally
requires the supervisor approval flow and the checklist in SAFETY_CONTROLS.md.

---

## 9. Experimental Mode Manager

`runtime/experiment_modes.py` centralizes read-only experimental flag bundles for
paper-safe tests. The helper prints or returns environment overrides only; it does
not edit `.env`, mutate the parent shell, or start writer/executor processes.

Supported modes:

```text
baseline
iforest_shadow
iforest_blocking
xgboost_shadow
xgboost_blocking
xgboost_shadow_outcome
survival_shadow
survival_active_placeholder
advanced_risk_shadow_placeholder
advanced_risk_active_placeholder
combined_shadow
combined_paper
```

Example inspection commands:

```bash
.venv/bin/python tools/experiment_mode.py --list
.venv/bin/python tools/experiment_mode.py --describe xgboost_shadow
.venv/bin/python tools/experiment_mode.py --print-env combined_shadow
.venv/bin/python tools/experiment_mode.py --json combined_paper
```

Do not update service runbooks to consume these modes until the Phase 8.1 wiring
step.

---

## 10. Isolation Forest blocking paper test

Use this only as a paper/test runbook. It starts `live_writer.py` only, refuses
if a repo-scoped writer/executor is already running, refuses live/mainnet
runtime settings, and does not start or change the executor.

PowerShell:

```powershell
Set-Location "C:\Project\Ai-trading-bot-Hyperliquid Remodel"
.\tools\stop_live.ps1
.\tools\run_isolation_forest_blocking_paper_test.ps1 -Minutes 30 -FreshShadowLog
```

For a longer run, use `-Minutes 60`. The script forces only this child writer
process to use:

```text
USE_ISOLATION_FOREST=true
ISOLATION_FOREST_BLOCKING=true
ISOLATION_FOREST_ARTIFACT=model_artifacts/isolation_forest.joblib
USE_XGBOOST_SIGNAL=false
USE_SURVIVAL_EXIT=false
```

Expected outputs:

- `logs/isolation_forest_shadow.csv` gains new rows.
- `reports/isolation_forest_blocking_paper_summary.json` is written.
- The text report includes `total_rows`, `would_block_count`,
  `actually_blocked_count`, `block_rate`, and `top_reasons`.

---

## 11. Controlled Experiment Matrix

Use the matrix runner for repeatable paper-only experimental tests with
consistent duration, fresh logs, and standardized reports. It requires either
one `-Mode` or `-All`; it does not run every mode by default.

```powershell
.\tools\run_experiment_matrix.ps1 -Mode combined_shadow -Minutes 30 -FreshLogs
.\tools\run_experiment_matrix.ps1 -All -Minutes 30 -FreshLogs
.\tools\run_experiment_matrix.ps1 -Mode xgboost_shadow_outcome -Minutes 5 -DryRun
```

Each run writes a `reports/matrix_index_<timestamp>.json` plus per-mode
`matrix_<mode>_<timestamp>_*` reports. `-DryRun` prints the commands and report
paths without starting writer or executor processes.

---

## 12. Final Experiment Comparison

After the matrix paper tests are complete, generate the read-only final
comparison report from the saved `reports/matrix_*` JSON files:

```powershell
python tools\final_experiment_comparison.py --reports-dir reports
python tools\final_experiment_comparison.py --reports-dir reports --json --json-out reports\final_experiment_comparison.json
```

The report is for comparison and signoff only. It does not enable any live,
testnet, or real-order behavior.

---

## 13. Phase 16: Calibration Recommendation Report

Use this read-only report after the final experiment comparison to plan the
next paper-only calibration pass:

```powershell
python tools\calibration_recommendation_report.py --reports-dir reports --logs-dir logs
python tools\calibration_recommendation_report.py --reports-dir reports --logs-dir logs --json --json-out reports\calibration_recommendation_report.json
```

The report recommends calibration steps only. It must not be used to enable
live trading, testnet real orders, or `PLACE_REAL_ORDERS`.

---

## Phase 17: Offline Calibration Sweep

Use this read-only tool to aggregate completed matrix runs, measure cross-run
stability, and simulate offline calibration thresholds from current row-level
shadow logs:

```powershell
python tools\offline_calibration_sweep.py --reports-dir reports --logs-dir logs
python tools\offline_calibration_sweep.py --reports-dir reports --logs-dir logs --json --json-out reports\offline_calibration_sweep.json
```

The generated `reports/offline_calibration_sweep.json` is local analysis output
and must remain untracked and uncommitted. Do not commit logs, archives,
generated reports, or model artifacts.

This phase is read-only. It must not:

- Change trading logic, `live_writer.py`, `live_executor.py`, or `features.py`.
- Change `FEATURE_COLS` or model/scaler feature columns.
- Retrain or overwrite model or scaler artifacts.
- Modify `.env`.
- Enable blocking, active exits, Advanced Risk active mode, or XGBoost blocking.
- Enable testnet real orders, mainnet, or `PLACE_REAL_ORDERS`.

---

## Phase 18: Offline Calibration Proposals

Use this read-only tool after Phase 17 to turn the accumulated evidence into
explicit calibration specifications, review gates, and the next paper-only
experiment plan:

```powershell
python tools\offline_calibration_proposals.py --reports-dir reports --logs-dir logs
python tools\offline_calibration_proposals.py --reports-dir reports --logs-dir logs --json --json-out reports\offline_calibration_proposals.json
```

Phase 17 JSON is the preferred input. If it is absent or malformed, the tool
reconstructs the required evidence from current matrix reports and current
row-level shadow logs. It does not aggregate archived CSV files without a
safely verified run identity.

The generated `reports/offline_calibration_proposals.json` is local analysis
output and must remain ignored, untracked, and uncommitted. Do not commit logs,
archives, generated reports, or model artifacts.

This phase is read-only. It does not retrain, overwrite artifacts, change
trading behavior, or enable any active, blocking, live, or real-order mode.
Specifically:

- Do not change `live_writer.py`, `live_executor.py`, or `features.py`.
- Do not change `FEATURE_COLS` or model/scaler feature columns.
- Do not retrain in this phase or overwrite model/scaler artifacts.
- Do not modify `.env`.
- Do not enable Isolation Forest blocking or XGBoost blocking.
- Do not enable Survival active mode or Advanced Risk active mode.
- Do not enable mainnet, testnet real orders, or `PLACE_REAL_ORDERS`.

---

## Phase 18.1: Stale Signal Replay Protection

The paper executor maintains a monotonic signal high-water mark for each
symbol. At startup it initializes those marks from the newest visible rows in
`logs/live_signals.csv`; existing history is drained but is never replayed as
new trading input. A later row is eligible only when its timestamp is strictly
newer than that symbol's high-water mark.

The matrix runner records `run_started_utc` for each mode and checks current
`trades_paper_*.csv` entry rows before accepting reports. A `BUY` or
`SELL_SHORT` timestamp earlier than the mode start, less the configured clock
tolerance, fails the mode with the note
`stale_signal_replay_or_prestart_entry_detected`. The matrix index records
`stale_entry_guard_checked`, `stale_entry_count`,
`stale_entry_signal_ids`, and `evidence_valid`; contaminated evidence always
has `evidence_valid=false`.

Contaminated paper reports and logs are retained unchanged for audit. The
`-FreshLogs` option archives the matrix trade and shadow logs; it does not mean
that `logs/live_signals.csv` is deleted or reset.

This protection is paper-only. It does not enable active or blocking modules,
testnet or mainnet orders, real-order mode, or `PLACE_REAL_ORDERS`; those modes
remain disabled.

---

## Phase 19 — Evidence Manifest and Contaminated-Run Exclusion

Build the authoritative, paper-only matrix evidence registry before reviewing
Phase 17 or Phase 18 results:

```powershell
python tools\evidence_manifest.py
python tools\evidence_manifest.py --reports-dir reports --overrides research\evidence_overrides.json --json-out reports\evidence_manifest.json
```

The registry groups the unified report, shadow summary, and XGBoost audit under
one canonical `mode:matrix_timestamp` identity. It retains every historical
report and log for audit; it never deletes, rewrites, or moves historical
evidence. `reports/evidence_manifest.json` is generated analysis and remains
ignored and untracked. `research/evidence_overrides.json` is the tracked,
reviewed source registry.

Pre-Phase-18.1 strategy evidence fails closed because those indexes do not
contain the `evidence_valid` and stale-entry guard metadata needed to prove an
independent experiment window. A report's existence, a zero process exit, or
the absence of an obvious stale row is not proof that the run is valid.

Each identity receives exactly one classification:

- `valid_strategy_evidence`: clean completed evidence with the mode-specific
  realized outcome needed for strategy aggregation.
- `valid_safety_only`: a clean, deliberately short safety validation; strategy
  outcomes are not required.
- `incomplete_no_outcomes`: a safe completed strategy-evidence window with no
  usable closed or matched outcomes.
- `contaminated_stale_signal`: confirmed stale or pre-start signal replay.
- `network_interrupted`: a connectivity interruption confirmed by reviewed
  override; connectivity is never inferred from exit status alone.
- `invalid_matrix_failure`: a failed matrix process or missing/malformed
  required generated evidence.
- `unverified_legacy`: legacy or report-only evidence without reliable
  verification or explicit reviewed approval.

`include_in_strategy_aggregate` and `include_in_safety_summary` are separate.
Only `valid_strategy_evidence` enters PnL, win rate, matched-outcome separation,
and promotion evidence. Safety-only and incomplete runs may enter safety
summaries but never strategy calculations. Contaminated, interrupted, failed,
and unverified runs enter neither. XGBoost shadow decision-row volume is not
closed-trade evidence: `xgboost_shadow_outcome` requires uniquely matched closed
trades.

Manual classifications require a non-empty reason, `reviewed: true`, a valid
classification, and a canonical identity in the tracked override registry.
Overrides take precedence over automatic classification. Do not add commands,
environment values, runtime settings, thresholds, risk settings, fees,
slippage, or position-sizing changes to the registry.

Phase 17 rebuilds the current manifest directly and records its deterministic
SHA-256 evidence digest plus every excluded identity and reason. Phase 18 only
reuses a Phase 17 JSON report when its required sections, manifest schema, and
digest match current evidence; otherwise it marks that report stale and
reconstructs Phase 17 using the current manifest.

Phase 19 changes no trading behavior, model/scaler artifact, feature columns,
runtime configuration, fee/slippage assumption, threshold, risk rule, or
position sizing. It does not activate blocking modules, start writers or
executors, run counterfactual replay, or enable testnet/mainnet real orders.

---

## Phase 20 — Deterministic Counterfactual Replay

Phase 20 reconstructs completed Phase 19 experiment windows offline and
replays the paper executor against the recorded signal ticks. It uses each
signal row's recorded price; candle high/low data and future rows are never
used for a current decision. The tool does not initialize an exchange adapter,
load a trading model, start a writer or executor, or place paper, testnet, or
mainnet orders.

Inventory the Phase 19 strategy identities without executing trade lifecycles:

```powershell
python tools\counterfactual_replay.py --all-strategy-runs --inventory-only --json-out reports\counterfactual_replay_inventory.json
```

Run one eligible identity or every strategy identity:

```powershell
python tools\counterfactual_replay.py --identity xgboost_shadow_outcome:20260803161821 --json-out reports\counterfactual_replay_20260803161821.json
python tools\counterfactual_replay.py --all-strategy-runs --json-out reports\counterfactual_replay.json
```

The replay contract captures only effective, non-secret executor settings. Its
priority is `config/run.json`, then `.env` loaded in memory, then the explicit
matrix-mode overrides. Credentials, wallet data, environment dumps, local
paths, and executable commands are not contract fields. A historical `.env`
is never inferred from the current `.env`: historical runs remain
`contract_missing` unless an exact matrix snapshot exists or a complete,
independently verified contract is added to
`research/replay_contract_overrides.json` with `reviewed: true`.

Future accepted matrix modes capture the contract before processes start and
copy only their in-window evidence into
`reports/replay_bundles/<mode>_<timestamp>/` after processes stop. Source rows
remain untouched. Archive directory names are inventory hints only, not run
identities; historical resolution uses the Phase 19 window, exact `signal_id`,
symbol, timestamp, reported counts, and canonical row digests. XGBoost joins
are by `signal_id` only. Missing IDs are recorded and excluded, timestamp
fallback is prohibited, and conflicting decisions fail closed.

Baseline replay must match recorded paper entries, closes, and final open
state before counterfactual output can become evidence-grade. Open positions
at the end of the window are censored and receive no realized PnL; they are
never force-closed. Zero closes are not testable, and one exact closed-trade
match is only a mechanical check. Promotion-grade parity requires at least ten
exact matched closes plus complete source coverage and a parity-grade contract.

The portfolio output includes the shadow baseline, XGBoost-confirm-only, and
research-only XGBoost-reject-only policies. Independent confirmed/rejected
decision cohorts overlap by design and are explicitly non-additive; their
outcomes must not be summed into a portfolio PnL claim. A positive cohort
difference is not a significance claim.

Generated contracts, bundles, inventories, and counterfactual reports remain
ignored analysis artifacts. Counterfactual results do not enter the Phase 17
or Phase 18 promotion gates. Isolation Forest blocking, XGBoost blocking,
Survival active exits, Advanced Risk active actions, position restoration,
and live or real-order modes remain disabled. Missing configuration, failed
coverage, unresolved adaptive/bias state, unsupported active behavior, or
failed parity leaves the result exploratory or excluded.

---

## Phase 21 — Ensemble Model Health Audit

Phase 21 is a deterministic, CPU-only diagnostic of the deployed deep-learning
ensemble. It reads model, scaler, metadata, and recorded probability artifacts;
it does not fetch candles or exchange data, initialize an exchange adapter,
start a writer or executor, or place paper, testnet, or real orders. It never
changes model weights, scalers, `FEATURE_COLS`, thresholds, ensemble weights, or
trading configuration.

Capture a paper-safe serving snapshot for a future matrix run with the exact
forced environment used by that run:

```powershell
python tools\model_serving_snapshot.py --identity xgboost_shadow_outcome:20260804160153 --mode xgboost_shadow_outcome --forced-env-json <temporary-safe-mode-json> --json-out reports\model_serving_snapshot.json
```

Run the historical audit and deterministic artifact probes offline:

```powershell
python tools\model_health_audit.py --identity xgboost_shadow_outcome:20260804160153 --json-out reports\model_health_audit_20260804160153.json
python tools\model_health_probe.py --json-out reports\model_health_probe.json
```

The serving snapshot contains only allowlisted, non-secret settings, normalized
artifact filenames, SHA-256 digests, metadata, scaler health, and read-only CPU
load status. It is accepted only with `paper_mode=true` and
`place_real_orders=false`. A current local environment is never promoted to an
exact historical snapshot. Missing historical snapshots remain explicitly
unverified.

For future matrix runs, the runner captures this snapshot before starting the
selected paper mode and includes it in the completed replay bundle. The bundle
also filters `live_models_by_symbol.csv` and `live_meta_log.csv` strictly to the
run window. Exact duplicate rows may collapse; different probabilities for the
same timestamp and symbol fail reproducibility. Existing completed Phase 20
bundles are not rewritten.

The audit compares training metadata with the effective serving timeframe,
sequence length, feature count, symbol-id setting, symbol universe, and paired
scaler width. A `1m` serving value against `5m` training metadata is a critical
contract mismatch even when dimensions happen to match. Artifact-load success
alone is not a health verdict.

Historical probability analysis is exact CSV parsing, not OCR or approximate
log interpretation. TCN flatness uses the deployed `0.002` standard-deviation
threshold. Thirty rows can support a warning but not a final flat-output
decision; the default decision gate is 100 rows. One-sided but varying output
is a warning, not proof of failure.

Offline probes are created in standardized scaler space, inverse-transformed to
raw feature space, and passed through the paired scaler/model twice on CPU.
The TCN architecture diagnostic reuses the trained weights without saving or
mutating them and compares the deployed symmetric-padding final endpoint with
right-cropped and causal-left-padding endpoints. Any padding diagnosis is an
advisory hypothesis only; Phase 21 does not patch `dl_models.py` or change the
production forward path.

Ensemble variants (`current_config`, equal/AUC weighted, LSTM+Transformer-only,
no-TCN, and TCN-only) reproduce positive-weight voting, `DL_MIN_AGREE`, neutral
suppression, centered probability, and the configured threshold. LSTM+TX and
no-TCN are shadow-configuration candidates only. Phase 21 reports no strategy
PnL and does not write a candidate into `.env` or `config/run.json`.

Generated snapshots and health reports remain ignored and untracked. Health
diagnostics do not enter Phase 17 or Phase 18 promotion evidence. Active or
blocking experimental modes, live mode, and real-order mode remain disabled.

# Phase 22 training-serving alignment shadow validation

Phase 22 is a research-only, non-trading workflow. It validates the deployed
artifacts at their 5-minute/64-row training contract without changing `.env`,
`config/run.json`, weights, calibration, models, or scalers. Historical evidence
establishes model-output health; the short live shadow establishes only current
completed-bar integration. Neither is profitability evidence or Phase 17/18
promotion evidence.

Capture at least 120 completed historical endpoints per serving symbol, then
evaluate the immutable bundle without another market-data request:

```powershell
python tools/model_alignment_shadow.py capture-history `
  --timeframe 5m `
  --symbols BTCUSDT,ETHUSDT `
  --unique-bars 120 `
  --lookback-bars 800 `
  --bundle-out reports/model_alignment_bundles/history_5m_latest

python tools/model_alignment_shadow.py evaluate `
  --bundle reports/model_alignment_bundles/history_5m_latest `
  --json-out reports/model_alignment_evaluation_5m.json

python tools/model_alignment_report.py `
  --historical-bundle reports/model_alignment_bundles/history_5m_latest `
  --historical-evaluation reports/model_alignment_evaluation_5m.json `
  --phase21-report reports/model_health_audit_20260804160153.json `
  --json-out reports/model_alignment_report.json
```

After code review, the writer-free live integration smoke can observe three new
completed bars per symbol. It never launches an execution process or writes
signal/trade rows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\run_model_alignment_shadow.ps1 `
  -UniqueBars 3 `
  -Symbols "BTCUSDT,ETHUSDT" `
  -FreshLogs
```

Each normal invocation creates a new campaign directory. `-CampaignDir` resumes
that exact campaign only when its symbols, timeframe, snapshot, artifact
digests, calibration, and requested bar count still match. `-FreshLogs` refuses
to overwrite an existing campaign directory. Repeated polls of one completed
bar do not increment counters, while a changed window digest for the same source
bar is a critical failure.

Before running the live smoke, inspect compatibility without changing packages:

```powershell
python tools/model_alignment_report.py --compatibility-only
```

Use `-DryRun` during implementation verification. A standard experiment matrix
now refuses a training-serving contract failure before starting its runtime
processes; it does not silently change a normal matrix run from 1 minute to 5
minutes. A scikit-learn artifact/runtime mismatch remains an explicit
reproducibility warning and must be remediated manually—no Phase 22 tool changes
installed dependencies.

## Phase 23: runtime reproducibility and retraining triage

Phase 23 is an offline, read-only diagnostic. It uses the completed Phase 22
bundle and never fetches replacement market data, changes incumbent artifacts,
or authorizes model activation. The dedicated environment contains only the
scaler transformation dependencies and is separate from `.venv`.

Inventory and safe runbook preview:

```powershell
python tools/model_runtime_repro.py --inventory-only
powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\run_model_runtime_repro.ps1" -DryRun
```

Create or validate the dedicated environment, then compare:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\run_model_runtime_repro.ps1" -Bootstrap -Compare -Bundle ".\reports\model_alignment_bundles\history_5m_final"
```

The environment path is fixed at `.venv-repro-sklearn180`. An existing
environment is reused only when its requirements digest, Python major/minor,
and exact package versions match. A mismatch is a hard failure; remove or
archive that dedicated environment manually before deliberately rebuilding it.

After a successful comparison, generate failure, lineage, and retraining
triage reports with the Phase 23 CLIs. `phase24_allowed` means only that
versioned candidate training work may begin; it never means live use or
promotion is approved.

## Phase 23.1: runtime stack isolation and canonical training pin

Phase 23.1 changes one scaler-only numerical dependency at a time below
`.venv-runtime-isolation`. These environments contain NumPy, SciPy, joblib,
scikit-learn, and threadpoolctl only. They never replace or install into the
project `.venv`. The immutable Phase 22 bundle is the only comparison input;
model inference for every scaled array uses the unchanged main CPU PyTorch
runtime.

Preview every exact version and output path without creating an environment or
report:

```powershell
python tools/runtime_stack_isolation.py --inventory-only
powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\run_runtime_stack_isolation.ps1" -DryRun
```

After unit-test and code review, create the primary isolated stacks and compare
them. If no single-package stack explains the full delta, the same invocation
creates only the required interaction stacks and repeats the comparison:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File ".\tools\run_runtime_stack_isolation.ps1" `
  -Bootstrap -Compare `
  -Bundle ".\reports\model_alignment_bundles\history_5m_final"

python tools/runtime_stack_attribution.py `
  --isolation-report reports/runtime_stack_isolation.json `
  --json-out reports/runtime_stack_attribution.json

python tools/runtime_stack_decision.py `
  --isolation-report reports/runtime_stack_isolation.json `
  --attribution-report reports/runtime_stack_attribution.json `
  --json-out reports/runtime_stack_decision.json
```

Canonical selection pins only the scaler-side numerical packages in
`requirements/model_numeric_canonical.txt`. It permits future Phase 24
candidate work only inside the selected dedicated environment. The current
serving runtime remains noncanonical and migration-blocked, and live activation
and promotion remain disallowed. A numerical delta remains material under the
strict Phase 23 tolerances even when every direction, exclusion, allow,
agreement, and ensemble decision is unchanged.
# Phase 24 — versioned candidate retraining and sealed health confirmation

Phase 24 is classification-health research only. It cannot place any kind of
order, activate a candidate, change calibration, or overwrite a serving model
or scaler. A passing confirmation gate means only that a candidate is eligible
for a later, explicitly reviewed shadow comparison.

Interpreter separation is mandatory:

- Project/main Python performs public, completed-bar capture and feature/label
  construction.
- `.venv-model-training/canonical/Scripts/python.exe` performs scaler fitting,
  candidate training, internal testing, comparisons, and health gates from
  frozen NPZ/JSON evidence. It has no market-data capture path and contains no
  exchange packages.

Start with the non-mutating checks:

```powershell
python tools/model_training_environment.py --inventory-only
powershell -NoProfile -ExecutionPolicy Bypass `
  -File ".\tools\run_model_candidate_training.ps1" `
  -Model lstm `
  -DryRun
```

Do not bootstrap, capture, or train during an implementation-verification run.
After code review, the explicit operational order is: bootstrap; capture the
pre-Phase-22 training history; build and verify the frozen dataset/scaler; then
for LSTM, TCN, and Transformer in that order train all three seeds, select on
validation only, open internal test once, and run the legacy repair gate. Only
after all three selected candidate IDs and digests are frozen may the sealed
confirmation set be captured and evaluated once per frozen candidate. Finish
by generating the review-only registry proposal. Never copy candidate files to
the serving names as part of this workflow.
