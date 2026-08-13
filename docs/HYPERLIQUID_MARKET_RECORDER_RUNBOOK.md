# Passive Hyperliquid market-recorder operations

This service collects only unauthenticated Hyperliquid mainnet public market
data. It is independent of every trading, writer, executor, serving, model, and
research service. Do not add an `EnvironmentFile`, wallet address, API key, or
private key to this unit.

The examples assume the repository is `/home/ubuntu/bot`, its existing virtual
environment is `/home/ubuntu/bot/.venv`, and the service runs as `ubuntu`.
Replace that path if the installation differs.

## Install the systemd unit

These commands install the checked-in unit template. They do not start any
trading service.

```bash
cd /home/ubuntu/bot
test -x .venv/bin/python
.venv/bin/python -c 'import aiohttp; print(aiohttp.__version__)'
.venv/bin/python -m pytest \
  tests/test_hyperliquid_market_recorder.py \
  tests/test_hyperliquid_market_aggregate.py \
  tests/test_hyperliquid_market_recorder_status.py -q

sed 's|__INSTALL_DIR__|/home/ubuntu/bot|g' \
  deploy/aws/hl-market-recorder.service \
  | sudo tee /etc/systemd/system/hl-market-recorder.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now hl-market-recorder.service
```

The exact continuous recorder command installed in `ExecStart` is:

```bash
/home/ubuntu/bot/.venv/bin/python -u \
  tools/hyperliquid_market_recorder.py \
  --output-root data/hyperliquid_market_capture \
  --heartbeat-seconds 5 \
  --max-backoff-seconds 30 \
  --log-level INFO
```

There is intentionally no duration argument. Systemd restarts unexpected exits,
but does not bypass the recorder's one-instance lock. Exit status 2 is treated
as a configuration/ownership failure instead of a restart loop.

## Service operations

```bash
sudo systemctl status hl-market-recorder.service --no-pager
sudo journalctl -u hl-market-recorder.service -n 100 --no-pager
sudo journalctl -u hl-market-recorder.service -f

# Clean stop: SIGINT lets the recorder flush and release its owned lock.
sudo systemctl stop hl-market-recorder.service

# Start or restart only this passive recorder.
sudo systemctl start hl-market-recorder.service
sudo systemctl restart hl-market-recorder.service
```

Never launch the recorder manually while the service is active. The lock will
fail closed, but an active systemd unit remains the sole operational owner.

## Read-only status

Run status after startup and during monitoring:

```bash
cd /home/ubuntu/bot
.venv/bin/python tools/hyperliquid_market_recorder_status.py \
  --capture-root data/hyperliquid_market_capture
echo "exit=$?"  # 0 HEALTHY, 1 WARNING, 2 FAILED
```

Machine-readable form:

```bash
.venv/bin/python tools/hyperliquid_market_recorder_status.py \
  --capture-root data/hyperliquid_market_capture \
  --json
```

If a known integrity-error count has been investigated and explicitly accepted,
pass that exact ceiling. Any later increase fails health:

```bash
.venv/bin/python tools/hyperliquid_market_recorder_status.py \
  --capture-root data/hyperliquid_market_capture \
  --allowed-integrity-errors 3
```

Do not increase the allowance merely to clear an alarm. Audit and retain the
diagnostic evidence first.

### Health policy

The command uses UTC ages and returns the most severe result. A value at or
below its warning threshold is healthy.

| Signal | Warning after | Failed after |
|---|---:|---:|
| heartbeat file modification | 15 seconds | 30 seconds |
| last received WebSocket message | 30 seconds | 90 seconds |
| trades, per symbol | 15 minutes | 60 minutes |
| BBO, per symbol | 2 minutes | 10 minutes |
| sampled/received L2 book, per symbol | 30 seconds | 2 minutes |
| active asset context, per symbol | 30 seconds | 2 minutes |
| native 5-minute candle update, per symbol | 10 minutes | 30 minutes |

Trades and candles deliberately have wider thresholds because their public
updates can be sparse in quiet markets. A missing timestamp is a warning during
the stream's startup grace and a failure once its failure interval passes.

The following fail immediately: a missing/incompatible manifest, missing raw
root or required stream/symbol directory, malformed heartbeat counters, a
missing non-empty partition after startup grace, a stale/invalid/absent local
recorder lock, a timestamp materially in the future, or integrity errors above
the explicitly accepted baseline. A lock owned by another host is a warning
because a read-only local process check cannot establish that remote PID's
liveness.

## Read-only offline continuity audit

The default audits completed UTC-hour partitions. If the recorder is stopped,
the final closed partial hour is also safe to scan.

```bash
cd /home/ubuntu/bot
.venv/bin/python tools/hyperliquid_market_recorder_status.py \
  --capture-root data/hyperliquid_market_capture \
  --audit
echo "exit=$?"  # 0 PASSED, 1 WARNING, 2 FAILED
```

Machine-readable audit:

```bash
.venv/bin/python tools/hyperliquid_market_recorder_status.py \
  --capture-root data/hyperliquid_market_capture \
  --audit \
  --json > /tmp/hyperliquid-market-audit.json
```

For incident diagnosis only, an operator may include the actively written
current UTC hour. Reads can observe different endpoints while the file grows,
so this mode is not the authoritative completed-partition audit:

```bash
.venv/bin/python tools/hyperliquid_market_recorder_status.py \
  --capture-root data/hyperliquid_market_capture \
  --audit \
  --include-current-hour
```

The command never repairs, deletes, truncates, rotates, or rewrites capture
data. A failed audit is an investigation signal, not authorization to edit raw
files.

### Continuity policy

- Missing BBO, L2-book, or asset-context hourly partitions fail the audit.
- Missing trade or candle partitions are reported as warnings because those
  feeds may legitimately be sparse during a quiet hour.
- Malformed JSON, invalid records, duplicate/conflicting trade identities,
  receive/exchange-time reversals, and invalid derived books fail the audit.
- Crossed BBO rows are counted and warned: the recorder intentionally retains
  their raw sides while rejecting their derived spread metrics.
- Missing, malformed, or bucket-incomplete 1m/5m aggregates warn. Aggregate
  coverage is measured independently for BTC and ETH from the capture-start
  bucket through each symbol's latest raw receive-time bucket.
- Ordering is assessed in append order independently for every stream/symbol.
  First/last receive and exchange timestamps and rows per UTC hour are reported.

## Offline aggregation

Run aggregation separately when research aggregates need refreshing. The
aggregator makes no network requests and does not modify raw capture files.

```bash
cd /home/ubuntu/bot
.venv/bin/python tools/hyperliquid_market_aggregate.py \
  --capture-root data/hyperliquid_market_capture
```

## Daily operator check

```bash
cd /home/ubuntu/bot
systemctl is-active hl-market-recorder.service
.venv/bin/python tools/hyperliquid_market_recorder_status.py \
  --capture-root data/hyperliquid_market_capture
df -h /home/ubuntu/bot/data/hyperliquid_market_capture
sudo journalctl -u hl-market-recorder.service --since '24 hours ago' --no-pager
```

Investigate `FAILED` immediately. `WARNING` means data may still be valid but a
stream is beyond its diagnostic warning interval, the lock owner cannot be
verified locally, a crossed BBO was observed, a naturally sparse trade/candle
hour has no file, or aggregate coverage is absent.
