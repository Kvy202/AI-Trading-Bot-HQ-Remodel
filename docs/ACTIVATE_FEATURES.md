# Activating Optional Features

Three independent features can be enabled without touching any trading logic:
1. **Hyperliquid Testnet Live Orders** — place real orders on the testnet DEX
2. **Tier 2 Shadow Data** — collect funding rates + open interest
3. **Telegram Alerts** — health monitoring and remote control

---

## 1  Hyperliquid Testnet Live Orders

> Currently the bot runs in **PAPER mode** — fills are simulated locally. To
> place real orders on `app.hyperliquid-testnet.xyz` (no real money), apply
> the three `.env` changes below.

### What the guardrail requires for `TESTNET_LIVE`

| Condition | Current value | Required |
|---|---|---|
| `LIVE_TRADING` | `false` | `true` |
| `PAPER_TRADING` | `true` | `false` |
| `HL_TESTNET` | `true` | `true` ✓ |
| `HL_ACCOUNT_ADDRESS` | placeholder | real testnet wallet address |
| `HL_AGENT_PRIVATE_KEY` | placeholder | real agent key |

**No `CONFIRM_LIVE_TRADING` required** — that is only for real-money mainnet.

### Steps

**Step A — Create / verify your agent wallet on Hyperliquid testnet**

1. Go to `app.hyperliquid-testnet.xyz`
2. Portfolio → API → Create API wallet (this is the agent wallet).
3. Copy the agent private key and your main account address.
4. You already have mock USDC from the testnet faucet (Portfolio Value ≈ $999).

**Step B — Update `.env` on your machine (or on AWS)**

```env
# Change these three lines:
LIVE_TRADING=true
PAPER_TRADING=false
HL_TESTNET=true           # already true — leave it

# Fill in your real testnet values:
HL_ACCOUNT_ADDRESS=0xYourMainWalletPublicAddress    # 0x + 40 hex (public only)
HL_AGENT_PRIVATE_KEY=0xYourAgentWalletPrivateKey    # 0x + 64 hex (SECRET)
```

**Step C — Verify the guardrail resolved correctly before running**

```powershell
python -c "
from runtime.settings import Settings
from runtime.guardrails import resolve_trading_mode
s = Settings.from_env()
d = resolve_trading_mode(s)
print(d.describe())
"
```

Expected output:
```
GUARDRAIL trading_mode=TESTNET_LIVE exchange=hyperliquid real_orders=True testnet=True sandbox=False :: Hyperliquid testnet live (no real money)
```

**Step D — Run the executor**

```powershell
python tools/live_executor.py --signals logs/live_signals.csv
```

Watch the startup lines — you should see `trading_mode=TESTNET_LIVE` and
`real_orders=True`. Within the first few signal cycles a small order will appear
on `app.hyperliquid-testnet.xyz` → Portfolio → Positions.

> **Safety note:** to go back to paper instantly, either:
> - Add `--paper` flag: `python tools/live_executor.py --paper --signals ...`
> - Or set `PAPER_TRADING=true` in `.env` and restart.

---

## 2  Tier 2 Shadow Data (funding rate + open interest)

The collector runs as an independent process, has **zero influence** on
`live_signals.csv`, and refuses to start with `TIER2_SHADOW_ONLY=0`.

### Steps (local)

**Step A — Enable in `.env`**

```env
TIER2_ENABLED=1
TIER2_SHADOW_ONLY=1       # enforced inside the runner — leave at 1
EXCHANGE_ID=bitget        # public OHLCV endpoint for FR/OI data
```

**Step B — Run**

```powershell
python tier2/shadow_runner.py
```

You should see log lines like:
```
INFO tier2.runner: Tier 2 shadow runner starting — exchange=bitget symbols=['BTCUSDT','ETHUSDT'] interval=60s
INFO tier2.runner: cycle=1 collectors_run=2 db_rows={'funding_rate': 2, 'open_interest': 2} elapsed=1.2s
```

The dashboard's **Tier 2 Shadow Data** panel will then populate on the next refresh.

### On AWS (systemd)

```bash
# After deploying with the updated service files:
sudo cp deploy/aws/hl-tier2.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hl-tier2
sudo systemctl start hl-tier2
sudo systemctl status hl-tier2
```

---

## 3  Telegram Alerts & Remote Control

Two bots exist in the codebase:
- **Notifier** (`tools/telegram_notifier.py`) — health alerts + daily PnL summary (read-only)
- **Controller** (`tools/telegram_controller.py`) — remote pause/resume/stop via the Supervisor API

### Step A — Create Telegram bots via BotFather

1. Open Telegram → search **@BotFather** → `/newbot`
2. Create two bots: one for notifications, one for control (or reuse one token for both).
3. Get your **Chat ID**: message `@userinfobot` or send `/start` to your bot and call
   `https://api.telegram.org/bot<TOKEN>/getUpdates` to find the chat id.

### Step B — Add to `.env`

```env
TELEGRAM_NOTIFIER_TOKEN=123456789:ABCdefGhijKlmNopQrsTuvWxyZ   # notifier bot token
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhijKlmNopQrsTuvWxyZ        # fallback (same or different)
TELEGRAM_CHAT_ID=-1001234567890                                  # your chat / group id
```

### Step C — Run the notifier

```powershell
python tools/telegram_notifier.py
```

You will receive a startup message in Telegram. The notifier checks heartbeats
every 60 s and fires alerts if the writer or executor becomes stale.

### On AWS (systemd)

```bash
sudo cp deploy/aws/hl-telegram.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hl-telegram
sudo systemctl start hl-telegram
sudo systemctl status hl-telegram
```

---

## Summary: complete `.env` changes at a glance

| Goal | Variables to change |
|---|---|
| Testnet live orders | `LIVE_TRADING=true`, `PAPER_TRADING=false`, `HL_ACCOUNT_ADDRESS=0x…`, `HL_AGENT_PRIVATE_KEY=0x…` |
| Tier 2 shadow data | `TIER2_ENABLED=1` |
| Telegram alerts | `TELEGRAM_NOTIFIER_TOKEN=…`, `TELEGRAM_CHAT_ID=…` |

None of these changes affect the other features — they are fully independent.
