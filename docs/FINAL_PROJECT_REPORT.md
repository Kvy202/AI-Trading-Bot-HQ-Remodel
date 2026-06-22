<!--
HOW TO USE THIS FILE
====================
1. This is your full project report in the standard B.Tech format.
2. Replace every [FILL: ...] with your real details.
3. Create a folder docs/images/ and drop your screenshots there with these EXACT names:
     fig-architecture.png            (your architecture diagram)
     fig-github-exchanges.png        (GitHub exchanges/ folder tree)
     fig-dashboard-desktop.png       (desktop dashboard)
     fig-dashboard-mobile.png        (mobile/responsive dashboard)
     fig-writer-terminal.png         (live_writer.py terminal)
     fig-executor-guardrail.png      (live_executor.py guardrail = PAPER)
     fig-pytest.png                  (75 passed)
     fig-hyperliquid-testnet.png     (Hyperliquid testnet account page)
4. Paste into Google Docs / Word, apply heading styles, export to PDF.
5. SECURITY: every screenshot must hide private keys, the agent key, and the .env contents.
   A public wallet address is acceptable but you may blur it.
-->

# Design and Implementation of a Risk-Managed AI Trading Bot on the Hyperliquid Decentralized Exchange

Project report submitted to
[FILL: University Name]
for the partial award of the degree of
**Bachelor of Technology — [FILL: Branch]**
([FILL: 20XX–20XX])

**Submitted To:** [FILL: Supervisor name(s) & designation]
**Submitted By:** [FILL: Your name] ([FILL: Roll/Enrolment No.])

Department of [FILL: Department]
[FILL: University Name]
[FILL: Month, Year]

---

## DECLARATION

I, [FILL: Your name] ([FILL: Roll No.]), certify that the work contained in this project report is
original and has been carried out by me under the guidance of my supervisor. This work has not
been submitted to any other institute for the award of any degree or diploma, and I have followed
the ethical practices and guidelines of the Department. Wherever I have used materials (data,
analysis, figures, and text) from other sources, I have given due credit by citing them in the text
and listing them in the references.

Signature
[FILL: Your name] — [FILL: Roll No.]
[FILL: Department, University]

Signature
[FILL: Supervisor name(s)]
[FILL: Department, University]

---

## ACKNOWLEDGEMENT

I would like to extend my sincere thanks to everyone who supported this project. I am highly
grateful to **[FILL: Supervisor name(s)]** for their supervision, guidance, and valuable feedback
throughout its completion. I also thank my parents and my peers for their continuous
encouragement and support during the development phase.

---

## CERTIFICATE

This is to certify that **[FILL: Your name] ([FILL: Roll No.])**, student of B.Tech [FILL: Branch],
Department of [FILL: Department], [FILL: University], has completed the project entitled
**"Design and Implementation of a Risk-Managed AI Trading Bot on the Hyperliquid Decentralized
Exchange."**

[FILL: Supervisor name(s)]
Department of [FILL]
[FILL: University]

---

## CONTENTS

1. **Introduction** — Introduction, Problem Formulation, Objectives, Tools & Technologies, Structure
2. **Background Details and Literature Review** — Background, Literature Review, Existing Systems
3. **Design / Framework** — Methodology, Technologies, System Design, Software & Hardware Requirements, Workflow, Data Flow Diagram
4. **Discussion and Analysis of Results** — Discussion & Analysis, Output
5. **Conclusion and Future Scope** — Conclusion, Relevance/Scope/Future Work
6. **References / Bibliography**
7. **Appendices**

---

## ABSTRACT

Algorithmic trading on cryptocurrency markets has grown rapidly, but most retail systems rely on
custodial centralized exchanges and offer little transparency or safety tooling. This project designs
and implements a **risk-managed, machine-learning trading bot for Hyperliquid**, a high-performance
**decentralized perpetual-futures exchange**. Order execution is built on the **official Hyperliquid
Python SDK** (not CCXT) behind a clean, swappable *exchange-adapter* architecture, allowing the same
trading engine to run on Hyperliquid or, as a legacy/comparison layer, on Bitget.

A deep-learning ensemble (LSTM, TCN, Transformer, and an advanced variant) generates directional
signals from price, volatility, volume, and order-book microstructure features. A multi-layer risk
framework — confidence thresholds, volatility and concurrency limits, cooldowns, portfolio-exposure
caps, take-profit/stop-loss, a wall-clock time-stop, daily loss/drawdown limits, and a file-based kill
switch — governs every order. A central **safety guardrail** keeps the system in paper mode by default;
real-money trading is impossible without an explicit, multi-condition confirmation.

The system was validated in **paper and testnet** modes with an automated test suite (75 passing tests),
a live monitoring dashboard, and reproducible AWS EC2 deployment. The results show a **complete and
robust execution, risk, and operations framework**; consistent trading profit is identified as an open
research problem, with a tiered roadmap proposed to add predictive edge. The project demonstrates
secure DEX automation, modular software design, and responsible, safety-first engineering.

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Introduction

Cryptocurrency derivatives trade 24/7 across volatile, fast-moving markets. Manually monitoring
positions and reacting to price changes is impractical, which motivates automated trading systems.
At the same time, **centralized exchanges (CEXs)** introduce custodial risk (the exchange holds user
funds), opaque execution, and account restrictions. **Decentralized exchanges (DEXs)** such as
**Hyperliquid** offer non-custodial, on-chain perpetual-futures trading with an **agent-wallet** model
that separates *signing* authority from *withdrawal* authority — a strong security primitive for bots.

This project remodels an existing machine-learning trading bot so that its execution layer runs on
Hyperliquid through the **official Hyperliquid Python SDK**, while preserving the deep-learning signal
pipeline, backtesting, risk controls, logging, and reporting. The central design goals are **safety**
(live trading off by default, behind explicit confirmations), **modularity** (a venue-neutral exchange
adapter), and **reproducible cloud deployment** (a dedicated AWS EC2 instance).

**Highlights:**
- Hyperliquid execution via the official SDK (not CCXT), behind an `ExchangeAdapter` interface.
- Safety guardrail: paper trading by default; real money requires a full confirmation set.
- Deep-learning ensemble with calibration, model-agreement, and out-of-distribution guards.
- Multi-layer risk management plus a file-based emergency kill switch.
- Live monitoring dashboard, automated tests, and AWS deployment templates.

## 1.2 Problem Formulation

Building a *safe*, automated DEX trading bot raises several challenges:

1. **Custodial and key-security risk.** A bot needs signing access but must never be able to withdraw
   funds, and its private key must never leak into logs, reports, or version control.
2. **Accidental real-money trading.** A single misconfiguration could place live orders. The system
   must make this effectively impossible without deliberate, explicit confirmation.
3. **Venue lock-in.** Execution logic tightly coupled to one exchange's API is hard to port or compare.
4. **Signal reliability.** Market signals must be calibrated and guarded against out-of-distribution
   inputs to avoid systematically biased trades.
5. **Operability.** A long-running bot needs monitoring, health checks, restartability, and a fast
   kill switch.
6. **Reproducible deployment.** The system must deploy cleanly to a fresh cloud server without
   touching any existing production system.

## 1.3 Objectives of the Project

- **Replace the execution layer** with the official Hyperliquid Python SDK (not CCXT).
- **Design a modular adapter architecture** (`ExchangeAdapter` interface, `HyperliquidSDKAdapter`,
  and a legacy `BitgetAdapter`) selectable by configuration.
- **Use an agent/API wallet for signing**; never require or store the main wallet's private key.
- **Disable live trading by default** and gate real money behind explicit, multi-condition confirmation.
- **Preserve** the DL signal pipeline, risk controls, logging, and reporting.
- **Provide a test suite, a monitoring dashboard, and reproducible AWS EC2 deployment.**
- **Evaluate honestly** in paper/testnet mode and document a roadmap for future improvement.

## 1.4 Tools and Technologies

**Languages & ML:** Python 3.13; PyTorch (CPU) for the deep-learning ensemble; scikit-learn for
feature scaling; NumPy/Pandas for data handling.

**Exchange / data:** official `hyperliquid-python-sdk` for execution and account/price queries;
`eth-account` for agent-wallet signing; `ccxt` used only for **public OHLCV** market data (Phase 1).

**Backend / services:** Flask (monitoring dashboard API); a signal-writer/executor process pipeline;
systemd services for process management; an optional signed supervisor control plane; optional
Telegram bots for alerts/control.

**Data / storage:** CSV trade logs and JSON state/heartbeat files; an optional SQLite store for Tier-2
shadow data.

**Deployment & tooling:** AWS EC2 (Ubuntu) + Nginx; Git/GitHub for version control; Visual Studio
Code; pytest for automated testing.

## 1.5 Structure of the Project Report

- **Chapter 1 – Introduction:** background, problem statement, objectives, tools, and report structure.
- **Chapter 2 – Background & Literature Review:** domain background, related research, existing systems.
- **Chapter 3 – Design / Framework:** methodology, technologies, system design, requirements, workflow,
  and data-flow diagrams.
- **Chapter 4 – Discussion & Analysis of Results:** validation, performance, limitations, and output
  screenshots.
- **Chapter 5 – Conclusion & Future Scope:** summary, relevance, and future work.
- **References** and **Appendices** (run instructions, key files, code snippets).

---

# CHAPTER 2 — BACKGROUND DETAILS AND LITERATURE REVIEW

## 2.1 Background Details

Automated trading uses software to generate and execute orders from market data without manual
intervention. In crypto, **perpetual futures** (perps) are the dominant derivative: leveraged contracts
with no expiry, kept near the spot price by a periodic **funding rate**.

**Hyperliquid** is a decentralized perpetual-futures exchange with an on-chain order book and a Python
SDK. A distinctive feature is the **agent (API) wallet**: a key that can place and cancel orders on behalf
of a main account but **cannot withdraw funds**. The bot therefore signs with the agent key while
querying account state by the **main wallet's public address** — giving automation power without
custody of withdrawal rights. This separation is central to the project's security model.

Machine learning, particularly **deep learning on financial time series**, is widely used to model
short-horizon price direction. This project combines four model families — **LSTM**, **TCN**
(temporal convolutional network), **Transformer**, and an advanced variant — into a calibrated
**ensemble**, with guards that reject out-of-distribution inputs and require model agreement before a
signal is acted upon.

## 2.2 Literature Review

**Deep learning for financial time series.** Recurrent and convolutional architectures (LSTM/GRU,
TCN) and, more recently, attention-based Transformers have been applied to price and volatility
forecasting. The literature consistently notes that financial signals have a **low signal-to-noise ratio**
and are prone to overfitting and regime change, motivating calibration, ensembling, and conservative
position sizing rather than raw prediction accuracy.

**Algorithmic and risk-managed trading.** Studies on systematic strategies emphasise that
**risk management and execution quality** often matter more than predictive accuracy: drawdown
control, position sizing, fees, and slippage frequently determine real-world outcomes. This informs the
project's multi-layer risk framework and realistic paper-fill model (fees + adverse slippage).

**Decentralized exchanges and on-chain execution.** DEX perpetual venues remove custodial risk
but introduce new considerations: agent-wallet permissions, on-chain order semantics, size/price
rounding rules, and minimum order values. The project addresses these directly in its Hyperliquid
adapter.

**Ensemble methods and calibration.** Combining diverse models and calibrating their probabilities
(temperature/bias) is a well-established technique for improving reliability of classifier outputs, used
here to convert model outputs into trustworthy trade gates.

> [FILL: optionally cite 3–5 specific papers/books from the References section in-line, e.g.
> (Mitchell, 2018); (Sezer et al., 2020 — financial time-series DL survey); (López de Prado, 2018).]

## 2.3 Existing Systems

- **Centralized-exchange trading bots** (e.g., CCXT-based bots, 3Commas, Freqtrade): mature and
  flexible, but custodial and CEX-focused; they do not natively target DEX agent-wallet execution.
- **DEX trading interfaces** (the Hyperliquid web app and similar): manual trading UIs, not automated
  ML systems.
- **Quant frameworks** (Backtrader, Freqtrade): strong backtesting/strategy tooling, but not built
  around a non-custodial DEX SDK or a safety-first live-trading guardrail.

**Limitations of existing systems addressed here:** lack of a venue-neutral adapter that includes a
DEX (Hyperliquid) via its official SDK; weak or absent guardrails preventing accidental real-money
trading; and limited integrated monitoring/kill-switch tooling for a long-running bot. This project
targets these gaps with a modular adapter layer, a single safety decision point, and an operations
stack.

---

# CHAPTER 3 — DESIGN / FRAMEWORK

## 3.1 Methodology

The project followed an incremental, safety-first methodology:

1. **Analysis & planning:** audit the existing bot, isolate exchange-specific code, define the adapter
   interface, and plan an execution-only migration (preserve the proven signal/risk code).
2. **Adapter design:** extract the legacy execution logic into `BitgetAdapter`; implement
   `HyperliquidSDKAdapter` on the official SDK; add a factory to select the venue by configuration.
3. **Safety layer:** centralise the live/paper decision in one guardrail that is safe by default.
4. **Configuration:** typed settings with secret redaction; environment-driven `.env` + `run.json`.
5. **Testing:** unit tests for settings, guardrails, adapter selection, the mocked Hyperliquid adapter,
   and the risk controls.
6. **Operations & deployment:** monitoring dashboard, systemd services, and AWS EC2 templates.
7. **Evaluation:** run in paper/testnet, collect evidence, and document results and limitations honestly.

## 3.2 Technologies

**Frontend (dashboard):** server-rendered HTML/CSS/JavaScript served by Flask; responsive layout
with horizontally scrollable tables for mobile.

**Backend / engine:** Python; a **signal writer** (`tools/live_writer.py`) that runs the DL ensemble and
writes signals to a CSV; a **signal executor** (`tools/live_executor.py`) that applies risk gates and
routes orders through the selected adapter.

**Execution:** official `hyperliquid-python-sdk` (`Info` for market/account data, `Exchange` for orders)
with `eth-account` signing; legacy Bitget via `ccxt`.

**ML:** PyTorch ensemble (LSTM/TCN/Transformer/advanced); scikit-learn `StandardScaler`; 27 input
features (26 base features + a per-symbol id channel).

**Data:** public OHLCV via `ccxt` (Phase 1); CSV/JSON logs; optional SQLite Tier-2 store.

**Ops:** systemd, Nginx, AWS EC2, Git/GitHub, pytest.

## 3.3 System Design

The system is a four-stage pipeline with a safety decision point in front of any order:

![Figure 3.1 — System architecture: data → signal writer → signals CSV → executor → guardrail → adapter (Hyperliquid SDK / legacy Bitget)](images/fig-architecture.png)

**Core components:**

- **Signal Writer** — loads the 4-model DL ensemble and feature pipeline, computes signals per symbol,
  and appends them to `logs/live_signals.csv`.
- **Signal Executor** — reads new signals and applies risk gates (whitelist, threshold, volatility,
  concurrency, cooldown, duplicate-fill guard, side-bias lock, portfolio cap, TP/SL, time-stop).
- **Guardrail (`runtime/guardrails.py`)** — the single authority that resolves the trading mode to
  **PAPER**, **TESTNET_LIVE**, or **MAINNET_LIVE**; safe by default.
- **Exchange Adapter (`exchanges/`)** — a venue-neutral interface with `HyperliquidSDKAdapter`
  (official SDK) and `BitgetAdapter` (legacy, ccxt), chosen by a factory from the `EXCHANGE` setting.
- **Risk Controls (`v2/risk_controls.py`)** — daily loss/drawdown limits, time-stop, and a file-based
  kill switch.
- **Monitoring** — a Flask dashboard plus heartbeat/health files; optional Telegram and supervisor
  control plane.

The modular adapter package is shown below:

![Figure 3.2 — Exchange-adapter package (base, hyperliquid_adapter, bitget_adapter, factory, types)](images/fig-github-exchanges.png)

## 3.4 Software and Hardware Requirements

**A. Software Requirements**

| Component | Description |
|---|---|
| Operating System | Ubuntu Linux (server) / Windows 10+ (development) |
| Language | Python 3.13 |
| Execution SDK | hyperliquid-python-sdk, eth-account |
| Market data | ccxt (public OHLCV) |
| ML | PyTorch (CPU), scikit-learn, NumPy, Pandas |
| Web / API | Flask |
| Process mgmt | systemd; Nginx (reverse proxy) |
| Testing | pytest |
| Version control | Git / GitHub |
| Editor | Visual Studio Code |

**B. Hardware Requirements (minimum)**

| Specification | Minimum |
|---|---|
| Processor | Dual-core (Intel i3 / AWS t3.small) |
| RAM | 4 GB (8 GB recommended) |
| Storage | 20 GB free |
| Network | Stable internet (HTTPS outbound to the exchange API) |

**C. Cloud / Deployment**

| Component | Description |
|---|---|
| Compute | AWS EC2 (Ubuntu, t3.small) — dedicated instance |
| Secrets | `.env` (chmod 600) or AWS SSM Parameter Store / Secrets Manager |
| Web | Nginx + HTTPS (dashboard at a subdomain) |

## 3.5 Workflow

**Step-by-step:**

1. **Configuration load** — `EXCHANGE`, mode flags, and (for live) Hyperliquid credentials are read;
   secrets are redacted in logs.
2. **Guardrail decision** — `resolve_trading_mode()` returns PAPER unless every live condition is met.
3. **Signal generation** — the writer runs the DL ensemble on fresh features and writes one signal per
   symbol per tick to `live_signals.csv`.
4. **Risk gating** — the executor evaluates each signal against all risk gates; rejected signals are
   logged with a reason (e.g., `SKIP … reason=already_long`).
5. **Order routing** — accepted signals are sized (USDT notional → base quantity) and routed through
   the adapter; in **paper** mode fills are simulated locally (with fees + slippage) and never sent
   on-chain.
6. **Logging & monitoring** — trades/closes are written to per-day CSVs; the dashboard and heartbeat
   files reflect live state.
7. **Safety** — the file-based kill switch (`run/V2_PAUSE`) blocks new entries within one poll; exits
   keep running.

```
[Config] -> [Guardrail: PAPER/TESTNET/MAINNET] -> [Signal Writer] -> [live_signals.csv]
        -> [Executor: risk gates] -> [Exchange Adapter] -> [Paper fills / on-chain orders]
        -> [Logs + Dashboard + Kill switch]
```

## 3.6 Data Flow Diagram

**Level 0 (context):**

```
[User/Operator] --config/commands--> ( AI Trading Bot ) <--market data-- [Exchange API]
                                       |  orders (live mode) ----------> [Hyperliquid]
                                       |  state/logs -------------------> [Dashboard / Files]
```

**Level 1 (decomposed):**

```
[Market Data] -> (1. Feature + Signal Generation) -> [live_signals.csv]
[live_signals.csv] -> (2. Risk Gating + Guardrail) -> (3. Order Routing via Adapter)
(3) -> [Hyperliquid (live)] | [Paper fill simulation]
(2,3) -> (4. Logging/State) -> [CSV/JSON logs] -> (5. Monitoring Dashboard) -> [Operator]
```

- **External entities:** Operator, Exchange API (Hyperliquid), Dashboard viewer.
- **Processes:** signal generation, risk gating/guardrail, order routing, logging, monitoring.
- **Data stores:** `live_signals.csv`, trade/closed CSVs, `executor_state.json`, heartbeat JSON, model
  artifacts.

---

# CHAPTER 4 — DISCUSSION AND ANALYSIS OF RESULTS

## 4.1 Discussion and Analysis

The system was validated end-to-end in **paper and testnet** modes. Validation focused on correctness
and safety of the framework rather than trading profitability.

**Functionality validation**
- The signal writer loads the full 4-model ensemble (LSTM, TCN, Transformer, advanced) for 6 symbols
  and emits signals continuously (Figure 4.3).
- The executor consumes signals, applies all risk gates, and logs accepted/rejected decisions with
  reasons (Figures 4.1, 4.4).
- The monitoring dashboard renders open positions, latest signals, recent paper trades, and closed-trade
  PnL in real time, on both desktop and mobile (Figures 4.1, 4.2).

**Safety validation (key result)**
- The guardrail resolves to **PAPER by default**: `trading_mode=PAPER … real_orders=False … live
  trading not requested` (Figure 4.4). Secrets are redacted in every log line
  (`hl_agent_private_key=0x36…(redacted)`).
- Real-money (mainnet) trading is unreachable unless **all** of `LIVE_TRADING=true`,
  `PAPER_TRADING=false`, `ENVIRONMENT=production`, `HL_TESTNET=false`, and
  `CONFIRM_LIVE_TRADING=I_UNDERSTAND_LIVE_TRADING` hold, with valid credentials.
- A file-based kill switch (`run/V2_PAUSE`) blocks new entries within one poll.

**Automated testing**
- The test suite passes **75/75** (Figure 4.5), covering configuration loading, the live-trading
  guardrails, adapter selection, the mocked Hyperliquid adapter (order-payload building, size rounding,
  symbol mapping), and the V2 risk controls / kill switch — all offline, with no real keys.

**Testnet connectivity**
- An approved agent wallet connects to Hyperliquid **testnet** with mock USDC (Figure 4.6: account
  `0xe344…2219`, Portfolio Value ≈ $999). Because the bot runs in **paper** mode, no on-chain position
  is opened ("No open positions yet"); positions are simulated locally and shown on the dashboard.
  Placing real testnet orders through the adapter is a documented next step.

**Performance metrics (paper/testnet)**

| Aspect | Observation |
|---|---|
| Models loaded | 4 (LSTM, TCN, Transformer, advanced), 6 symbols |
| Automated tests | 75 passed |
| Default mode | PAPER (real_orders = False) |
| Signal cadence | ~ every few seconds per symbol |
| Dashboard refresh | ~1 s (real-time) |
| Realized PnL (sample paper day) | ≈ **−0.39** (net of simulated fees + slippage) |
| Dominant exit reason | FLIP_CLOSE (frequent direction flips) |

**Honest evaluation.** In paper trading the current signal does **not** yet produce consistent profit:
exits are dominated by **flip-churn** at a tight threshold, and the sample day's realized PnL is slightly
negative. This is treated as a **finding**, not a failure — the execution, risk-management, monitoring,
and deployment framework is complete and tested, while generating durable *alpha* is the open problem.
Section 5.2 proposes a tiered roadmap to address it. Limitations observed include simulated (not native)
TP/SL, frequent flips at the current threshold, and market data still sourced from public OHLCV rather
than the Hyperliquid Info API.

## 4.2 Output

**4.2.1 Live Dashboard (Desktop)**

![Figure 4.1 — Live dashboard (desktop): open positions, latest signals, recent paper trades, and closed-trade PnL](images/fig-dashboard-desktop.png)

**4.2.2 Live Dashboard (Mobile / Responsive)**

![Figure 4.2 — Responsive mobile dashboard with horizontally scrollable tables](images/fig-dashboard-mobile.png)

**4.2.3 Signal Writer (Model Ensemble Load)**

![Figure 4.3 — Signal writer loading the 4-model ensemble for 6 symbols](images/fig-writer-terminal.png)

**4.2.4 Executor Safety Guardrail (PAPER mode)**

![Figure 4.4 — Executor resolving to PAPER mode with secrets redacted](images/fig-executor-guardrail.png)

**4.2.5 Automated Test Suite**

![Figure 4.5 — pytest: 75 passed](images/fig-pytest.png)

**4.2.6 Hyperliquid Testnet Account (Agent Connectivity)**

![Figure 4.6 — Hyperliquid testnet account funded with mock USDC; no on-chain position while in paper mode](images/fig-hyperliquid-testnet.png)

---

# CHAPTER 5 — CONCLUSION AND FUTURE SCOPE

## 5.1 Conclusion

This project successfully designed and implemented a **risk-managed AI trading bot for the Hyperliquid
decentralized exchange** using the official Hyperliquid Python SDK. The execution layer was rebuilt
behind a clean, venue-neutral **adapter architecture**, with a legacy Bitget adapter retained for
comparison. A **single safety guardrail** keeps the system in paper mode by default and makes
accidental real-money trading effectively impossible, while an **agent-wallet** signing model provides
automation without custody of withdrawal rights.

The deep-learning signal pipeline, multi-layer risk controls, monitoring dashboard, and reproducible
AWS deployment were preserved and validated, with a **75-passing automated test suite** and live
paper/testnet evidence. The project meets its core objectives of secure DEX automation, modular
design, and safety-first engineering. Honest evaluation shows a complete framework whose remaining
challenge is generating consistent trading edge — a problem addressed by the roadmap below.

## 5.2 Relevance, Scope, and Future Work

**Relevance & current scope.** As decentralized derivatives grow, non-custodial automated trading with
strong safety guarantees is increasingly relevant. The current system implements: Hyperliquid
execution via the official SDK; a modular adapter layer; the full DL ensemble and feature pipeline;
multi-layer risk controls and a kill switch; a monitoring dashboard; an automated test suite; and AWS
deployment templates. Telegram alerts/control and the Tier-2 shadow-data layer exist in the codebase
but are not enabled in the current deployment.

**Feature tiers — implemented vs. roadmap:**

| Tier | Capability | Status |
|---|---|---|
| 1 | DL ensemble, technical/volatility/volume features, microstructure, risk controls | Implemented |
| 1.5 | Heartbeat, watchdog/auto-restart, unified config, log rotation | Implemented |
| 1.5 | Telegram controller + notifier bots | Implemented (not enabled in this deployment) |
| 2 | Funding-rate / open-interest collectors + shadow store | Implemented (shadow; collectors not enabled here) |
| 2 | Token-unlock schedule, news/sentiment, whale/CEX flow | Future work |
| 3 | Attention-based signal routing, regime detection, social sentiment | Future work |
| 4 | On-chain/dev-activity, Google Trends, macro correlations (DXY/Gold/S&P) | Future work |

**Specific future enhancements:**
- **Native Hyperliquid trigger orders** (on-chain TP/SL) instead of executor-simulated exits.
- **Place small testnet orders** through the adapter to demonstrate full on-chain execution.
- **Market data on Hyperliquid** — migrate features to the Hyperliquid Info API (candles, mids, L2 book).
- **Reduce flip-churn** — higher/adaptive thresholds, flip-confirmation, regime-aware sizing.
- **Enable Telegram + supervisor control** on the server for full remote operability.
- **Account-equity drawdown enforcement** and volatility-scaled position sizing.
- A formal **Bitget-vs-Hyperliquid execution-quality comparison** (a configuration change, thanks to
  the adapter layer).

---

# REFERENCES / BIBLIOGRAPHY

**Books & Academic Resources**
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly.

**Journals & Articles**
- Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). "Financial time series forecasting with deep
  learning: A systematic literature review." *Applied Soft Computing*, 90, 106181.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling." *arXiv:1803.01271* (TCN).
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS* (Transformer).
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8).
- [FILL: optionally add 1–2 more domain papers you read.]

**Websites & Documentation**
- Hyperliquid Python SDK — https://github.com/hyperliquid-dex/hyperliquid-python-sdk
- Hyperliquid Documentation — https://hyperliquid.gitbook.io/hyperliquid-docs
- eth-account — https://eth-account.readthedocs.io/
- CCXT — https://docs.ccxt.com/
- PyTorch — https://pytorch.org/docs/
- scikit-learn — https://scikit-learn.org/stable/
- Flask — https://flask.palletsprojects.com/
- AWS EC2 — https://docs.aws.amazon.com/ec2/
- Project repository — https://github.com/Kvy202/Ai-Trading-Bot-HQ
- Live demonstration dashboard — https://trading-bot.tradyai.live

---

# APPENDICES

## Appendix A — How to Run the Project (paper / testnet)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure (safe defaults: paper + testnet)
cp .env.example .env
#   EXCHANGE=hyperliquid, LIVE_TRADING=false, PAPER_TRADING=true, HL_TESTNET=true
#   DL_ADD_SYMBOL_ID=1, and HL_ACCOUNT_ADDRESS / HL_AGENT_PRIVATE_KEY for testnet

# 3. Verify the safety guardrail resolves to PAPER
python -c "from runtime.settings import Settings; from runtime.guardrails import resolve_trading_mode as r; print(r(Settings.from_env()).describe())"

# 4. Run the pipeline
python tools/live_writer.py                                     # signals
python tools/live_executor.py --signals logs/live_signals.csv  # executor (paper)

# 5. Tests
python -m pytest        # 75 passed
```

## Appendix B — Key Project Files

- `exchanges/base.py` — `ExchangeAdapter` interface
- `exchanges/hyperliquid_adapter.py` — Hyperliquid execution (official SDK)
- `exchanges/bitget_adapter.py` — legacy Bitget execution (ccxt)
- `exchanges/factory.py` — venue selection
- `runtime/settings.py` — typed config + secret redaction
- `runtime/guardrails.py` — live-trading safety decision
- `tools/live_writer.py` — DL-ensemble signal generation
- `tools/live_executor.py` — risk gates + order routing
- `v2/risk_controls.py` — daily limits + kill switch
- `tools/dashboard.py`, `tools/dashboard_index.html` — monitoring dashboard
- `deploy/aws/` — EC2 setup guide + systemd templates

## Appendix C — Selected Code Snippet (safety guardrail, simplified)

```python
# runtime/guardrails.py — real money requires ALL confirmations, else PAPER
def resolve_trading_mode(settings, cli_live=False, cli_paper=False, log=None):
    if cli_paper:
        return PAPER  # --paper always wins
    requested = cli_live or (settings.live_trading and not settings.paper_trading)
    if not requested:
        return PAPER  # safe default
    if settings.exchange == "hyperliquid" and settings.hl_testnet:
        return TESTNET_LIVE if settings.has_hl_credentials else PAPER
    # mainnet real money: needs LIVE_TRADING, !PAPER_TRADING, ENVIRONMENT=production,
    # HL_TESTNET=false, CONFIRM_LIVE_TRADING=I_UNDERSTAND_LIVE_TRADING, valid creds
    return MAINNET_LIVE if all_confirmations_met(settings) else PAPER
```

## Appendix D — Ethical Disclaimer

This software is for **educational and research purposes only**. It is not financial advice and makes no
guarantee of profit. Trading cryptocurrency derivatives can result in the total loss of capital. The
author accepts no liability for any losses. Private keys and API secrets must never be shared, committed
to version control, or exposed in logs, reports, or screenshots; demonstrations should use **testnet and
paper trading**.
