# SAM Architecture

SAM is a thin orchestration layer over [ML4T](https://github.com/orgs/ml4t/repositories). **SAM orchestrates; ML4T computes, validates, backtests, and executes.** A single `Strategy` subclass is shared by `ml4t.backtest.Engine` and `ml4t.live.LiveEngine`.

## System context

```mermaid
flowchart TB
  subgraph operator["Operator"]
    CLI["sam CLI"]
    CFG["YAML configs + .env"]
  end

  subgraph sam["SAM (orchestration)"]
    direction TB
    CLI --> Config["sam.config"]
    CLI --> Pipe["sam.pipeline"]
    CLI --> Live["sam.live"]
    CLI --> Ops["sam.ops"]
    Config --> Strat["sam.strategies"]
    Pipe --> Strat
    Live --> Strat
    Pipe --> Man["sam.manifests"]
    Live --> Man
    Ops --> Man
  end

  subgraph ml4t["ML4T libraries"]
    Data["ml4t-data"]
    BT["ml4t-backtest"]
    LiveLib["ml4t-live"]
    Eng["ml4t-engineer"]
    Diag["ml4t-diagnostic"]
    Mod["ml4t-models"]
  end

  CFG --> Config
  Pipe --> Data
  Pipe --> BT
  Pipe --> Eng
  Pipe --> Diag
  Pipe --> Mod
  Live --> LiveLib
  BT --> Artifacts["artifacts/ + state/"]
  LiveLib --> Artifacts
  Data --> Artifacts
  Man --> Artifacts
```

## Layers

| Layer | Package | Responsibility |
|-------|---------|----------------|
| CLI | `sam.cli` | Argument parsing; dispatch to pipeline, live, research, ops |
| Config | `sam.config` | YAML + `SamSettings` (paths, API keys); no quant logic |
| Strategies | `sam.strategies` | Signal and rule implementations registered for backtest/live |
| Pipeline | `sam.pipeline` | Data sync/validate, backtest runs, research steps |
| Live | `sam.live` | `LiveEngine` + `SafeBroker` + feeds (fixture replay, Alpaca, IB) |
| Manifests | `sam.manifests` | `run_manifest.json` — promotion state, hashes, artifact paths |
| Ops | `sam.ops` | Status, brief, kill-switch, explicit `sam ops promote` |

## Request flow (backtest)

```mermaid
sequenceDiagram
  participant Op as Operator
  participant CLI as sam backtest run
  participant SAM as sam.pipeline
  participant Strat as sam.strategies
  participant ML4T as ml4t.backtest.Engine
  participant Disk as artifacts/

  Op->>CLI: configs/backtest/*.yaml
  CLI->>SAM: BacktestRunConfig
  SAM->>Strat: build_strategy()
  SAM->>ML4T: DataFeed + Engine.run()
  ML4T->>Disk: parquet + metrics.json
  SAM->>Disk: run_manifest.json
  Note over SAM,Disk: promotion_state = backtest_passed if gates pass
```

## Request flow (live)

```mermaid
sequenceDiagram
  participant Op as Operator
  participant CLI as sam live *
  participant SAM as sam.live.runner
  participant ML4T as ml4t.live
  participant Broker as SafeBroker / fixture / Alpaca / IB
  participant Disk as artifacts/live + state/

  Op->>CLI: live + environment YAML
  CLI->>SAM: LiveRunConfig (merged risk)
  alt shadow
    SAM->>Broker: fixture replay or Alpaca (if keys set)
    Note over Disk: state/shadow_risk.json
  else paper / live
    SAM->>Broker: Alpaca or IB
    Note over Disk: state from live config (e.g. ma_baseline_risk.json)
  end
  SAM->>ML4T: LiveEngine.run()
  SAM->>Disk: run_manifest.json under artifacts/live/{shadow|paper|live}/
```

## Promotion lifecycle

Stages are recorded in `run_manifest.json`. Most transitions are **automatic** when a command completes; operators can also advance state explicitly with `sam ops promote`.

```mermaid
stateDiagram-v2
  [*] --> research: data sync / failed backtest
  research --> backtest_passed: sam backtest run (gates pass)
  backtest_passed --> shadow: sam live shadow
  shadow --> paper: sam live paper
  paper --> live: sam live live
  note right of shadow
    Manifest: artifacts/live/shadow/
    Risk state: state/shadow_risk.json
  end note
  note right of paper
    Requires prior shadow manifest
    unless require_shadow_before_paper: false
  end note
  note right of live
    Requires prior paper manifest
    unless require_paper_before_live: false
  end note
```

| Stage | Typical command | Manifest location |
|-------|-----------------|-------------------|
| `research` | `sam data sync`, failed backtest | `data/.../run_manifest.json` |
| `backtest_passed` | `sam backtest run` (checks passed) | `artifacts/backtest/<run>/` |
| `shadow` | `sam live shadow` | `artifacts/live/shadow/` |
| `paper` | `sam live paper` | `artifacts/live/paper/` |
| `live` | `sam live live` | `artifacts/live/live/` |

Explicit promotion (e.g. after manual review):

```bash
sam ops promote --from research --to backtest_passed \
  --manifest artifacts/backtest/ma_baseline/run_manifest.json
```

## Repository layout

```
sam/
├── configs/          # backtest, live, data, environments, strategies, universes
├── deploy/           # launchd/cron examples
├── docs/             # ARCHITECTURE.md, RUNBOOK.md
├── scripts/          # fixture generation
├── src/sam/          # package source
├── tests/            # unit + integration (fixtures under tests/fixtures/)
├── artifacts/        # gitignored — run outputs
├── data/             # gitignored — synced market data
└── state/            # gitignored — risk / kill-switch JSON
```

## Realism controls

Backtest configs declare calendar, timezone, data frequency, commission, slippage, data window, corporate-action assumption, signal lag, and promotion gates. Live configs declare exposure, loss, order-rate, data-staleness, drawdown, and kill-switch limits before wiring into `ml4t.live.LiveRiskConfig`.

## Operator surface

- `sam ops status` — local risk files, manifests, data freshness; optional `--broker-snapshot` (Alpaca when configured).
- `sam ops brief` — markdown/json summary for scheduled checks.
- `sam ops kill-switch` — activate or clear kill-switch on a chosen `state/*.json` file.
- `sam ops preflight` — broker connectivity check via `SafeBroker`.

See [RUNBOOK.md](RUNBOOK.md) for step-by-step promotion and drills.
