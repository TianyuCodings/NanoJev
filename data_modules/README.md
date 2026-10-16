# Optional Blackjack and real-market datasets

Two standalone datasets for experiments with the native decision trainer. These are optional additions; the unified gaming mixture, training configuration, checkpoint and reported game benchmarks are unchanged. Format compatibility does not imply improved mixed-task performance.

| Dataset | State records | Questions | Supervision | Public download |
|---|---:|---:|---|---|
| Finite-deck Blackjack v2 | 4,000 | 20,000 | Exact model outcome distributions and optimal-action policy | [Hugging Face](https://huggingface.co/datasets/XinranSong/nanojev-blackjack-finite-v2) |
| Coinbase BTC direction v2 | 15,089 | 45,267 | Observed 1/5/15-minute direction labels | [Hugging Face](https://huggingface.co/datasets/XinranSong/nanojev-coinbase-btc-direction-v2) |

Each HF card documents source, rules, license, limitations and loading. Code and one full native example per module live here; complete datasets and raw market data stay on HF. Snapshot revisions and checksum-manifest hashes are pinned in `datasets.json`.

## Download and validate

Run from the NanoJev repository root. Python 3.10+; downloading, generation and validation use the standard library and need no HF login.

```bash
python data_modules/download_datasets.py
python scripts/train_pipeline_decisions.py --input data_modules/blackjack/data --validate-only
python scripts/train_pipeline_decisions.py --input data_modules/trading/data --validate-only
```

The downloader verifies the pinned manifest and each file; an existing mismatched file is not overwritten. `--module blackjack` or `--module trading` selects one module. `--include-raw` also downloads the unchanged BTC ZIP needed for Trading reproduction and integration tests. Generated/downloaded data directories are ignored by Git.

Native JSONL fields include `state`, mapping-valued `questions`, `gold`, `gold_probs`, per-question target kinds and `metadata`. HF's Parquet viewer stores mapping columns as lossless JSON strings; use the native `{module}/data/*.jsonl` files for the trainer, not the viewer columns. Viewer and native files are the same observations and must not be counted twice.

Train each separately in an environment with the existing model dependencies and suitable hardware:

```bash
python scripts/train_pipeline_decisions.py --input data_modules/blackjack/data \
  --objective gold_distribution --loss ce --max-length 512 --output-dir runs/blackjack_v2
python scripts/train_pipeline_decisions.py --input data_modules/trading/data \
  --objective observed_outcome --loss ce --max-length 512 --output-dir runs/trading_v2
```

These are example commands, not trained results. Mixed-task benefit is unestablished; evaluate standalone behavior and tune sampling/weights on `dev` before considering inclusion in a shared mixture.

## Blackjack: rules and probability semantics

See [one complete state](blackjack/example.json). The generator draws 100 parent compositions from a 52-card deck, with 32–44 cards before the visible deal, and creates 40 states per composition. Parent shoes require at least two aces, one six and two ten-value cards. Each source contributes 20 soft and 20 hard hands, with 2–3 player cards and totals 12–21; one fixed [6,10] versus dealer 10 anchor makes composition effects inspectable. This is designed coverage, not real-game visitation frequency.

- All remaining rank counts are supplied in the state; draws are without replacement. Visible player cards and the dealer upcard are already removed.
- No predealt dealer hole card, no peek; the dealer draws after the player stops. Player bust loses immediately.
- Hit/stand only, +1/0/-1 win/push/loss rewards; no natural-21 bonus, doubling, splitting, surrender or insurance.
- After the first action, the player continues with the expected-reward-optimal policy, updating counts and standing on numerical ties.
- Dynamic programming exhaustively enumerates the stated model in float64; labels are not Monte Carlo estimates.

Five questions share each state: `best_action`, `hit_outcome`, `stand_outcome`, `hit_win`, `stand_win`. Event targets are `programmatic_conditional_distribution`; no actual event was drawn, so hard event labels are `unobserved`. `best_action` is an `optimal_action_policy`, uniformly split over numerical ties. **A policy value of 1 is not a win probability; no softmax(Q) is presented as a true event distribution.** `metadata.action_values` stores P(win)-P(loss), and `metadata.regret` stores max(Q)-Q(action).

`source_group_id` identifies the parent shoe, keeping related states together. Train/dev/calibration/test/ood have 2,400/400/400/400/400 states from 60/10/10/10/10 source groups. Semantic states are deduplicated globally, even ignoring the rule. OOD changes both composition and dealer rule (S17 → H17), not a matched rule-only comparison. Test has only ten source groups. Complete remaining-card information and simplified rules make this a controlled probability task, not ordinary casino play.

## Trading: real observations and chronological targets

See [one complete state](trading/example.json), [configuration](trading/config.json), and [source/license notice](THIRD_PARTY_NOTICES.md). The source is Martin Søgaard Nielsen's CC0 Coinbase BTC minute order-book dataset, version 1, April 7–19, 2021: 17,113 raw rows. No synthetic market prices are added.

State contains completed historical midpoint/spread, top-five book-notional imbalance and past-return summaries over 15 rows, with exact elapsed ages. Availability is explicitly assumed to be source timestamp +60 seconds because the provider does not fully specify aggregation endpoints. This conservative adapter assumption does not independently certify original real-time availability.

For h = 60, 300 or 900 seconds, take the first snapshot available at least h seconds after decision time, with at most 15 extra seconds. Label the realized midpoint return `down` below −2 bps, `up` above +2 bps, and `flat` within the inclusive ±2 bps band. `gold_probs` remains empty: one realized outcome does not supply the true conditional distribution. These are forecast targets, not hindsight buy/sell or profitable-execution labels.

Calendar intervals are left-closed/right-open, UTC in April 2021:

| Split | Interval | State records |
|---|---|---:|
| train | Apr 07–14 | 8,260 |
| dev | Apr 14–15 | 1,258 |
| calibration | Apr 15–16 | 1,258 |
| test | Apr 16–18 | 2,518 |
| ood | Apr 18–20 (observations end Apr 19) | 1,795 |

The full window from earliest history to latest target must remain inside one split with a 60-second boundary embargo; a gap over 90 seconds removes the entire state. All horizons stay together, grouped by UTC day. No future filling, interpolation, centered windows or all-data fitted scaler is used. Adjacent windows and horizons remain correlated. The final `ood` is a temporal holdout, not proof of a different market regime. The 13 UTC dates and one asset do not establish broad market generalization or profitability.

**Encode only `state` and `questions`.** Metadata contains future prices/line numbers for replay, and Blackjack action values/regret; it must not enter model input. Do not feed Trading into evaluation that requires known simulator conditional probabilities. Keep calibration separate from development/model selection and reserve test/ood for final reporting.

## Reproduce and test

```bash
python data_modules/download_datasets.py --include-raw
python data_modules/blackjack/generate_blackjack.py --output /tmp/blackjack-reproduced
python data_modules/trading/build_trading.py --output /tmp/trading-reproduced
python -m unittest discover -s data_modules/tests -v
```

Blackjack regeneration takes several minutes. The default seed/config reproduce the published native split files. Tests include a hand-solvable finite-deck case, independent simulation, future perturbation preserving Trading inputs, target replay, gap/boundary purging, and source-group rejection by this checkout's actual trainer. Trading integration tests explicitly skip until the raw ZIP is downloaded. The remaining tests run offline with just the checked-in examples/code.

Code is MIT; Blackjack data/example is MIT; real-market observations and derived Trading data/example are CC0-1.0. No model weights, vendored trainer copy or full datasets are added to Git.
