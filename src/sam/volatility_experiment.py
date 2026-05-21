"""Orchestrate the volatility-regime scoring experiment (P19)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sam.io import ensure_dir, write_table
from sam.progress import progress_iter, progress_task, progress_write
from sam.research import (
    ExperimentRegistry,
    validate_research_frame,
    write_experiment_bundle,
)
from sam.risk_metrics import (
    build_operating_summary,
    feature_drift_psi_summary,
    month_block_bootstrap_ap,
    regime_metrics,
    score_decile_summary,
    threshold_sensitivity_table,
)
from sam.risk_policy import run_risk_policy_diagnostic
from sam.volatility_config import VolatilityRegimeRunConfig, load_volatility_run_config
from sam.volatility_data import ingest_volatility_research_data, synthetic_volatility_fixtures
from sam.volatility_features import (
    assign_split_masks,
    build_volatility_feature_frame,
    leakage_audit,
)
from sam.volatility_models import (
    ModelRunResult,
    results_to_holdout_summary,
    run_domain_rules,
    run_tabicl_direct,
    run_tabpfn_direct,
    run_xgboost_incumbent,
)
from sam.volatility_report import build_volatility_market_snapshot, write_volatility_regime_report

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolatilityExperimentResult:
    out_dir: Path
    frame: pd.DataFrame
    holdout_summary: pd.DataFrame
    bundle_dir: Path | None


def run_volatility_regime_experiment(
    *,
    experiment_id: str = "volatility-regime-scoring",
    run_config_path: str | Path = "configs/risk/volatility_regime_run.toml",
    universe_path: str | Path = "configs/universes/volatility_regime.toml",
    out_dir: str | Path = "reports/experiments/volatility-regime-scoring/run",
    cache_dir: str | Path | None = "data/research/volatility_regime",
    registry_path: str | Path = "configs/experiments",
    use_synthetic_data: bool = False,
    vix_csv: str | Path | None = None,
    fred_csv: str | Path | None = None,
    skip_tfm: bool = False,
    skip_xgboost: bool = False,
    fast_mode: bool = False,
    refresh_data: bool = False,
    write_bundle: bool = True,
    write_figures: bool = True,
    write_report: bool = True,
    show_progress: bool = True,
) -> VolatilityExperimentResult:
    """Run the full P19 volatility-regime workflow and write research artifacts."""

    run_cfg = load_volatility_run_config(run_config_path)
    if fast_mode:
        run_cfg = run_cfg.model_copy(update={"fast_mode": True})
    spec = ExperimentRegistry(registry_path).get(experiment_id)
    target = ensure_dir(out_dir)

    with progress_task("Load market data", enabled=show_progress):
        if use_synthetic_data:
            data = synthetic_volatility_fixtures(n_days=4000, start="2006-01-01")
        else:
            data = ingest_volatility_research_data(
                universe_path=universe_path,
                run_config_path=run_config_path,
                cache_dir=cache_dir,
                vix_csv=vix_csv,
                fred_csv=fred_csv,
                refresh=refresh_data,
                show_progress=show_progress,
            )

    with progress_task("Build features", enabled=show_progress):
        feature_result = build_volatility_feature_frame(
            data["prices"],
            data["panel"],
            config=run_cfg,
        )
    frame = assign_split_masks(feature_result.frame, config=run_cfg)
    validate_research_frame(
        frame,
        target_date_column="target_date",
        feature_available_column="feature_available_date",
    )

    holdout_mask = frame["split_holdout"]
    train_mask = frame["split_train_all_history"]
    calibration_mask = frame["split_calibration"]
    holdout = frame.loc[holdout_mask]
    labeled_holdout = holdout["target_high_vol_20d"].dropna()
    holdout_rate = (
        float(labeled_holdout.mean()) if not labeled_holdout.empty else float("nan")
    )

    with progress_task("Domain rule baselines", enabled=show_progress):
        results: list[ModelRunResult] = run_domain_rules(frame, holdout_mask=holdout_mask)
    score_columns: dict[str, str] = {}

    if run_cfg.run_xgboost and not skip_xgboost:
        results.append(
            run_xgboost_incumbent(
                frame,
                feature_columns=feature_result.model_features,
                train_mask=train_mask,
                holdout_mask=holdout_mask,
                config=run_cfg,
                show_progress=show_progress,
            )
        )
    if run_cfg.run_tabpfn and not skip_tfm:
        with progress_task("TabPFN direct fit", enabled=show_progress):
            results.append(
                run_tabpfn_direct(
                    frame,
                    feature_columns=feature_result.model_features,
                    train_mask=train_mask,
                    holdout_mask=holdout_mask,
                )
            )
    if run_cfg.run_tabicl and not skip_tfm:
        with progress_task("TabICL direct fit", enabled=show_progress):
            results.append(
                run_tabicl_direct(
                    frame,
                    feature_columns=feature_result.model_features,
                    train_mask=train_mask,
                    holdout_mask=holdout_mask,
                )
            )

    holdout_scores = frame.loc[holdout_mask].copy()
    for result in results:
        if result.error or result.holdout_scores.empty:
            logger.warning("model %s skipped: %s", result.name, result.error)
            continue
        col = _score_column_name(result.name)
        holdout_scores.loc[result.holdout_scores.index, col] = result.holdout_scores.astype(float)
        score_columns[result.name] = col

    summary = results_to_holdout_summary(
        results,
        holdout_rows=int(len(holdout)),
        holdout_rate=holdout_rate,
    )
    write_table(summary, target / "publication_holdout_summary.csv")

    operating = build_operating_summary(
        holdout_scores,
        score_columns={
            name: col for name, col in score_columns.items() if col in holdout_scores.columns
        },
    )
    write_table(operating, target / "publication_operating_summary.csv")
    write_table(operating, target / "operating_summary.csv")

    leakage = leakage_audit(
        frame,
        config=run_cfg,
        model_feature_count=len(feature_result.model_features),
    )
    write_table(leakage, target / "leakage_checks.csv")

    drift_features = feature_result.model_features or feature_result.base_features
    drift = feature_drift_psi_summary(
        frame,
        drift_features,
        reference_mask=calibration_mask,
        comparison_mask=holdout_mask,
        show_progress=show_progress,
    )
    if drift.empty:
        drift = pd.DataFrame(
            columns=["feature", "psi", "reference_rows", "comparison_rows"]
        )
    write_table(drift, target / "feature_drift_summary.csv")

    split_rows = _split_summary(frame, run_cfg)
    write_table(split_rows, target / "split_summary.csv")

    bootstrap_rows = []
    iterations = (
        run_cfg.fast_bootstrap_iterations if run_cfg.fast_mode else run_cfg.bootstrap_iterations
    )
    for model_name, col in progress_iter(
        score_columns.items(),
        desc="Bootstrap AP by model",
        total=len(score_columns),
        enabled=show_progress,
        unit="model",
    ):
        boot = month_block_bootstrap_ap(
            holdout_scores,
            score_column=col,
            n_iterations=iterations,
            show_progress=show_progress,
            progress_desc=f"Bootstrap: {model_name[:32]}",
        )
        bootstrap_rows.append({"model": model_name, **boot})
    bootstrap = pd.DataFrame(bootstrap_rows)
    write_table(bootstrap, target / "publication_month_block_bootstrap_uncertainty.csv")

    sensitivity_rows = []
    for model_name, col in score_columns.items():
        if col not in holdout_scores.columns:
            continue
        table = threshold_sensitivity_table(
            holdout_scores,
            score_column=col,
            quantiles=run_cfg.threshold_sensitivity_quantiles,
            holdout_mask=pd.Series(True, index=holdout_scores.index),
        )
        table["model"] = model_name
        sensitivity_rows.append(table)
    if sensitivity_rows:
        write_table(
            pd.concat(sensitivity_rows, ignore_index=True),
            target / "publication_threshold_sensitivity.csv",
        )

    if run_cfg.holdout_regimes:
        regime = regime_metrics(
            holdout_scores,
            score_columns=score_columns,
            regimes=run_cfg.holdout_regimes,
        )
        write_table(regime, target / "regime_holdout_summary.csv")

    decile_rows = []
    for model_name, col in score_columns.items():
        deciles = score_decile_summary(
            holdout_scores["target_high_vol_20d"],
            holdout_scores[col],
        )
        decile_rows.append({"model": model_name, **deciles})
    write_table(pd.DataFrame(decile_rows), target / "publication_score_decile_summary.csv")

    policy_summaries = []
    policy_daily = []
    from sam.volatility_models import DOMAIN_RULES

    for model_name, raw_col in progress_iter(
        DOMAIN_RULES.items(),
        desc="Risk policy diagnostics",
        total=len(DOMAIN_RULES),
        enabled=show_progress,
        unit="rule",
    ):
        if raw_col not in frame.columns:
            continue
        daily, policy_summary = run_risk_policy_diagnostic(
            frame,
            data["prices"],
            score_column=raw_col,
            calibration_mask=calibration_mask,
            holdout_mask=holdout_mask,
            action_rate=run_cfg.risk_policy_action_rate,
            high_risk_weight=run_cfg.risk_policy_high_risk_weight,
            cost_bps=run_cfg.risk_policy_cost_bps,
        )
        policy_summary["model"] = model_name
        policy_summaries.append(policy_summary)
        daily["model"] = model_name
        policy_daily.append(daily)
    if policy_summaries:
        write_table(
            pd.concat(policy_summaries, ignore_index=True),
            target / "risk_policy_selected_summary.csv",
        )
    policy_daily_frame = None
    if policy_daily:
        policy_daily_frame = pd.concat(policy_daily, ignore_index=True)
        write_table(policy_daily_frame, target / "risk_policy_daily_returns.csv")
        equity_curves = policy_daily_frame[["date", "model", "equity"]].copy()
        write_table(equity_curves, target / "risk_policy_equity_curves.csv")

    timing_rows = []
    cv_frames = []
    for result in results:
        for phase, seconds in result.timing.items():
            timing_rows.append(
                {
                    "model": result.name,
                    "phase": phase,
                    "seconds": seconds,
                }
            )
        if result.cv_summary is not None and not result.cv_summary.empty:
            summary = result.cv_summary.copy()
            summary.insert(0, "model", result.name)
            cv_frames.append(summary)
    if timing_rows:
        write_table(pd.DataFrame(timing_rows), target / "timing_notes.csv")
    if cv_frames:
        write_table(pd.concat(cv_frames, ignore_index=True), target / "cv_summary.csv")

    metrics_for_bundle = _metrics_table_for_bundle(results)
    write_table(metrics_for_bundle, target / "metrics.csv")
    write_table(feature_result.frame.head(20), target / "model_frame_head.csv")

    with progress_task("Publication figures", enabled=show_progress and write_figures):
        if write_figures:
            _write_publication_figures(
                holdout_scores,
                score_columns,
                target,
                policy_daily=policy_daily_frame,
            )

    bundle_dir = None
    if write_bundle:
        progress_write("Writing experiment bundle", enabled=show_progress)
        artifact_paths = sorted(str(path) for path in target.glob("*.csv"))
        bundle = write_experiment_bundle(spec, target.parent, artifact_paths=artifact_paths)
        bundle_dir = bundle.out_dir

    if write_report:
        with progress_task("Write consolidated report", enabled=show_progress):
            snapshot = {}
            if not use_synthetic_data and cache_dir is not None:
                try:
                    snapshot = build_volatility_market_snapshot(cache_dir=cache_dir)
                except (ValueError, OSError) as exc:
                    logger.warning("market snapshot skipped: %s", exc)
            write_volatility_regime_report(
                target,
                bundle_dir=bundle_dir,
                snapshot=snapshot,
                open_browser_hint=False,
            )

    return VolatilityExperimentResult(
        out_dir=target,
        frame=frame,
        holdout_summary=summary,
        bundle_dir=bundle_dir,
    )


def _score_column_name(model_name: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in model_name)
    return "score__" + slug.strip("_")


def _split_summary(frame: pd.DataFrame, config: VolatilityRegimeRunConfig) -> pd.DataFrame:
    purposes = {
        "tfm_representation_context": (
            "TFM representation context; excluded from post-context optional benchmark fitting"
        ),
        "post_context_tuning_window": "optional CPU benchmark model selection",
        "calibration_window": "post-hoc calibration and threshold-policy diagnostics",
        "holdout_2020_forward": "final deployment-facing evaluation",
        "raw_all_history_final_train": "strong classical incumbent using all pre-holdout labels",
    }
    boundary_dates = {
        "tfm_representation_context": ("2006-01-03", config.context_end),
        "post_context_tuning_window": ("2010-01-04", config.tuning_end),
        "calibration_window": ("2018-01-02", config.calibration_end),
        "holdout_2020_forward": (config.holdout_start, config.data_end),
        "raw_all_history_final_train": ("2006-01-03", config.calibration_end),
    }
    rows = []
    splits = [
        ("tfm_representation_context", frame["split_context"]),
        ("post_context_tuning_window", frame["split_tuning"]),
        ("calibration_window", frame["split_calibration"]),
        ("holdout_2020_forward", frame["split_holdout"]),
        ("raw_all_history_final_train", frame["split_train_all_history"]),
    ]
    for name, mask in splits:
        subset = frame.loc[mask]
        positives = int(subset["target_high_vol_20d"].sum())
        dates = pd.to_datetime(subset["date"]) if len(subset) else pd.Series(dtype="datetime64[ns]")
        config_start, config_end = boundary_dates[name]
        rows.append(
            {
                "split": name,
                "start_date": dates.min().date() if len(dates) else config_start,
                "end_date": dates.max().date() if len(dates) else config_end,
                "rows": int(len(subset)),
                "high_vol_rows": positives,
                "high_vol_rate": (
                    float(subset["target_high_vol_20d"].mean()) if len(subset) else 0.0
                ),
                "purpose": purposes[name],
            }
        )
    return pd.DataFrame(rows)


def _metrics_table_for_bundle(results: list[ModelRunResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        if result.error:
            continue
        rows.append({"model": result.name, "name": result.name, **result.metrics})
    return pd.DataFrame(rows)


def _write_publication_figures(
    holdout: pd.DataFrame,
    score_columns: dict[str, str],
    out_dir: Path,
    *,
    policy_daily: pd.DataFrame | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import auc, precision_recall_curve
    except ImportError:
        logger.info("matplotlib unavailable; skipping figure generation")
        return

    y = holdout["target_high_vol_20d"].astype(int)
    fig, axis = plt.subplots(figsize=(8, 6))
    for model_name, column in score_columns.items():
        if column not in holdout.columns:
            continue
        precision, recall, _ = precision_recall_curve(y, holdout[column])
        ap = auc(recall, precision)
        axis.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    axis.set_title("Holdout precision-recall curves")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "publication_precision_recall_curves_holdout.png", dpi=120)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for model_name, column in score_columns.items():
        if column not in holdout.columns:
            continue
        prob_true, prob_pred = calibration_curve(
            y, holdout[column].clip(0, 1), n_bins=10, strategy="quantile"
        )
        axis.plot(prob_pred, prob_true, marker="o", label=model_name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_title("Holdout calibration curves")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "publication_calibration_curves_holdout.png", dpi=120)
    plt.close(fig)

    if policy_daily is not None and not policy_daily.empty:
        fig, axis = plt.subplots(figsize=(8, 4))
        for model_name, group in policy_daily.groupby("model"):
            axis.plot(
                pd.to_datetime(group["date"]),
                group["equity"],
                label=str(model_name),
            )
        axis.set_title("Risk policy holdout equity curves")
        axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "risk_policy_equity_curves.png", dpi=120)
        plt.close(fig)


__all__ = ["VolatilityExperimentResult", "run_volatility_regime_experiment"]
