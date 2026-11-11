"""Seeded Optuna search restricted to the three allowed tuning folds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna

from supply_chain_opt_engine.evaluation import Fold
from supply_chain_opt_engine.learning import (
    DirectDataset,
    ForecastConfig,
    fit_quantiles,
    predict_raw,
)
from supply_chain_opt_engine.metrics import pinball_loss

LOGGER = logging.getLogger(__name__)


@dataclass
class FoldData:
    fold: Fold
    train: DirectDataset
    validation: DirectDataset


def tune_parameters(data: list[FoldData], config: ForecastConfig, study_path: Path) -> optuna.Study:
    if [item.fold.fold_id for item in data] != [1, 2, 3] or any(
        item.fold.role != "tuning" for item in data
    ):
        raise ValueError("Optuna may only receive tuning folds 1, 2, and 3; never the holdout.")
    study_path.parent.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_path.stem,
        storage=f"sqlite:///{study_path.resolve()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.set_user_attr("fold_ids", [1, 2, 3])
    study.set_user_attr("origins", [item.fold.origin.isoformat() for item in data])
    study.set_user_attr(
        "objective", "Mean P50 pinball across tuning folds; shared parameters for all quantiles"
    )
    study.set_user_attr("seed", config.seed)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 40, 200, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        losses = []
        begin = perf_counter()
        for item in data:
            model = fit_quantiles(item.train, params, config, quantiles=(0.5,))
            raw = predict_raw(model, item.validation.X)["q50"].to_numpy()
            loss = pinball_loss(item.validation.y, np.maximum(raw, 0), 0.5)
            losses.append(loss)
            trial.set_user_attr(f"fold_{item.fold.fold_id}_pinball_q50", loss)
            trial.report(float(np.mean(losses)), item.fold.fold_id)
            if trial.should_prune():
                LOGGER.info("Trial %d pruned after fold %d", trial.number, item.fold.fold_id)
                raise optuna.TrialPruned()
        trial.set_user_attr("fit_predict_seconds", perf_counter() - begin)
        LOGGER.info(
            "Trial %d/%d: P50 pinball %.5f in %.1fs",
            trial.number + 1,
            config.optuna_trials,
            np.mean(losses),
            perf_counter() - begin,
        )
        return float(np.mean(losses))

    study.optimize(objective, n_trials=config.optuna_trials, n_jobs=1)
    return study
