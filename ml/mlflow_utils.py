# ml/mlflow_utils.py
"""
MLflow utilities for Phase 5 — experiment tracking and model registry.

Responsibilities:
  - Configure tracking URI and experiment
  - Context-manager for starting named runs
  - Log parameters, metrics, and sklearn-compatible models
  - Query the best run by metric
  - Load models from run URI or the Model Registry
"""

import os
from contextlib import contextmanager
from typing import Any, Optional

import mlflow
import mlflow.sklearn
from loguru import logger

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
DEFAULT_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT",  "trading_strategy_selector")
DEFAULT_MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME",  "strategy_selector")


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_mlflow(experiment_name: str = DEFAULT_EXPERIMENT) -> str:
    """
    Point MLflow at the tracking server and create/get the experiment.
    Returns the experiment ID string.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(
        f"MLflow | uri={MLFLOW_TRACKING_URI} "
        f"| experiment='{experiment_name}' (id={experiment.experiment_id})"
    )
    return experiment.experiment_id


# ── Run context manager ───────────────────────────────────────────────────────

@contextmanager
def start_run(run_name: str, experiment_name: str = DEFAULT_EXPERIMENT):
    """
    Set up MLflow and yield an active run object.

    Usage:
        with start_run("xgboost_training") as run:
            mlflow.log_metric("accuracy", 0.9)
            ...
    """
    setup_mlflow(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow run started | name='{run_name}' | id={run.info.run_id}")
        yield run
    logger.info(f"MLflow run finished | id={run.info.run_id}")


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_params_and_metrics(params: dict, metrics: dict) -> None:
    """Log params and metrics dicts into the currently active run."""
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)


def log_model(
    model:           Any,
    artifact_path:   str = "model",
    registered_name: Optional[str] = None,
) -> str:
    """
    Log a scikit-learn-compatible model to the active run.
    Optionally registers it in the MLflow Model Registry.

    Returns the artifact URI (runs:/<run_id>/<artifact_path>).
    """
    mlflow.sklearn.log_model(
        sk_model              = model,
        artifact_path         = artifact_path,
        registered_model_name = registered_name,
    )
    active = mlflow.active_run()
    uri = f"runs:/{active.info.run_id}/{artifact_path}"
    logger.info(f"Model logged | uri={uri}" + (f" | registry='{registered_name}'" if registered_name else ""))
    return uri


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_best_run_id(
    experiment_name: str = DEFAULT_EXPERIMENT,
    metric:          str = "val_accuracy",
) -> Optional[str]:
    """
    Return the run_id with the highest value for `metric`.
    Returns None if no qualifying runs exist.
    """
    setup_mlflow(experiment_name)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experiment '{experiment_name}' not found in MLflow.")
        return None

    runs = client.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string  = f"metrics.{metric} > 0",
        order_by       = [f"metrics.{metric} DESC"],
        max_results    = 1,
    )

    if not runs:
        logger.warning(f"No runs with metric '{metric}' in experiment '{experiment_name}'.")
        return None

    best = runs[0]
    logger.info(
        f"Best run | id={best.info.run_id} | {metric}={best.data.metrics.get(metric, 0):.4f}"
    )
    return best.info.run_id


def get_run_params(run_id: str) -> dict:
    """Return the params dict for a specific run."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    run    = client.get_run(run_id)
    return run.data.params


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_from_run(run_id: str, artifact_path: str = "model") -> Any:
    """Load a logged sklearn model from a specific run."""
    uri = f"runs:/{run_id}/{artifact_path}"
    logger.info(f"Loading model | uri={uri}")
    return mlflow.sklearn.load_model(uri)


def load_registered_model(
    model_name: str = DEFAULT_MODEL_NAME,
    stage:      str = "Production",
) -> Any:
    """
    Load the latest Production version of a registered model.
    Falls back to the best run when no registered model is found.
    """
    setup_mlflow()
    uri = f"models:/{model_name}/{stage}"
    try:
        model = mlflow.sklearn.load_model(uri)
        logger.info(f"Loaded registered model | uri={uri}")
        return model
    except Exception as exc:
        logger.warning(f"Registry load failed ({exc}). Falling back to best run.")
        run_id = get_best_run_id()
        if run_id is None:
            raise RuntimeError(
                "No trained model found in MLflow. Run StrategyTrainer.train() first."
            ) from exc
        return load_model_from_run(run_id)


# ─── Standalone check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    exp_id = setup_mlflow()
    print(f"Experiment ID: {exp_id}")
    print(f"Tracking URI:  {MLFLOW_TRACKING_URI}")
    print(f"Model name:    {DEFAULT_MODEL_NAME}")
