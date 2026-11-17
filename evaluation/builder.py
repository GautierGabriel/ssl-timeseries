import os
import json
import logging
from vivml.SSL.evaluation import evaluation as evaluators_module
from vivml.SSL.evaluation.baseevaluation import BaseEvaluationEngine

logger = logging.getLogger(__name__)


def build_evaluator_from_config(config: dict, device) -> BaseEvaluationEngine:

    eval_cfg = config.get("evaluation", {})
    eval_type = eval_cfg.get("type", "RepresentationEvaluation")

    for module in [evaluators_module]:
        if hasattr(module, eval_type):
            evaluator_cls = getattr(module, eval_type)
            params = {k: v for k, v in eval_cfg.items() if k != "type"}
            return evaluator_cls(device=device, **params)

    raise ValueError(f"Evaluation engine '{eval_type}' not found")


def run_full_evaluation_pipeline(config: dict, model, val_loader, device, run_dir: str) -> dict:

    evaluation_engine = build_evaluator_from_config(config, device)
    logger.info(f"Running evaluation with: {evaluation_engine.__class__.__name__}")

    raw_results = evaluation_engine.run_inference(model, val_loader)
    report = evaluation_engine.compute_metrics(raw_results)

    metrics_path = os.path.join(run_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report["scalars"], f, indent=4)

    logger.info(f"Saved evaluation metrics to {metrics_path}")

    return report