"""Model architectures, training pipelines, and prediction utilities."""
from .metadata import (
    find_metrics_file,
    format_benchmark_report,
    format_model_info,
    format_models_table,
    get_benchmark_reference,
    get_model_details,
    list_models,
)

from .predict_model import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    compute_metrics,
    evaluate_model,
    load_trained_model,
    predict_image,
    save_evaluation_report,
)
from .train_model import (
    TBDataset,
    TrainConfig,
    build_model,
    get_device,
    train_model,
    train_one_epoch,
    validate_epoch,
)

__all__ = [
    "ID_TO_LABEL",
    "LABEL_TO_ID",
    "TBDataset",
    "TrainConfig",
    "build_model",
    "compute_metrics",
    "evaluate_model",
    "get_device",
    "load_trained_model",
    "predict_image",
    "save_evaluation_report",
    "train_model",
    "train_one_epoch",
    "validate_epoch",
    "find_metrics_file",
    "format_benchmark_report",
    "format_model_info",
    "format_models_table",
    "get_benchmark_reference",
    "get_model_details",
    "list_models",
]
