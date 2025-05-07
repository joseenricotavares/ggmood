import mlflow
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model
import tempfile
import json
import os
import joblib
from datetime import datetime
import uuid
import re


class PyCaretEmbeddingClassificationTrainer:
    def __init__(
        self,
        train_dataset: pd.DataFrame,
        target_column: str,
        drop_columns: list = None,
        seed: int = 42,
        n_select: int = 5,
        sort_metric: str = 'AUC'
    ):
        """
        Trainer for classification models using PyCaret.

        Parameters:
        - train_dataset: Pandas DataFrame with training data.
        - target_column: Target column name.
        - drop_columns: List of columns to drop (optional).
        - seed: Random seed for reproducibility.
        - n_select: Number of top models to train.
        - sort_metric: Metric used to sort models.
        """
        self.train_dataset = train_dataset.drop(columns=drop_columns, errors='ignore') if drop_columns else train_dataset.copy()
        self.target_column = target_column
        self.seed = seed
        self.n_select = n_select
        self.sort_metric = sort_metric
        self.drop_columns = drop_columns

        self.tested_models = []
        self.model_results = None

    def train(self):
        """Trains models using PyCaret and stores them internally."""

        setup(
            data=self.train_dataset,
            target=self.target_column,
            fix_imbalance=False,
            remove_outliers=False,
            normalize=False,
            remove_multicollinearity=False,
            session_id=self.seed,
            verbose=False
        )

        self.tested_models = compare_models(
            n_select=self.n_select,
            sort=self.sort_metric,
            errors='raise',
            turbo=True,
            verbose=False
        )

        if not isinstance(self.tested_models, list):
            self.tested_models = [self.tested_models]

        # Generate random model IDs
        self.model_ids = [str(uuid.uuid4()) for _ in self.tested_models]

        self.model_results = pull()

        return self.tested_models

    def log_to_mlflow(
        self,
        add_tags: dict = None,
        experiment_name: str = "pycaret-embeddings-classification",
        use_default_tags: bool = True,
        include_dataset_info: bool = True,
        include_model_params: bool = True
    ) -> list:
        """
        Logs trained models to MLflow. Call after `.train()`.

        Parameters:
        - set_tags: Dictionary containing custom tags to log to MLflow (optional).
        - use_default_tags: Boolean flag to indicate whether to use default tags.
        - experiment_name: MLflow experiment name.
        - include_dataset_info: Boolean flag to log dataset info (optional).
        - include_model_params: Boolean flag to log model hyperparameters (optional).

        Returns:
        - List of MLflow run IDs for the logged models.
        """
        if not self.tested_models or self.model_results is None:
            raise RuntimeError("You must call `.train()` before logging.")

        mlflow.set_experiment(experiment_name)
        run_ids = []

        # Define default tags
        default_tags = {
            "train_seed": self.seed,
            "n_select": self.n_select,
            "sort_metric": self.sort_metric,
            "target_column": self.target_column,
            "dropped_columns": self.drop_columns,
        }

        # Include dataset information if required
        if include_dataset_info:
            dataset_info = {
                "num_rows": self.train_dataset.shape[0],
                "num_columns": self.train_dataset.shape[1],
                "columns": list(self.train_dataset.columns)
            }
            default_tags.update(dataset_info)

        for idx, model in enumerate(self.tested_models):
            model_name = model.__class__.__name__
            model_id = self.model_ids[idx]

            # Merge custom tags with default ones if applicable
            if use_default_tags and add_tags is None:
                base_tags = default_tags
            else:
                base_tags = add_tags if add_tags is not None else {}

            # Optionally, include model hyperparameters
            if include_model_params:
                model_params = model.get_params()
                base_tags.update(model_params)

            # Prepend model_name and model_id
            tags_to_log = {
                "model_name": model_name,
                "model_id": model_id,
                **base_tags
            }


            with mlflow.start_run(run_name=f"{experiment_name}-{model_id}") as run:
                run_id = run.info.run_id
                run_ids.append(run_id)

                mlflow.set_tags(tags_to_log)

                mlflow.log_params(model.get_params())

                model_metrics = self.model_results.iloc[idx].to_dict()
                sanitized_metrics = {
                re.sub(r'[^A-Za-z0-9_\-\.\ /\\]', '_', k): float(v)
                for k, v in model_metrics.items() if isinstance(v, (int, float))
                }
            
                mlflow.log_metrics(sanitized_metrics)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=self.train_dataset.drop(columns=[self.target_column]).iloc[:1]
                )

        return run_ids
    
    def save_locally(
        self,
        save_dir: str,
        add_tags: dict = None,
        use_default_tags: bool = True,
        include_dataset_info: bool = True,
        include_model_params: bool = True
    ):
        """
        Saves models and metadata locally.

        Parameters:
        - save_dir: Path to the directory to save models and metadata.
        - add_tags: Optional dict of tags to include.
        - use_default_tags: Whether to include default tags.
        - include_dataset_info: Whether to include dataset metadata.
        - include_model_params: Whether to include model hyperparameters.
        """
        if not self.tested_models or self.model_results is None:
            raise RuntimeError("You must call `.train()` before saving.")

        os.makedirs(save_dir, exist_ok=True)
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "models": []
        }

        default_tags = {
            "train_seed": self.seed,
            "n_select": self.n_select,
            "sort_metric": self.sort_metric,
            "target_column": self.target_column,
            "dropped_columns": self.drop_columns,
        }

        if include_dataset_info:
            dataset_info = {
                "num_rows": self.train_dataset.shape[0],
                "num_columns": self.train_dataset.shape[1],
                "columns": list(self.train_dataset.columns)
            }
            default_tags.update(dataset_info)

        for idx, model in enumerate(self.tested_models):
            model_name = model.__class__.__name__
            model_id = self.model_ids[idx]
            model_path = os.path.join(save_dir, f"{model_id}.pkl")

            joblib.dump(model, model_path)

            model_meta = {
                "model_id": model_id,
                "model_name": model_name,
                "model_path": model_path
            }

            tags = {}
            if use_default_tags:
                tags.update(default_tags)
            if add_tags:
                tags.update(add_tags)

            if include_model_params:
                model_params = model.get_params()
                tags.update(model_params)

            model_meta["tags"] = tags
            model_meta["metrics"] = {
                k: float(v) for k, v in self.model_results.iloc[idx].to_dict().items() if isinstance(v, (int, float))
            }

            metadata["models"].append(model_meta)

        # Save metadata JSON
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
