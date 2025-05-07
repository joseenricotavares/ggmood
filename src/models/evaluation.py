import time
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import mlflow
import torch
from torch.utils.data import Dataset, DataLoader

class BinaryClassificationEvaluator:
    """
    A utility class for evaluating binary classification models (sklearn or transformer-based)
    and optionally logging results to MLflow.
    """

    VALID_METRICS = {'AUC', 'Accuracy', 'Recall', 'Precision', 'F1', 'PredictionTime'}

    def __init__(
        self,
        model,
        test_dataset_name: str,
        test_dataset_prefix: str = '',
        tokenizer=None,
        device=None
    ):
        """
        Initialize the BinaryClassificationEvaluator.

        Parameters:
        - model: Trained classification model (e.g., sklearn or HuggingFace).
        - test_dataset_name: Optional name to tag the test dataset in MLflow.
        - test_dataset_prefix: Prefix to use for each logged metric name.
        - tokenizer: Tokenizer for text-based models (required for HuggingFace models).
        - device: Torch device (e.g., 'cuda' or 'cpu') for model inference.
        """
        self.model = model
        self.test_dataset_name = test_dataset_name
        self.test_dataset_prefix = test_dataset_prefix
        self.tokenizer = tokenizer
        self.device = device

    def _compute_metrics(
        self,
        y_true,
        y_pred,
        y_scores,
        selected_metrics,
        prediction_time=None
    ) -> dict:
        """
        Compute and return evaluation metrics based on predictions.

        Parameters:
        - y_true: Ground-truth labels.
        - y_pred: Predicted class labels.
        - y_scores: Predicted probability scores.
        - selected_metrics: List of metrics to compute.
        - prediction_time: Optional time taken to make predictions.

        Returns:
        - Dictionary of computed metrics.
        """
        metrics = {}
        # Use test_dataset_name as prefix if test_dataset_prefix is empty
        prefix = self.test_dataset_prefix or f'{self.test_dataset_name}_'
        
        if 'AUC' in selected_metrics:
            auc_value = roc_auc_score(y_true, y_scores) if len(np.unique(y_scores)) > 1 else 0.0
            metrics[f'{prefix}TestAUC'] = auc_value

        if 'Accuracy' in selected_metrics:
            metrics[f'{prefix}TestAccuracy'] = accuracy_score(y_true, y_pred)

        if 'Recall' in selected_metrics:
            metrics[f'{prefix}TestRecall'] = recall_score(y_true, y_pred)

        if 'Precision' in selected_metrics:
            metrics[f'{prefix}TestPrecision'] = precision_score(y_true, y_pred)

        if 'F1' in selected_metrics:
            metrics[f'{prefix}TestF1'] = f1_score(y_true, y_pred)

        if 'PredictionTime' in selected_metrics and prediction_time is not None:
            metrics[f'{prefix}PredictionTime'] = prediction_time

        return metrics

    def evaluate_sklearn_model(
        self,
        df_test,
        target_column: str,
        drop_columns: list = None,
        selected_metrics=None
    ) -> dict:
        """
        Evaluate a scikit-learn binary classification model on tabular data.

        Parameters:
        - df_test: Test dataset as a DataFrame.
        - target_column: Column name of the target variable.
        - drop_columns: List of columns to drop from the test dataset (optional).
        - selected_metrics: List of metrics to evaluate (default: all VALID_METRICS).

        Returns:
        - Dictionary of computed metrics.
        """
        if selected_metrics is None:
            selected_metrics = list(self.VALID_METRICS)

        invalid = [m for m in selected_metrics if m not in self.VALID_METRICS]
        if invalid:
            raise ValueError(f"Invalid metrics requested: {invalid}. Supported: {sorted(self.VALID_METRICS)}")

        # Split features and target
        X_test = df_test.drop(columns=target_column).copy()
        if drop_columns:
            X_test = X_test.drop(columns=drop_columns, errors='ignore')
        y_true = df_test[target_column].values.astype(int)

        if len(set(y_true)) > 2:
            raise ValueError("Multi-class classification is not supported. Only binary classification is allowed.")

        start_time = time.time()

        # Predict labels
        try:
            y_pred = self.model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")

        # Predict scores or fallback to predicted labels
        try:
            y_scores = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred
        except Exception as e:
            raise RuntimeError(f"Score prediction error: {e}")

        prediction_time = time.time() - start_time

        return self._compute_metrics(y_true, y_pred, y_scores, selected_metrics, prediction_time)

    def evaluate_sentiment_model(
        self,
        df_test,
        target_column: str,
        text_column: str,
        selected_metrics=None,
        batch_size=16
    ) -> dict:
        """
        Evaluate a transformer-based sentiment model on a text dataset.

        Parameters:
        - df_test: Test DataFrame with a 'text' column.
        - target_column: Name of the target column.
        - text_column: Name of the text column.
        - selected_metrics: List of metrics to evaluate (default: all VALID_METRICS).
        - batch_size: Batch size for evaluation (default: 16).

        Returns:
        - Dictionary of computed metrics.
        """
        if selected_metrics is None:
            selected_metrics = list(self.VALID_METRICS)

        invalid = [m for m in selected_metrics if m not in self.VALID_METRICS]
        if invalid:
            raise ValueError(f"Invalid metrics requested: {invalid}. Supported: {sorted(self.VALID_METRICS)}")

        if self.tokenizer is None or self.device is None:
            raise ValueError("Tokenizer and device must be provided for sentiment model evaluation.")

        # Define simple dataset wrapper for inference
        class SimpleTextDataset(Dataset):
            def __init__(self, texts):
                self.texts = texts

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx]

        y_true = df_test[target_column].values.astype(int)
        dataset = SimpleTextDataset(list(df_test[text_column]))
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_scores = []
        all_preds = []

        self.model.eval()
        start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_scores.extend(probs)
                all_preds.extend(preds)

        prediction_time = time.time() - start_time

        all_scores = np.array(all_scores)
        all_preds = np.array(all_preds)

        return self._compute_metrics(y_true, all_preds, all_scores, selected_metrics, prediction_time)

    def log_to_mlflow(self, metrics: dict, experiment_name: str = "pycaret-embeddings-classification"):
        """
        Ensures there is an active MLflow run.
        If so, logs the new test dataset name (appending it if the tag is already present) and a dictionary of metrics to the active run.
        
        If not (normally in the case of non-fine-tuned sentiment models),
        starts a new MLflow run and logs:

        - Model name
        - Tokenizer name (if available)
        - Device (if available)
        - Test dataset name (if provided)
        - Metrics
        """

        if mlflow.active_run():
            # Get the current list of test datasets, if any
            current_tags = mlflow.active_run().data.tags
            existing_datasets = current_tags.get("test_dataset", "")
            # Split into list, check if current dataset is already listed
            dataset_list = [ds.strip() for ds in existing_datasets.split(",") if ds.strip()]
            if self.test_dataset_name and self.test_dataset_name not in dataset_list:
                dataset_list.append(self.test_dataset_name)
                updated_value = ", ".join(dataset_list)
                mlflow.set_tag("test_dataset", updated_value)

            mlflow.log_metrics(metrics)

        else:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()

            # General metadata
            model_name = getattr(self.model, '__class__', type(self.model)).__name__
            mlflow.set_tag("model_name", model_name)

            if self.tokenizer is not None:
                tokenizer_name = getattr(self.tokenizer, 'name_or_path', type(self.tokenizer).__name__)
                mlflow.set_tag("tokenizer_name", tokenizer_name)

            if self.device is not None:
                mlflow.set_tag("device", str(self.device))

            if self.test_dataset_name:
                mlflow.set_tag("test_dataset", self.test_dataset_name)

            # Log metrics
            mlflow.log_metrics(metrics)
