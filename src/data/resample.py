from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

def reduce_text_df_per_class(
    df,
    class_column: str,
    embedding_input,  # Can be a list of vectors or the name of the column with text
    target_clusters: int = None,
    target_proportion: float = None,
    embedding_model_path: str = None,
    device: str = "cpu",
    batch_size: int = 256,
    seed: int = 42,
    verbose: bool = True,
    show_progress_bar: bool = True
) -> pd.DataFrame:
    
    """
    Reduces a labeled text dataset by clustering SBERT-like embeddings using MiniBatchKMeans, separately for each class.

    This function is intended for textual datasets where each sample has:
    - an associated embedding vector (e.g., SBERT)
    - a class label (used for stratified reduction)

    It performs dimensionality reduction by identifying representative samples (closest to cluster centroids)
    within each class using MiniBatchKMeans clustering.

    Parameters:
    - df: Pandas DataFrame containing the original text data and class labels.
    - class_column: Name of the column containing class labels.
    - embedding_input: Either:
        - a NumPy array or list of precomputed embeddings (must align with `df`), OR
        - the name of a column in `df` containing text to be embedded.
    - target_clusters: Integer, number of clusters per class (overrides `target_proportion`).
    - target_proportion: Float, proportion of samples to retain per class (used if `target_clusters` is not given).
    - embedding_model_path: Path to a SentenceTransformer model (used if embeddings are to be computed).
    - device: "cpu" or "cuda".
    - batch_size: Batch size for computing embeddings (default: 256).
    - seed: Random seed for reproducibility.
    - verbose: Whether to log detailed information (default: True).
    - show_progress_bar: Whether to show progress bar during embedding computation (default: True).
    """

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Avoid adding multiple handlers if the function is called more than once.
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if target_clusters is None and target_proportion is None:
        raise ValueError("You must provide either target_clusters or target_proportion.")
    
    # Compute or use given embeddings
    if isinstance(embedding_input, str):  # embedding_input is a column name
        if embedding_model_path is None:
            raise ValueError("Embedding model path must be provided when embedding_input is a text column name.")
        logger.info("Generating embeddings...")
        model = SentenceTransformer(embedding_model_path, device=device)
        embeddings = model.encode(
            df[embedding_input].tolist(),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True
        )
    else:
        embeddings = np.array(embedding_input)

    reduced_dfs = []
    for cls in df[class_column].unique():
        logger.info(f"Processing class: {cls}")
        class_mask = df[class_column] == cls
        df_class = df[class_mask].reset_index(drop=True)
        emb_class = embeddings[class_mask]

        if target_clusters is not None:
            n_clusters = min(len(emb_class), int(target_clusters))
        else:
            n_clusters = max(1, int(len(emb_class) * target_proportion))

        logger.info(f"Class size: {len(emb_class)} → Clusters: {n_clusters}")
        logger.info("Fitting MiniBatchKMeans...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=max(1024, n_clusters),
            n_init='auto'
        )
        labels = kmeans.fit_predict(emb_class)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, emb_class)

        reduced_df = df_class.iloc[closest].copy()
        reduced_dfs.append(reduced_df)

    # Combine and shuffle
    df_reduced = pd.concat(reduced_dfs, ignore_index=True).sample(frac=1, random_state=seed)

    logger.info(f"Final reduced dataset size: {df_reduced.shape}")
    if verbose:
        logger.info("\n" + str(df_reduced[class_column].value_counts()))

    return df_reduced