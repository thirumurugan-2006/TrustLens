# TrustLens Experiments

This directory contains experimental configurations and results for model benchmarking and selection.

## Structure

*   `configs/`: Contains JSON/YAML configuration files defining experiment parameters.
*   `results/`: Contains output metrics, evaluation reports, and logs from each experiment run.

## Experiment Logging Requirements

Every experiment must record the following metadata to ensure reproducibility:

*   **model**: The specific model name/path used.
*   **model_version**: The version or tag of the model.
*   **dataset_version**: The specific dataset/split used for evaluation.
*   **language**: Target language(s) for the experiment.
*   **task**: The specific task (e.g., semantic representation, retrieval, reranking, graph classification).
*   **hyperparameters**: Any relevant tuning parameters (learning rate, batch size, Top-K, etc.).
*   **hardware**: Device used (CPU/CUDA) and specs if applicable.
*   **metrics**: Benchmark scores (Recall@K, MRR, Accuracy, F1, etc.).
*   **timestamp**: When the experiment was executed.

Do not permanently select a final model without first logging its baseline vs candidates here.
