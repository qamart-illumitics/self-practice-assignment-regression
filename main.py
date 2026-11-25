# Standard library imports
import logging
import warnings

# Third-party imports
import pandas as pd
import yaml

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
warnings.filterwarnings("ignore")


def main():
    # Configuration file path
    config_path = "./src/config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load CSV file into a DataFrame
    olevel_df = pd.read_csv(config["file_path"])

    # Initialize and run data preparation
    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(olevel_df)

    # Initialize model training with the created preprocessor
    model_training = ModelTraining(config, data_prep.preprocessor)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(cleaned_df)

    # Impute missing attendance_rate using training-set medians only (avoids leakage)
    X_train, X_val, X_test = data_prep.impute_attendance(X_train, X_val, X_test)

    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = model_training.train_and_evaluate_baseline_models(
        X_train, y_train, X_val, y_val
    )

    # Train and evaluate tuned models with hyperparameter tuning
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    # Combine all models and their metrics into dictionaries
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    # Summary table of validation results
    summary = pd.DataFrame(all_metrics).T.round(3).sort_values("R2", ascending=False)
    logging.info("Validation results:\n%s", summary.to_string())

    # Find the best model based on validation R2 score
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["R2"])
    best_model = all_models[best_model_name]
    logging.info(f"Best Model Found: {best_model_name}")

    # Evaluate the best model on the test set (used only once)
    model_training.evaluate_final_model(best_model, X_test, y_test, best_model_name)


if __name__ == "__main__":
    main()