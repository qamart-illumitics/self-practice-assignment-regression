# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreparation:
    """
    A class used to clean and preprocess O-level mathematics examination data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical and nominal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, olevel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        olevel_df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        olevel_df = olevel_df.copy()  # don't modify the caller's DataFrame

        # Missing target: drop rows with missing "final_test" (~3% of rows).
        # The target cannot be imputed without inventing labels.
        olevel_df = olevel_df.dropna(subset=[self.config["target_column"]])

        # "index" is just a row number and "bag_color" is non-informative.
        olevel_df = olevel_df.drop(columns=["index", "bag_color"])

        # Duplicates: 846 repeat student records, identical except for bag_color and,
        # in 83 cases, a missing attendance_rate. Keep one record per student,
        # preferring the one with attendance recorded.
        olevel_df = (olevel_df
                     .sort_values("attendance_rate", na_position="last")
                     .drop_duplicates(subset="student_id", keep="first")
                     .sort_index()
                     .reset_index(drop=True))

        # "CCA" has inconsistent casing (Clubs, CLUBS) -> lowercase.
        # Missing CCA is filled as "unknown" to keep it distinct from "none".
        olevel_df["CCA"] = olevel_df["CCA"].str.lower().fillna("unknown")

        # "tuition" mixes (Yes, Y) and (No, N) -> standardise.
        olevel_df["tuition"] = olevel_df["tuition"].replace({"Y": "Yes", "N": "No"})

        # "age" has values 15, 16, 6, 5, -5, -4. Treat negatives and 5/6 as typos
        # (missing leading "1"); drop the single unresolvable age-4 record.
        olevel_df["age"] = olevel_df["age"].abs()
        olevel_df.loc[olevel_df["age"].isin([5, 6]), "age"] += 10
        olevel_df = olevel_df[olevel_df["age"] != 4]

        # Feature engineering: sleep_duration from sleep/wake times (hh:mm strings).
        # Times after midnight are shifted by +24h so they sort after 23:00.
        olevel_df["sleep_hr"] = olevel_df["sleep_time"].apply(self.to_hours)
        olevel_df["wake_hr"] = olevel_df["wake_time"].apply(self.to_hours)
        olevel_df.loc[olevel_df["sleep_hr"] < 12, "sleep_hr"] += 24
        olevel_df["sleep_duration"] = (olevel_df["wake_hr"] + 24) - olevel_df["sleep_hr"]

        # Feature engineering: class_size combining n_male and n_female.
        olevel_df["class_size"] = olevel_df["n_male"] + olevel_df["n_female"]

        # Drop identifiers, raw strings and helper columns now captured elsewhere.
        cols_to_drop = [
            "student_id",                # identifier
            "sleep_time", "wake_time",   # replaced by sleep_duration
            "sleep_hr", "wake_hr",       # helper columns
            "n_male", "n_female",        # combined into class_size
        ]
        olevel_df = olevel_df.drop(columns=cols_to_drop).reset_index(drop=True)

        logging.info(f"Data cleaning completed: {olevel_df.shape[0]} rows, {olevel_df.shape[1]} columns.")
        return olevel_df

    def impute_attendance(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Impute missing attendance_rate with the median per sleep_duration group.

        Sleep duration and attendance are strongly linked (r = 0.89), so group medians
        are more accurate than a global median. Medians are learned from the training
        set only and applied to all three sets to avoid data leakage.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Imputed train, validation and test features.
        """
        medians = X_train.groupby("sleep_duration")["attendance_rate"].median()
        global_median = X_train["attendance_rate"].median()  # fallback for unseen groups

        imputed = []
        for df_ in (X_train, X_val, X_test):
            df_ = df_.copy()
            fill = df_["sleep_duration"].map(medians).fillna(global_median)
            df_["attendance_rate"] = df_["attendance_rate"].fillna(fill)
            imputed.append(df_)

        logging.info("Imputed missing attendance_rate using training-set group medians.")
        return tuple(imputed)

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical and nominal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        # drop="if_binary" avoids the dummy-variable trap for two-category features
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
            ],
            remainder="drop",
        )
        return preprocessor

    @staticmethod
    def to_hours(t: str) -> float:
        """
        Convert a time string in hh:mm format to a number of hours.

        Args:
        -----
        t (str): Time as a string, e.g. "22:30".

        Returns:
        --------
        float: The time in hours, e.g. 22.5.
        """
        h, m = map(int, t.split(":"))
        return h + (m / 60)