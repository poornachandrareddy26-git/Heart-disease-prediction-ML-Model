import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

from log import setup_logging

logger = setup_logging("feature_selection")


def best_col(X_train, X_test, y_train, y_test):

    try:

        logger.info(
            f"Before constant technique : "
            f"{X_train.columns}, {X_train.shape}, "
            f"{X_test.columns}, {X_test.shape}"
        )

        # -----------------------------------
        # 1. Constant Feature Removal
        # -----------------------------------

        cons_obj = VarianceThreshold(threshold=0.0)

        cons_obj.fit(X_train)

        logger.info(
            f"Columns to remove : "
            f"{X_train.columns[~cons_obj.get_support()]}"
        )

        X_train = X_train.drop(
            ["fbs_yeo_trim"],
            axis=1
        )

        X_test = X_test.drop(
            ["fbs_yeo_trim"],
            axis=1
        )

        logger.info(
            f"After constant technique : "
            f"{X_train.columns}, {X_train.shape}, "
            f"{X_test.columns}, {X_test.shape}"
        )

        # -----------------------------------
        # 2. Quasi-Constant Feature Removal
        # -----------------------------------

        quasi_obj = VarianceThreshold(threshold=0.1)

        quasi_obj.fit(X_train)

        logger.info(
            f"Columns to remove : "
            f"{X_train.columns[~quasi_obj.get_support()]}"
        )

        X_train = X_train.drop(
            [
                "trestbps_yeo_trim",
                "chol_yeo_trim",
                "exang_yeo_trim",
                "ca_yeo_trim"
            ],
            axis=1
        )

        X_test = X_test.drop(
            [
                "trestbps_yeo_trim",
                "chol_yeo_trim",
                "exang_yeo_trim",
                "ca_yeo_trim"
            ],
            axis=1
        )

        logger.info(
            f"After quasi constant : "
            f"{X_train.columns}, {X_train.shape}, "
            f"{X_test.columns}, {X_test.shape}"
        )

        # -----------------------------------
        # 3. Hypothesis Testing
        # -----------------------------------

        X_train = X_train.drop(
            ["restecg_yeo_trim"],
            axis=1
        )

        X_test = X_test.drop(
            ["restecg_yeo_trim"],
            axis=1
        )

        logger.info(
            f"After hypothesis testing : "
            f"{X_train.columns}, {X_train.shape}, "
            f"{X_test.columns}, {X_test.shape}"
        )

        return X_train, X_test

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )