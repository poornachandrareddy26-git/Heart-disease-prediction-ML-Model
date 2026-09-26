import numpy as np
import pandas as pd
import sys

from scipy.stats import yeojohnson

from log import setup_logging

logger = setup_logging("variable_transformation")


def outliers(X_train, X_test):

    try:

        # Store transformation information
        transformation_info = {}

        logger.info(
            f"Before X_train column names : {X_train.columns}"
        )

        logger.info(
            f"Before X_test column names : {X_test.columns}"
        )

        # -----------------------------------
        # Yeo-Johnson transformation
        # -----------------------------------

        # Make copies so the original DataFrames
        # are not modified unexpectedly
        X_train = X_train.copy()
        X_test = X_test.copy()

        # Store original column names
        original_columns = X_train.columns.tolist()

        for i in original_columns:

            # -----------------------------------
            # Learn lambda ONLY from training data
            # -----------------------------------

            X_train_yeo, lam_value = yeojohnson(
                X_train[i]
            )

            # -----------------------------------
            # Apply SAME lambda to test data
            # -----------------------------------

            X_test_yeo = yeojohnson(
                X_test[i],
                lmbda=lam_value
            )

            # -----------------------------------
            # Calculate IQR from transformed
            # training data
            # -----------------------------------

            q1 = np.percentile(
                X_train_yeo,
                25
            )

            q3 = np.percentile(
                X_train_yeo,
                75
            )

            iqr = q3 - q1

            upper_limit = q3 + (1.5 * iqr)

            lower_limit = q1 - (1.5 * iqr)

            # -----------------------------------
            # Store transformation information
            # -----------------------------------

            transformation_info[i] = {
                "lambda": lam_value,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit
            }

            # -----------------------------------
            # Trim training outliers
            # -----------------------------------

            X_train_trim = np.where(
                X_train_yeo > upper_limit,
                upper_limit,
                np.where(
                    X_train_yeo < lower_limit,
                    lower_limit,
                    X_train_yeo
                )
            )

            # -----------------------------------
            # Trim testing outliers
            # -----------------------------------

            X_test_trim = np.where(
                X_test_yeo > upper_limit,
                upper_limit,
                np.where(
                    X_test_yeo < lower_limit,
                    lower_limit,
                    X_test_yeo
                )
            )

            # -----------------------------------
            # Create final transformed columns
            # -----------------------------------

            X_train[i + "_yeo_trim"] = X_train_trim

            X_test[i + "_yeo_trim"] = X_test_trim

        # -----------------------------------
        # Remove original columns
        # -----------------------------------

        X_train = X_train[
            [
                i + "_yeo_trim"
                for i in original_columns
            ]
        ]

        X_test = X_test[
            [
                i + "_yeo_trim"
                for i in original_columns
            ]
        ]

        logger.info(
            f"After X_train column names : {X_train.columns}"
        )

        logger.info(
            f"After X_test column names : {X_test.columns}"
        )

        logger.info(
            "Transformation information stored successfully"
        )

        # -----------------------------------
        # Return transformed data + parameters
        # -----------------------------------

        return (
            X_train,
            X_test,
            transformation_info
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )

        raise