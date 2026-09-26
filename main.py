import pickle
import sys
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

from log import setup_logging
logger = setup_logging("main")

from variable_transformation import outliers
from feature_selection import best_col
from all_models import common


class HEART:

    def __init__(self, path):

        try:

            self.path = path
            self.transformation_info = None

            self.df = pd.read_csv(path)

            logger.info(
                f"Null values in the data:\n{self.df.isnull().sum()}"
            )

            self.x = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]

            # -----------------------------------
            # Train-test split
            # -----------------------------------

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.x,
                self.y,
                test_size=0.2,
                random_state=42
            )

            # Save ORIGINAL test data
            # before preprocessing
            self.X_test_original = self.X_test.copy()

            logger.info(
                f"Training dataset size : "
                f"{self.X_train.shape} => {self.y_train.shape}"
            )

            logger.info(
                f"Testing dataset size : "
                f"{self.X_test.shape} => {self.y_test.shape}"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.info(
                f"Error in line no : {er_line.tb_lineno} : "
                f"due to : {er_type} : reason : {er_msg}"
            )

    # -----------------------------------
    # Variable Transformation
    # -----------------------------------

    def vt_outliers(self):

        try:

            result = outliers(
                self.X_train,
                self.X_test
            )

            self.X_train = result[0]
            self.X_test = result[1]

            self.transformation_info = result[2]

            logger.info(
                "Transformation information stored successfully"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.info(
                f"Error in line no : {er_line.tb_lineno} : "
                f"due to : {er_type} : reason : {er_msg}"
            )

    # -----------------------------------
    # Feature Selection
    # -----------------------------------

    def feature_selection(self):

        try:

            self.X_train, self.X_test = best_col(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.info(
                f"Error in line no : {er_line.tb_lineno} : "
                f"due to : {er_type} : reason : {er_msg}"
            )

    # -----------------------------------
    # Data Balancing + Scaling
    # -----------------------------------

    def data_balancing(self):

        try:

            logger.info(
                f"Data balancing : "
                f"{self.X_train.shape} => {self.y_train.shape}"
            )

            logger.info(
                f"Number of target 1 : "
                f"{sum(self.y_train == 1)}"
            )

            logger.info(
                f"Number of target 0 : "
                f"{sum(self.y_train == 0)}"
            )

            # -----------------------------------
            # SMOTE
            # -----------------------------------

            sm_obj = SMOTE(random_state=42)

            self.X_train_bal, self.y_train_bal = sm_obj.fit_resample(
                self.X_train,
                self.y_train
            )

            logger.info(
                f"After data balancing : "
                f"{self.X_train_bal.shape} => "
                f"{self.y_train_bal.shape}"
            )

            logger.info(
                f"Balanced target 1 : "
                f"{sum(self.y_train_bal == 1)}"
            )

            logger.info(
                f"Balanced target 0 : "
                f"{sum(self.y_train_bal == 0)}"
            )

            # -----------------------------------
            # Standard Scaling
            # -----------------------------------

            self.sc = StandardScaler()

            # FIT only on training data
            self.X_train_bal_scaled = self.sc.fit_transform(
                self.X_train_bal
            )

            # TRANSFORM test data using same scaler
            self.X_test_scaled = self.sc.transform(
                self.X_test
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.info(
                f"Error in line no : {er_line.tb_lineno} : "
                f"due to : {er_type} : reason : {er_msg}"
            )

    # -----------------------------------
    # All Models
    # -----------------------------------

    def all_model(self):

        try:

            self.model = common(
                self.X_train_bal_scaled,
                self.y_train_bal,
                self.X_test_scaled,
                self.y_test
            )

            with open("model.pkl", "wb") as file:
                pickle.dump(self.model, file)

            preprocessing = {
                "scaler": self.sc,
                "transformation_info": self.transformation_info
            }

            with open("scaler.pkl", "wb") as file:
                pickle.dump(preprocessing, file)

            logger.info(
                "Model saved successfully"
            )

            logger.info(
                "Preprocessing saved successfully"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.info(
                f"Error in line no : {er_line.tb_lineno} : "
                f"due to : {er_type} : reason : {er_msg}"
            )

    # -----------------------------------
    # Hyperparameter Tuning
    # -----------------------------------

    def best_model(self):

        try:

            nb_obj = GaussianNB()

            parameter_list = {
                "var_smoothing": np.logspace(-12, -1, 12)
            }

            grid_obj = GridSearchCV(
                estimator=nb_obj,
                param_grid=parameter_list,
                cv=10,
                scoring="accuracy",
                n_jobs=-1
            )

            grid_obj.fit(
                self.X_train_bal_scaled,
                self.y_train_bal
            )

            logger.info(
                f"Best parameters: "
                f"{grid_obj.best_params_}"
            )

            logger.info(
                f"Best CV score: "
                f"{grid_obj.best_score_}"
            )

            # Best tuned model
            self.model = grid_obj.best_estimator_

            # -----------------------------------
            # Predictions on test data
            # -----------------------------------

            prediction = self.model.predict(
                self.X_test_scaled
            )

            # -----------------------------------
            # First five test samples
            # -----------------------------------

            for i in range(5):

                logger.info(
                    f"Test sample {i} | "
                    f"Actual: {self.y_test.iloc[i]} | "
                    f"Predicted: {prediction[i]}"
                )

            # -----------------------------------
            # Test sample 2 details
            # -----------------------------------

            logger.info(
                "========== TEST SAMPLE 2 DETAILS =========="
            )

            logger.info(
                f"Test sample 2 original features:\n"
                f"{self.X_test_original.iloc[2]}"
            )

            logger.info(
                f"Test sample 2 actual target: "
                f"{self.y_test.iloc[2]}"
            )

            logger.info(
                f"Test sample 2 prediction: "
                f"{prediction[2]}"
            )

            # -----------------------------------
            # Model evaluation
            # -----------------------------------

            logger.info(
                f"Test accuracy: "
                f"{accuracy_score(self.y_test, prediction)}"
            )

            logger.info(
                f"Confusion matrix:\n"
                f"{confusion_matrix(self.y_test, prediction)}"
            )

            logger.info(
                f"Classification report:\n"
                f"{classification_report(self.y_test, prediction)}"
            )

            # -----------------------------------
            # Save tuned model
            # -----------------------------------

            with open("model.pkl", "wb") as file:

                pickle.dump(
                    self.model,
                    file
                )

            # -----------------------------------
            # Save preprocessing
            # -----------------------------------

            preprocessing = {
                "scaler": self.sc,
                "transformation_info": self.transformation_info
            }

            with open("scaler.pkl", "wb") as file:

                pickle.dump(
                    preprocessing,
                    file
                )

            logger.info(
                "Tuned Naive Bayes model saved successfully"
            )

            logger.info(
                "Preprocessing saved successfully"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.info(
                f"Error in line no : {er_line.tb_lineno} : "
                f"due to : {er_type} : reason : {er_msg}"
            )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    try:

        obj = HEART(
            "dataset/heart_disease_dataset.csv"
        )

        obj.vt_outliers()

        obj.feature_selection()

        obj.data_balancing()

        # Keep all_models commented
        # because we are using tuned Naive Bayes

        # obj.all_model()

        obj.best_model()

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )