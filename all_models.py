import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from log import setup_logging

logger = setup_logging("all_models")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.metrics import roc_curve


def knn_algo(X_train, y_train, X_test, y_test):

    try:
        global knn_reg

        knn_reg = KNeighborsClassifier(n_neighbors=5)

        knn_reg.fit(X_train, y_train)

        logger.info(
            f"Test Data Accuracy : "
            f"{accuracy_score(y_test, knn_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Confusion Matrix : "
            f"{confusion_matrix(y_test, knn_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Classification Report : "
            f"{classification_report(y_test, knn_reg.predict(X_test))}"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )


def nb_algo(X_train, y_train, X_test, y_test):

    try:
        global nb_reg

        nb_reg = GaussianNB()

        nb_reg.fit(X_train, y_train)

        logger.info(
            f"Test Data Accuracy : "
            f"{accuracy_score(y_test, nb_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Confusion Matrix : "
            f"{confusion_matrix(y_test, nb_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Classification Report : "
            f"{classification_report(y_test, nb_reg.predict(X_test))}"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )


def lr_algo(X_train, y_train, X_test, y_test):

    try:
        global lr_reg

        lr_reg = LogisticRegression()

        lr_reg.fit(X_train, y_train)

        logger.info(
            f"Test Data Accuracy : "
            f"{accuracy_score(y_test, lr_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Confusion Matrix : "
            f"{confusion_matrix(y_test, lr_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Classification Report : "
            f"{classification_report(y_test, lr_reg.predict(X_test))}"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )


def dt_algo(X_train, y_train, X_test, y_test):

    try:
        global dt_reg

        dt_reg = DecisionTreeClassifier(
            criterion="entropy"
        )

        dt_reg.fit(X_train, y_train)

        logger.info(
            f"Test Data Accuracy : "
            f"{accuracy_score(y_test, dt_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Confusion Matrix : "
            f"{confusion_matrix(y_test, dt_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Classification Report : "
            f"{classification_report(y_test, dt_reg.predict(X_test))}"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )


def rf_algo(X_train, y_train, X_test, y_test):

    try:
        global rf_reg

        rf_reg = RandomForestClassifier(
            criterion="entropy",
            n_estimators=10
        )

        rf_reg.fit(X_train, y_train)

        logger.info(
            f"Test Data Accuracy : "
            f"{accuracy_score(y_test, rf_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Confusion Matrix : "
            f"{confusion_matrix(y_test, rf_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Classification Report : "
            f"{classification_report(y_test, rf_reg.predict(X_test))}"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )


def ada_algo(X_train, y_train, X_test, y_test):

    try:
        global ada_reg

        lr = LogisticRegression()

        ada_reg = AdaBoostClassifier(
            estimator=lr,
            n_estimators=10
        )

        ada_reg.fit(X_train, y_train)

        logger.info(
            f"Test Data Accuracy : "
            f"{accuracy_score(y_test, ada_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Confusion Matrix : "
            f"{confusion_matrix(y_test, ada_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Classification Report : "
            f"{classification_report(y_test, ada_reg.predict(X_test))}"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )


def gb_algo(X_train, y_train, X_test, y_test):

    try:
        global gb_reg

        gb_reg = GradientBoostingClassifier(
            n_estimators=10
        )

        gb_reg.fit(X_train, y_train)

        logger.info(
            f"Test Data Accuracy : "
            f"{accuracy_score(y_test, gb_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Confusion Matrix : "
            f"{confusion_matrix(y_test, gb_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Classification Report : "
            f"{classification_report(y_test, gb_reg.predict(X_test))}"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )


def xgb_algo(X_train, y_train, X_test, y_test):

    try:
        global xgb_reg

        xgb_reg = XGBClassifier()

        xgb_reg.fit(X_train, y_train)

        logger.info(
            f"Test Data Accuracy : "
            f"{accuracy_score(y_test, xgb_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Confusion Matrix : "
            f"{confusion_matrix(y_test, xgb_reg.predict(X_test))}"
        )

        logger.info(
            f"Test Data Classification Report : "
            f"{classification_report(y_test, xgb_reg.predict(X_test))}"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )


def auc_roc_curve(X_train, y_train, X_test, y_test):

    try:

        knn_predictions = knn_reg.predict(X_test)
        lr_predictions = lr_reg.predict(X_test)
        nb_predictions = nb_reg.predict(X_test)
        dt_predictions = dt_reg.predict(X_test)
        rf_predictions = rf_reg.predict(X_test)
        ada_predictions = ada_reg.predict(X_test)
        gb_predictions = gb_reg.predict(X_test)
        xgb_predictions = xgb_reg.predict(X_test)

        knn_fpr, knn_tpr, knn_thre = roc_curve(
            y_test, knn_predictions
        )

        lr_fpr, lr_tpr, lr_thre = roc_curve(
            y_test, lr_predictions
        )

        nb_fpr, nb_tpr, nb_thre = roc_curve(
            y_test, nb_predictions
        )

        dt_fpr, dt_tpr, dt_thre = roc_curve(
            y_test, dt_predictions
        )

        rf_fpr, rf_tpr, rf_thre = roc_curve(
            y_test, rf_predictions
        )

        ada_fpr, ada_tpr, ada_thre = roc_curve(
            y_test, ada_predictions
        )

        gb_fpr, gb_tpr, gb_thre = roc_curve(
            y_test, gb_predictions
        )

        xgb_fpr, xgb_tpr, xgb_thre = roc_curve(
            y_test, xgb_predictions
        )

        plt.figure(figsize=(5, 3))

        plt.title("AUC and ROC curve")

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.plot(
            knn_fpr,
            knn_tpr,
            label="KNN"
        )

        plt.plot(
            lr_fpr,
            lr_tpr,
            label="LR"
        )

        plt.plot(
            nb_fpr,
            nb_tpr,
            label="NB"
        )

        plt.plot(
            dt_fpr,
            dt_tpr,
            label="DT"
        )

        plt.plot(
            rf_fpr,
            rf_tpr,
            label="RF"
        )

        plt.plot(
            ada_fpr,
            ada_tpr,
            label="ADA"
        )

        plt.plot(
            gb_fpr,
            gb_tpr,
            label="GB"
        )

        plt.plot(
            xgb_fpr,
            xgb_tpr,
            label="XGB"
        )

        plt.legend(loc=0)

        plt.show()

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )


def common(X_train, y_train, X_test, y_test):

    try:

        logger.info("======== KNN ==========")
        knn_algo(
            X_train,
            y_train,
            X_test,
            y_test
        )

        logger.info(
            "======== Naive Bayes =========="
        )

        nb_algo(
            X_train,
            y_train,
            X_test,
            y_test
        )

        logger.info(
            "======== Logistic Regression =========="
        )

        lr_algo(
            X_train,
            y_train,
            X_test,
            y_test
        )

        logger.info(
            "======== Decision Tree =========="
        )

        dt_algo(
            X_train,
            y_train,
            X_test,
            y_test
        )

        logger.info(
            "======== Random Forest =========="
        )

        rf_algo(
            X_train,
            y_train,
            X_test,
            y_test
        )

        logger.info(
            "======== AdaBoost =========="
        )

        ada_algo(
            X_train,
            y_train,
            X_test,
            y_test
        )

        logger.info(
            "======== Gradient Boosting =========="
        )

        gb_algo(
            X_train,
            y_train,
            X_test,
            y_test
        )

        logger.info(
            "======== XGBoost =========="
        )

        xgb_algo(
            X_train,
            y_train,
            X_test,
            y_test
        )

        logger.info(
            "======== AUC and ROC Curve =========="
        )

        auc_roc_curve(
            X_train,
            y_train,
            X_test,
            y_test
        )

        return nb_reg

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.info(
            f"Error in line no : {er_line.tb_lineno} : "
            f"due to : {er_type} : reason : {er_msg}"
        )