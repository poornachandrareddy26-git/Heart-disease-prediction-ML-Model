from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson

app = Flask(__name__)


# -----------------------------------
# Load trained model
# -----------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# -----------------------------------
# Load scaler + transformation info
# -----------------------------------

with open("scaler.pkl", "rb") as f:
    preprocessing = pickle.load(f)

scaler = preprocessing["scaler"]
transformation_info = preprocessing["transformation_info"]


# -----------------------------------
# Selected features
# -----------------------------------

FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "thalach",
    "oldpeak",
    "slope",
    "thal"
]


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        try:

            # -----------------------------------
            # Get input values from HTML form
            # -----------------------------------

            features = [
                float(request.form[col])
                for col in FEATURE_ORDER
            ]

            input_data = pd.DataFrame(
                [features],
                columns=FEATURE_ORDER
            )

            # -----------------------------------
            # Apply Yeo-Johnson transformation
            # using training parameters
            # -----------------------------------

            transformed_data = pd.DataFrame(
                index=input_data.index
            )

            for column in FEATURE_ORDER:

                lambda_value = transformation_info[column]["lambda"]

                lower_limit = transformation_info[column]["lower_limit"]

                upper_limit = transformation_info[column]["upper_limit"]

                transformed_value = yeojohnson(
                    input_data[column],
                    lmbda=lambda_value
                )

                # -----------------------------------
                # Apply training outlier limits
                # -----------------------------------

                transformed_value = np.where(
                    transformed_value > upper_limit,
                    upper_limit,
                    np.where(
                        transformed_value < lower_limit,
                        lower_limit,
                        transformed_value
                    )
                )

                transformed_data[
                    column + "_yeo_trim"
                ] = transformed_value

            # -----------------------------------
            # Keep selected seven features
            # -----------------------------------

            final_data = transformed_data[
                [
                    "age_yeo_trim",
                    "sex_yeo_trim",
                    "cp_yeo_trim",
                    "thalach_yeo_trim",
                    "oldpeak_yeo_trim",
                    "slope_yeo_trim",
                    "thal_yeo_trim"
                ]
            ]

            # -----------------------------------
            # Apply fitted StandardScaler
            # -----------------------------------

            features_scaled = scaler.transform(
                final_data
            )

            # -----------------------------------
            # Model prediction
            # -----------------------------------

            prediction_value = model.predict(
                features_scaled
            )[0]

            # -----------------------------------
            # TEMPORARY:
            # Show raw model prediction
            # -----------------------------------

            if prediction_value == 1:
                prediction = "Heart Disease Detected"
            else:
                prediction = "No Heart Disease"

        except Exception as e:

            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run()