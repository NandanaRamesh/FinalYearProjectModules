import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


# --- IMPORTANT: PLACE YOUR SAVED MODEL AND TRAINING DATA IN THE SAME DIRECTORY ---
# The model file saved from your Jupyter notebook
MODEL_FILE = os.path.join("assets", "xgboost.joblib")
# A sample of the training data used for SHAP explainer
TRAINING_DATA_FILE = os.path.join("assets", "train_data.csv")
def load_model_and_data():
    """Loads the pre-trained model and training data for the explainer."""
    try:
        model = joblib.load(MODEL_FILE)
        training_data = pd.read_csv(TRAINING_DATA_FILE)
        
        # Clean up column names by stripping leading/trailing whitespace
        training_data.columns = training_data.columns.str.strip()

        # Drop the target column and any ID columns from the training data
        training_data = training_data.drop(columns=['loan_status'], errors='ignore')

        # The SHAP explainer needs the training data in the same numerical format
        # that the model was trained on. Do NOT convert categorical columns back to strings.
        
        return model, training_data
    except FileNotFoundError:
        st.error("Model or training data file not found. Please ensure that the file paths are correct.")
        return None, None
    except ImportError as e:
        st.error(f"Failed to load model due to a missing library: {e}. Please ensure all required libraries (like xgboost) are installed.")
        return None, None

def loan_eligibility_page():
    st.header("Loan Eligibility Check")
    st.markdown("Enter the applicant's details to get a prediction on their loan eligibility.")

    # Load model and data
    model, training_data = load_model_and_data()

    if model is None or training_data is None:
        return

    # User inputs for loan application features
    st.subheader("Applicant Details")
    no_of_dependents = st.selectbox('Number of Dependents', ('0', '1', '2', '3+'))
    education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
    self_employed = st.selectbox('Self Employed', ('Yes', 'No'))
    income_annum = st.number_input('Annual Income', min_value=0, step=1000)
    loan_amount = st.number_input('Loan Amount', min_value=0, step=1000)
    loan_term = st.selectbox('Loan Term (in months)', (360, 180, 120, 300, 240, 60, 90, 480, 84, 36, 12))
    cibil_score = st.number_input('CIBIL Score', min_value=300, max_value=900)
    movable_assets = st.number_input('Movable Assets Value', min_value=0, step=1000)
    immovable_assets = st.number_input('Immovable Assets Value', min_value=0, step=1000)

    if st.button('Check Eligibility'):
        try:
            # Manually encode categorical inputs into one-hot format
            input_data_dict = {
                'income_annum': income_annum,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'cibil_score': cibil_score,
                'Movable_assets': movable_assets,
                'Immovable_assets': immovable_assets,
                'no_of_dependents_0': 1 if no_of_dependents == '0' else 0,
                'no_of_dependents_1': 1 if no_of_dependents == '1' else 0,
                'no_of_dependents_2': 1 if no_of_dependents == '2' else 0,
                'no_of_dependents_3+': 1 if no_of_dependents == '3+' else 0,
                'education_Not Graduate': 1 if education == 'Not Graduate' else 0,
                'self_employed_Yes': 1 if self_employed == 'Yes' else 0
            }
            input_df = pd.DataFrame([input_data_dict])

            # Use training data (already preprocessed) to align columns
            training_processed = training_data.copy()
            input_processed = input_df.reindex(columns=training_processed.columns, fill_value=0)

            # Ensure all dtypes are numeric (float)
            input_processed = input_processed.astype(float)

            # === Make prediction ===
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[:, 1][0]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"Loan Accepted! 🎉 (Probability: {probability:.2f})")
            else:
                st.error(f"Loan Rejected. 😔 (Probability: {probability:.2f})")

            # === SHAP Explanation ===
            st.subheader("Explanation (SHAP)")
            st.markdown("The SHAP force plot below shows how each factor contributes to the loan decision.")
            st.markdown("Red values push the prediction towards approval, while blue values push it towards rejection.")
            
            explainer = shap.Explainer(model, training_processed)
            shap_values = explainer(input_processed)

            shap.initjs()
            fig = shap.force_plot(
                shap_values.base_values[0],
                shap_values.values[0],
                input_processed.iloc[0],
                matplotlib=True,
                show=False,
            )
            st.pyplot(fig)

            # === LIME Explanation ===
            from lime.lime_tabular import LimeTabularExplainer

            st.subheader("Explanation (LIME)")
            st.markdown("The LIME plot below shows which features most influenced this prediction.")

            lime_explainer = LimeTabularExplainer(
                training_processed.values,
                feature_names=training_processed.columns.tolist(),
                class_names=["Rejected", "Accepted"],
                mode="classification"
            )

            lime_exp = lime_explainer.explain_instance(
                input_processed.iloc[0].values,
                model.predict_proba,
                num_features=10
            )

            fig_lime = lime_exp.as_pyplot_figure()
            st.pyplot(fig_lime)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check the format of your `loan_model.joblib` and `train_data.csv` files.")

