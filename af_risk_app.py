import sys
import platform
import streamlit as st

st.write(f"Python Version: {sys.version}")
st.write(f"Platform: {platform.platform()}")
try:
    import sklearn
    st.write(f"Scikit-learn Version: {sklearn.__version__}")
except ImportError as e:
    st.write(f"Error importing Scikit-learn: {e}")
import streamlit as st
import pandas as pd
import numpy as np
# No need to import xgboost or GradientBoostingClassifier directly here,
# as the model is loaded from the pickled file.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve  # Not directly used in the app, but good for context
import pickle
import traceback  # For detailed errorpip freeze > requirements.txt logging
import warnings

# --- Global Settings ---
warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Feature definitions (Must be consistent with model training) ---
continuous_features = ['AGE', 'Apelin_12', 'NLRP3', 'NTproBNP', 'globulin',
                       'TC', 'HDL', 'triglyceride', 'Urea.nitrogen', 'Cr',
                       'LA', 'LV', 'RV', 'AAO', 'SV', 'DtoW', 'CKMB']
categorical_features = ['SEX']
target = 'AF'

# --- Display names for features ---
feature_name_display_map = {
    'AGE'          : 'AGE',
    'Apelin_12'    : 'Apelin_12',
    'NLRP3'        : 'NLRP3',
    'NTproBNP'     : 'NTproBNP',
    'globulin'     : 'Globulin',
    'TC'           : 'TC',
    'HDL'          : 'HDL',
    'triglyceride' : 'Triglyceride',
    'Urea.nitrogen': 'Urea Nitrogen',
    'Cr'           : 'Cr',
    'LA'           : 'LA',
    'LV'           : 'LV',
    'RV'           : 'RV',
    'AAO'          : 'AAO',
    'SV'           : 'SV',
    'DtoW'         : 'DtoW',
    'CKMB'         : 'CKMB',
    'SEX'          : 'SEX',
    'AF'           : 'AF'
}



# --- Utility Functions ---
@st.cache_resource(show_spinner="⏳ Loading pre-trained model and scaler...")
def load_pretrained_model_and_params():
    st.info("🔄 **Step 1/1**: Loading `af_risk_model_and_params.pkl`...")
    try:
        with open('af_risk_model_and_params.pkl', 'rb') as f:
            # Load the dictionary containing all saved objects
            saved_objects = pickle.load(f)

        # Renamed from xgboost_classifier_model to trained_model for generality
        trained_model = saved_objects['model']
        feature_scaler = saved_objects['scaler']
        optimal_threshold = saved_objects['optimal_threshold']
        feature_stats = saved_objects['feature_stats']  # Dictionary for slider min/max/mean
        all_features_for_models_loaded = saved_objects.get('all_features_for_models', [])  # Load the feature list

        st.success("✓ Pre-trained model loaded successfully.")
        return trained_model, feature_scaler, optimal_threshold, feature_stats, all_features_for_models_loaded
    except FileNotFoundError:
        st.error(
            "❌ **Error**: `af_risk_model_and_params.pkl` not found. Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Error**: Failed to load `af_risk_model_and_params.pkl`. Error message: {e}")
        st.code(traceback.format_exc())
        st.stop()
    return None, None, None, None, None


# --- Streamlit App Layout ---
st.set_page_config(
    page_title="AF Risk Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model and parameters at the start
trained_model, feature_scaler, optimal_threshold, feature_stats, all_features_for_models = load_pretrained_model_and_params()

st.title("AF Risk Prediction App")
st.markdown("""
This application helps assess the risk of Atrial Fibrillation (AF) based on patient features.
Please input the patient's data in the sidebar to get a personalized risk assessment.
""")

# ===========================
# Streamlit Sidebar: User Input Interface
# ===========================
st.sidebar.header("AF Risk Assessment")
st.sidebar.markdown("Please enter the patient's feature values below for Atrial Fibrillation (AF) risk assessment.")

user_patient_input = {}

st.sidebar.subheader("Continuous Features")
for feature_short_name in continuous_features:
    display_name = feature_name_display_map.get(feature_short_name, feature_short_name)
    stats = feature_stats.get(feature_short_name, {'min': 0, 'max': 100, 'mean': 50})  # Fallback for safety

    # Determine step size for the slider (e.g., 0.1 for floats, 1 for integers)
    # This logic assumes the 'mean' from feature_stats dictates the data type
    if isinstance(stats['mean'], float) and stats['mean'] != int(stats['mean']):
        step = 0.1
        format_str = "%.1f"
    else:
        step = 1
        format_str = "%d"

    user_patient_input[feature_short_name] = st.sidebar.slider(
        label=display_name,
        min_value=float(stats['min']),
        max_value=float(stats['max']),
        value=float(stats['mean']),
        step=step,
        format=format_str,
        key=f"slider_{feature_short_name}"
    )

st.sidebar.subheader("Categorical Features")
# Assuming categorical features are binary (0/1) and can be represented as Yes/No
for cat_feature in categorical_features:
    display_name = feature_name_display_map.get(cat_feature, cat_feature)
    user_patient_input[cat_feature] = st.sidebar.selectbox(
        label=display_name,
        options=['No', 'Yes'],  # Representing 0 and 1
        index=0,  # Default to 'No' (0)
        key=f"select_{cat_feature}"
    )

st.sidebar.markdown("---")
if st.sidebar.button("Assess AF Risk", type="primary"):
    # Convert user input to DataFrame
    input_df_original = pd.DataFrame([user_patient_input])

    # Process categorical features: convert 'Yes'/'No' to 1/0
    input_df_processed = input_df_original.copy()
    for cat_feature in categorical_features:
        input_df_processed[cat_feature] = input_df_processed[cat_feature].apply(lambda x: 1 if x == 'Yes' else 0)

    # Scale only continuous features
    # Ensure input_df_processed[continuous_features] has the correct columns and order
    scaled_continuous_features = feature_scaler.transform(input_df_processed[continuous_features])
    scaled_continuous_df = pd.DataFrame(scaled_continuous_features, columns=continuous_features,
                                        index=input_df_processed.index)

    # Combine scaled continuous features with original categorical features
    # Ensure the final DataFrame for prediction has all_features_for_models in the exact order
    final_input_for_prediction = pd.DataFrame(index=input_df_processed.index, columns=all_features_for_models)
    for col in all_features_for_models:
        if col in continuous_features:
            final_input_for_prediction[col] = scaled_continuous_df[col]
        elif col in categorical_features:
            final_input_for_prediction[col] = input_df_processed[col]

    # Make prediction (probability)
    prediction_proba = trained_model.predict_proba(final_input_for_prediction)[:, 1][0]  # Probability of AF (class 1)

    # Apply the optimal threshold
    prediction_class = 1 if prediction_proba >= optimal_threshold else 0

    st.subheader("Assessment Results")
    st.markdown(f"**Patient AF Risk Probability:** `{prediction_proba:.4f}`")
    st.markdown(f"**Optimal Classification Threshold:** `{optimal_threshold:.4f}`")

    if prediction_class == 1:
        st.error("⚠️ **Assessment:** High Risk of Atrial Fibrillation (AF)")
        st.write("Based on the input features, the model indicates a high risk of Atrial Fibrillation.")
    else:
        st.success("✅ **Assessment:** Low Risk of Atrial Fibrillation (AF)")
        st.write("Based on the input features, the model indicates a low risk of Atrial Fibrillation.")

    st.markdown("""
    ---
    **Disclaimer:** This is an automated risk assessment for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.
    """)


st.markdown("---")
