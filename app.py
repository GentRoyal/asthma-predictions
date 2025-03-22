import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

# Page config
st.set_page_config(
    page_title="Asthma Risk Prediction",
    page_icon="🫁",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('bestmodel.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the scaler
@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        # If scaler file doesn't exist, create a new one
        st.warning("Scaler file not found. Using default StandardScaler.")
        return StandardScaler()

# Main function
def main():
    # App title and description
    st.title("Asthma Risk Prediction System")
    st.markdown("""
    This application predicts asthma risk based on environmental factors and personal health data.
    Complete the form below to receive your personalized asthma risk assessment.
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the prediction model. Please check if 'bestmodel.pkl' exists in the correct directory.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Prediction", "About"])
    
    with tab1:
        st.header("Enter Your Information")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Demographic Information
        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Hispanic", "Asian", "Other"])
            education = st.selectbox("Education Level", ["Less than High School", "High School", "Some College", "Bachelor's Degree", "Graduate Degree"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5)
        
        # Lifestyle Factors
        with col2:
            st.subheader("Lifestyle Factors")
            smoking = st.radio("Smoking Status", ["Non-smoker", "Former Smoker", "Current Smoker"], index=0)
            physical_activity = st.slider("Physical Activity (hours/week)", 0.0, 20.0, 3.0, 0.5)
            diet_quality = st.slider("Diet Quality (1-10)", 1, 10, 5)
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 5)
            
            # Environmental Exposures
            st.subheader("Environmental Exposures")
            pollution_exposure = st.slider("Pollution Exposure (1-10)", 1, 10, 5)
            pollen_exposure = st.slider("Pollen Exposure (1-10)", 1, 10, 5)
            dust_exposure = st.slider("Dust Exposure (1-10)", 1, 10, 5)
        
        # Medical History
        with col3:
            st.subheader("Medical History")
            pet_allergy = st.checkbox("Pet Allergy")
            family_history = st.checkbox("Family History of Asthma")
            history_allergies = st.checkbox("History of Allergies")
            eczema = st.checkbox("Eczema")
            hay_fever = st.checkbox("Hay Fever")
            gerd = st.checkbox("Gastroesophageal Reflux (GERD)")
            
            # Lung Function & Symptoms
            st.subheader("Lung Function & Symptoms")
            fev1 = st.number_input("Lung Function FEV1 (%)", min_value=20.0, max_value=150.0, value=90.0)
            fvc = st.number_input("Lung Function FVC (%)", min_value=20.0, max_value=150.0, value=90.0)
            
            wheezing = st.checkbox("Wheezing")
            shortness_breath = st.checkbox("Shortness of Breath")
            chest_tightness = st.checkbox("Chest Tightness")
            coughing = st.checkbox("Coughing")
            nighttime_symptoms = st.checkbox("Nighttime Symptoms")
            exercise_induced = st.checkbox("Exercise-Induced Symptoms")
        
        # Prediction button
        predict_button = st.button("Predict Asthma Risk")
        
        if predict_button:
            # Prepare the input data
            # Map categorical inputs to numerical values based on the training data
            gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
            ethnicity_map = {'Caucasian': 0, 'African American': 1, 'Hispanic': 2, 'Asian': 3, 'Other': 4}
            education_map = {'Less than High School': 0, 'High School': 1, 'Some College': 2, 'Bachelor\'s Degree': 3, 'Graduate Degree': 4}
            smoking_map = {'Non-smoker': 0, 'Former Smoker': 1, 'Current Smoker': 2}
            
            # Convert boolean to int
            bool_to_int = lambda x: 1 if x else 0
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender_map[gender]],
                'Ethnicity': [ethnicity_map[ethnicity]],
                'EducationLevel': [education_map[education]],
                'BMI': [bmi],
                'Smoking': [smoking_map[smoking]],
                'PhysicalActivity': [physical_activity],
                'DietQuality': [diet_quality],
                'SleepQuality': [sleep_quality],
                'PollutionExposure': [pollution_exposure],
                'PollenExposure': [pollen_exposure],
                'DustExposure': [dust_exposure],
                'PetAllergy': [bool_to_int(pet_allergy)],
                'FamilyHistoryAsthma': [bool_to_int(family_history)],
                'HistoryOfAllergies': [bool_to_int(history_allergies)],
                'Eczema': [bool_to_int(eczema)],
                'HayFever': [bool_to_int(hay_fever)],
                'GastroesophagealReflux': [bool_to_int(gerd)],
                'LungFunctionFEV1': [fev1],
                'LungFunctionFVC': [fvc],
                'Wheezing': [bool_to_int(wheezing)],
                'ShortnessOfBreath': [bool_to_int(shortness_breath)],
                'ChestTightness': [bool_to_int(chest_tightness)],
                'Coughing': [bool_to_int(coughing)],
                'NighttimeSymptoms': [bool_to_int(nighttime_symptoms)],
                'ExerciseInduced': [bool_to_int(exercise_induced)]
            })

            input_data["Gender"] = input_data["Gender"].astype("category")
            input_data["Ethnicity"] = input_data["Ethnicity"].astype("category")
            input_data["EducationLevel"] = input_data["EducationLevel"].astype("category")

            
            # Get the scaler
            scaler = load_scaler()
            
            # Scale the numerical features
            numeric_cols = input_data.select_dtypes(include=["int64", "float64"]).columns
            input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.divider()
            st.header("Your Asthma Risk Assessment")
            
            # Define risk levels based on model output
            if prediction == 1:
                risk_level = "High Risk"
                risk_color = "red"
            else:
                risk_level = "Low Risk"
                risk_color = "green"
            
            # Display prediction with styling
            st.markdown(f"<h2 style='color:{risk_color};'>Prediction: {risk_level}</h2>", unsafe_allow_html=True)
            
            # Display confidence
            confidence = max(prediction_proba) * 100
            st.markdown(f"Confidence: {confidence:.2f}%")
            
            # Display primary risk factors
            st.subheader("Your Primary Risk Factors")
            
            risk_factors = []
            if bool_to_int(family_history) == 1:
                risk_factors.append("Family history of asthma")
            if bool_to_int(history_allergies) == 1:
                risk_factors.append("History of allergies")
            if bool_to_int(hay_fever) == 1:
                risk_factors.append("Hay fever")
            if bool_to_int(eczema) == 1:
                risk_factors.append("Eczema")
            if bool_to_int(wheezing) == 1:
                risk_factors.append("Wheezing")
            if bool_to_int(shortness_breath) == 1:
                risk_factors.append("Shortness of breath")
            if bool_to_int(nighttime_symptoms) == 1:
                risk_factors.append("Nighttime symptoms")
            if bool_to_int(exercise_induced) == 1:
                risk_factors.append("Exercise-induced symptoms")
            if pollution_exposure > 7:
                risk_factors.append("High pollution exposure")
            if pollen_exposure > 7:
                risk_factors.append("High pollen exposure")
            if dust_exposure > 7:
                risk_factors.append("High dust exposure")
            if fev1 < 80:
                risk_factors.append("Reduced lung function (FEV1)")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("No significant risk factors identified.")
            
            # Recommendations
            st.subheader("Recommendations")
            if risk_level == "High Risk":
                st.markdown("""
                - Consult with a healthcare provider for a comprehensive asthma evaluation
                - Consider allergy testing to identify specific triggers
                - Develop an asthma action plan with your doctor
                - Monitor your symptoms regularly
                - Reduce exposure to identified environmental triggers
                """)
            else:
                st.markdown("""
                - Maintain a healthy lifestyle with regular exercise
                - Continue monitoring any symptoms that may develop
                - Consider reducing exposure to environmental triggers
                - Follow up with a healthcare provider if symptoms develop
                """)
            
            st.warning("This prediction is for informational purposes only and should not replace professional medical advice.")
    
    # About tab content
    with tab2:
        st.header("About This Application")
        st.markdown("""
        ### How It Works
        
        This application uses a machine learning model trained on data from individuals with 
        and without asthma diagnoses. The model analyzes various factors including:
        
        - Demographic information
        - Environmental exposures
        - Medical history
        - Lifestyle factors
        - Current symptoms
        - Lung function measurements
        
        ### Data Privacy
        
        All data entered is processed locally on your device and is not stored or transmitted elsewhere.
        
        ### Limitations
        
        This tool is intended for educational and informational purposes only. It should not be used as a 
        replacement for professional medical diagnosis, advice, or treatment. Always consult with a qualified 
        healthcare provider for medical concerns.
        
        ### Model Information
        
        The prediction model was developed using supervised machine learning techniques on a dataset of 2,392 
        individuals. The model has been validated, but as with all predictive tools, it has limitations and 
        should be used as one of many tools in healthcare decision-making.
        """)

if __name__ == '__main__':
    main()
