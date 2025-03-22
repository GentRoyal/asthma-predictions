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
            age = st.number_input("Age", min_value=5, max_value=79, value=30)
            
            # Gender (using exact categorical encoding from training)
            gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            
            # Ethnicity (using exact categorical encoding from training)
            ethnicity_options = {0: "Type 0", 1: "Type 1", 2: "Type 2", 3: "Type 3"}
            ethnicity = st.selectbox("Ethnicity", options=list(ethnicity_options.keys()), 
                                     format_func=lambda x: ethnicity_options[x])
            
            # Education Level (using exact categorical encoding from training)
            education_options = {0: "Level 0", 1: "Level 1", 2: "Level 2", 3: "Level 3"}
            education = st.selectbox("Education Level", options=list(education_options.keys()),
                                     format_func=lambda x: education_options[x])
            
            bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=24.5)
        
        # Lifestyle Factors
        with col2:
            st.subheader("Lifestyle Factors")
            smoking = st.radio("Smoking Status", [0, 1], format_func=lambda x: "Non-smoker" if x == 0 else "Smoker")
            
            physical_activity = st.slider("Physical Activity", 0.0, 10.0, 3.0, 0.1,
                                          help="Physical activity level from 0 to 10")
            
            diet_quality = st.slider("Diet Quality", 0.0, 10.0, 5.0, 0.1,
                                     help="Diet quality from 0 to 10")
            
            sleep_quality = st.slider("Sleep Quality", 4.0, 10.0, 7.0, 0.1,
                                      help="Sleep quality from 4 to 10")
            
            # Environmental Exposures
            st.subheader("Environmental Exposures")
            pollution_exposure = st.slider("Pollution Exposure", 0.0, 10.0, 5.0, 0.1,
                                          help="Exposure to air pollution from 0 to 10")
            
            pollen_exposure = st.slider("Pollen Exposure", 0.0, 10.0, 5.0, 0.1,
                                       help="Exposure to pollen from 0 to 10")
            
            dust_exposure = st.slider("Dust Exposure", 0.0, 10.0, 5.0, 0.1,
                                     help="Exposure to dust from 0 to 10")
        
        # Medical History
        with col3:
            st.subheader("Medical History")
            pet_allergy = st.radio("Pet Allergy", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            family_history = st.radio("Family History of Asthma", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            history_allergies = st.radio("History of Allergies", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            eczema = st.radio("Eczema", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            hay_fever = st.radio("Hay Fever", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            gerd = st.radio("Gastroesophageal Reflux (GERD)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            # Lung Function & Symptoms
            st.subheader("Lung Function & Symptoms")
            fev1 = st.number_input("Lung Function FEV1", min_value=1.0, max_value=4.0, value=2.5,
                                  help="Forced Expiratory Volume in 1 second")
            
            fvc = st.number_input("Lung Function FVC", min_value=1.5, max_value=6.0, value=3.5,
                                 help="Forced Vital Capacity")
            
            wheezing = st.radio("Wheezing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            shortness_breath = st.radio("Shortness of Breath", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            chest_tightness = st.radio("Chest Tightness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            coughing = st.radio("Coughing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            nighttime_symptoms = st.radio("Nighttime Symptoms", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            exercise_induced = st.radio("Exercise-Induced Symptoms", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        # Prediction button
        predict_button = st.button("Predict Asthma Risk")
        
        if predict_button:
            # Create input dataframe with exactly the same structure as training data
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Ethnicity': [ethnicity],
                'EducationLevel': [education],
                'BMI': [bmi],
                'Smoking': [smoking],
                'PhysicalActivity': [physical_activity],
                'DietQuality': [diet_quality],
                'SleepQuality': [sleep_quality],
                'PollutionExposure': [pollution_exposure],
                'PollenExposure': [pollen_exposure],
                'DustExposure': [dust_exposure],
                'PetAllergy': [pet_allergy],
                'FamilyHistoryAsthma': [family_history],
                'HistoryOfAllergies': [history_allergies],
                'Eczema': [eczema],
                'HayFever': [hay_fever],
                'GastroesophagealReflux': [gerd],
                'LungFunctionFEV1': [fev1],
                'LungFunctionFVC': [fvc],
                'Wheezing': [wheezing],
                'ShortnessOfBreath': [shortness_breath],
                'ChestTightness': [chest_tightness],
                'Coughing': [coughing],
                'NighttimeSymptoms': [nighttime_symptoms],
                'ExerciseInduced': [exercise_induced]
            })
            
            # Convert categorical columns to category dtype
            input_data["Gender"] = input_data["Gender"].astype("category")
            input_data["Ethnicity"] = input_data["Ethnicity"].astype("category")
            input_data["EducationLevel"] = input_data["EducationLevel"].astype("category")
            
            # Get the scaler
            scaler = load_scaler()
            
            # Scale the numerical features (avoid scaling categorical ones)
            numeric_cols = input_data.select_dtypes(include=["int64", "float64"]).columns
            input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
            
            try:
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
                if family_history == 1:
                    risk_factors.append("Family history of asthma")
                if history_allergies == 1:
                    risk_factors.append("History of allergies")
                if hay_fever == 1:
                    risk_factors.append("Hay fever")
                if eczema == 1:
                    risk_factors.append("Eczema")
                if wheezing == 1:
                    risk_factors.append("Wheezing")
                if shortness_breath == 1:
                    risk_factors.append("Shortness of breath")
                if nighttime_symptoms == 1:
                    risk_factors.append("Nighttime symptoms")
                if exercise_induced == 1:
                    risk_factors.append("Exercise-induced symptoms")
                if pollution_exposure > 7:
                    risk_factors.append("High pollution exposure")
                if pollen_exposure > 7:
                    risk_factors.append("High pollen exposure")
                if dust_exposure > 7:
                    risk_factors.append("High dust exposure")
                if fev1 < 2:
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
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please check if the input data format matches what the model expects.")
    
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
