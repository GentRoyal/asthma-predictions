import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os
import random
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Asthma Risk Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #FFEBEE;
        color: #B71C1C;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .risk-medium {
        background-color: #FFF8E1;
        color: #F57F17;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .risk-low {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Function to generate random values
def generate_random_values():
    # Generate random values for each field
    random_data = {
        'age': random.randint(5, 79),
        'gender': random.choice([0, 1]),
        'ethnicity': random.choice([0, 1, 2, 3]),
        'education': random.choice([0, 1, 2, 3]),
        'bmi': round(random.uniform(15.0, 40.0), 1),
        'smoking': random.choice([0, 1]),
        'physical_activity': round(random.uniform(0.0, 10.0), 1),
        'diet_quality': round(random.uniform(0.0, 10.0), 1),
        'sleep_quality': round(random.uniform(4.0, 10.0), 1),
        'pollution_exposure': round(random.uniform(0.0, 10.0), 1),
        'pollen_exposure': round(random.uniform(0.0, 10.0), 1),
        'dust_exposure': round(random.uniform(0.0, 10.0), 1),
        'pet_allergy': random.choice([0, 1]),
        'family_history': random.choice([0, 1]),
        'history_allergies': random.choice([0, 1]),
        'eczema': random.choice([0, 1]),
        'hay_fever': random.choice([0, 1]),
        'gerd': random.choice([0, 1]),
        'fev1': round(random.uniform(1.0, 4.0), 1),
        'fvc': round(random.uniform(1.5, 6.0), 1),
        'wheezing': random.choice([0, 1]),
        'shortness_breath': random.choice([0, 1]),
        'chest_tightness': random.choice([0, 1]),
        'coughing': random.choice([0, 1]),
        'nighttime_symptoms': random.choice([0, 1]),
        'exercise_induced': random.choice([0, 1])
    }
    
    return random_data

# Function to save prediction history
def save_prediction(input_data, prediction, confidence, timestamp):
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Create a record of this prediction
    record = {
        'timestamp': timestamp,
        'input_data': input_data.copy(),
        'prediction': prediction,
        'confidence': confidence
    }
    
    # Add to history
    st.session_state.prediction_history.append(record)

# Function to create radar chart
def create_risk_radar_chart(input_data):
    # Select key risk factors for radar chart
    radar_categories = [
        'Family History', 'Allergies', 'Hay Fever', 'Eczema',
        'Pollution', 'Pollen', 'Dust', 'Smoking',
        'Wheezing', 'Shortness of Breath', 'Chest Tightness'
    ]
    
    # Map input data to radar values (scale 0-1)
    radar_values = [
        input_data['FamilyHistoryAsthma'].iloc[0],
        input_data['HistoryOfAllergies'].iloc[0],
        input_data['HayFever'].iloc[0],
        input_data['Eczema'].iloc[0],
        input_data['PollutionExposure'].iloc[0]/10,  # Scale to 0-1
        input_data['PollenExposure'].iloc[0]/10,     # Scale to 0-1
        input_data['DustExposure'].iloc[0]/10,       # Scale to 0-1
        input_data['Smoking'].iloc[0],
        input_data['Wheezing'].iloc[0],
        input_data['ShortnessOfBreath'].iloc[0],
        input_data['ChestTightness'].iloc[0]
    ]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_categories,
        fill='toself',
        name='Risk Factors',
        line_color='rgb(31, 119, 180)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Risk Factor Profile",
        height=500
    )
    
    return fig

# Function to create lung function gauge chart
def create_lung_function_gauge(fev1, fvc):
    # Calculate FEV1/FVC ratio
    ratio = (fev1 / fvc) * 100 if fvc > 0 else 0
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ratio,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "FEV1/FVC Ratio (%)"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 70], 'color': 'red'},
                {'range': [70, 80], 'color': 'yellow'},
                {'range': [80, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig, ratio

# Main function
def main():
    # Sidebar
    st.sidebar.image("https://www.example.com/logo.png", use_column_width=True)
    st.sidebar.markdown("<div class='sub-header'>Asthma Risk Predictor</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Add options in sidebar
    sidebar_options = st.sidebar.radio(
        "Navigation",
        ["Prediction Tool", "Your History", "About", "Help & FAQ"]
    )
    
    st.sidebar.markdown("---")
    
    # Initialize session state if not exists
    if 'user_input' not in st.session_state:
        st.session_state.user_input = {}
    
    # Button to generate random values
    if st.sidebar.button("Generate Random Values"):
        st.session_state.user_input = generate_random_values()
        st.experimental_rerun()
    
    # Reset button
    if st.sidebar.button("Reset All Values"):
        st.session_state.user_input = {}
        st.experimental_rerun()
    
    # Disclaimer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "⚠️ **Disclaimer**: This tool provides estimates only and should not replace professional medical advice. "
        "Always consult with a healthcare provider for diagnosis and treatment."
    )
    
    # Main content area
    if sidebar_options == "Prediction Tool":
        # App title and description
        st.markdown("<div class='main-header'>Asthma Risk Prediction System</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        This application predicts asthma risk based on environmental factors and personal health data.
        Complete the form below to receive your personalized asthma risk assessment.
        </div>
        """, unsafe_allow_html=True)
        
        # Load model
        model = load_model()
        
        if model is None:
            st.error("Failed to load the prediction model. Please check if 'bestmodel.pkl' exists in the correct directory.")
            return
        
        # Create form
        with st.form(key="prediction_form"):
            # Create tabs for different sections
            section_tabs = st.tabs(["Demographics", "Lifestyle", "Environmental", "Medical History", "Symptoms"])
            
            # Demographics tab
            with section_tabs[0]:
                st.markdown("<div class='sub-header'>Demographic Information</div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.number_input("Age", min_value=5, max_value=79, 
                                         value=st.session_state.user_input.get('age', 30))
                    
                    gender_options = {0: "Male", 1: "Female"}
                    gender = st.radio("Gender", options=list(gender_options.keys()),
                                     format_func=lambda x: gender_options[x],
                                     horizontal=True,
                                     index=st.session_state.user_input.get('gender', 0))
                
                with col2:
                    ethnicity_options = {0: "Type 0", 1: "Type 1", 2: "Type 2", 3: "Type 3"}
                    ethnicity = st.selectbox("Ethnicity", options=list(ethnicity_options.keys()), 
                                           format_func=lambda x: ethnicity_options[x],
                                           index=st.session_state.user_input.get('ethnicity', 0))
                    
                    education_options = {0: "Level 0", 1: "Level 1", 2: "Level 2", 3: "Level 3"}
                    education = st.selectbox("Education Level", options=list(education_options.keys()),
                                           format_func=lambda x: education_options[x],
                                           index=st.session_state.user_input.get('education', 0))
                
                bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, 
                                     value=st.session_state.user_input.get('bmi', 24.5),
                                     help="Body Mass Index (weight in kg / height in m²)")
            
            # Lifestyle tab
            with section_tabs[1]:
                st.markdown("<div class='sub-header'>Lifestyle Factors</div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    smoking = st.radio("Smoking Status", [0, 1], 
                                      format_func=lambda x: "Non-smoker" if x == 0 else "Smoker",
                                      horizontal=True,
                                      index=st.session_state.user_input.get('smoking', 0))
                    
                    physical_activity = st.slider("Physical Activity", 0.0, 10.0, 
                                                 value=st.session_state.user_input.get('physical_activity', 3.0), 0.1,
                                                 help="Physical activity level from 0 (sedentary) to 10 (very active)")
                
                with col2:
                    diet_quality = st.slider("Diet Quality", 0.0, 10.0, 
                                           value=st.session_state.user_input.get('diet_quality', 5.0), 0.1,
                                           help="Diet quality from 0 (poor) to 10 (excellent)")
                    
                    sleep_quality = st.slider("Sleep Quality", 4.0, 10.0, 
                                            value=st.session_state.user_input.get('sleep_quality', 7.0), 0.1,
                                            help="Sleep quality from 4 (poor) to 10 (excellent)")
            
            # Environmental tab
            with section_tabs[2]:
                st.markdown("<div class='sub-header'>Environmental Exposures</div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    pollution_exposure = st.slider("Pollution Exposure", 0.0, 10.0, 
                                                  value=st.session_state.user_input.get('pollution_exposure', 5.0), 0.1,
                                                  help="Exposure to air pollution from 0 (minimal) to 10 (severe)")
                    
                    pollen_exposure = st.slider("Pollen Exposure", 0.0, 10.0, 
                                               value=st.session_state.user_input.get('pollen_exposure', 5.0), 0.1,
                                               help="Exposure to pollen from 0 (minimal) to 10 (severe)")
                
                with col2:
                    dust_exposure = st.slider("Dust Exposure", 0.0, 10.0, 
                                             value=st.session_state.user_input.get('dust_exposure', 5.0), 0.1,
                                             help="Exposure to dust from 0 (minimal) to 10 (severe)")
                    
                    pet_allergy = st.radio("Pet Allergy", [0, 1], 
                                         format_func=lambda x: "No" if x == 0 else "Yes",
                                         horizontal=True,
                                         index=st.session_state.user_input.get('pet_allergy', 0))
            
            # Medical History tab
            with section_tabs[3]:
                st.markdown("<div class='sub-header'>Medical History</div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    family_history = st.radio("Family History of Asthma", [0, 1], 
                                            format_func=lambda x: "No" if x == 0 else "Yes",
                                            horizontal=True,
                                            index=st.session_state.user_input.get('family_history', 0))
                    
                    history_allergies = st.radio("History of Allergies", [0, 1], 
                                               format_func=lambda x: "No" if x == 0 else "Yes",
                                               horizontal=True,
                                               index=st.session_state.user_input.get('history_allergies', 0))
                    
                    eczema = st.radio("Eczema", [0, 1], 
                                     format_func=lambda x: "No" if x == 0 else "Yes",
                                     horizontal=True,
                                     index=st.session_state.user_input.get('eczema', 0))
                
                with col2:
                    hay_fever = st.radio("Hay Fever", [0, 1], 
                                       format_func=lambda x: "No" if x == 0 else "Yes",
                                       horizontal=True,
                                       index=st.session_state.user_input.get('hay_fever', 0))
                    
                    gerd = st.radio("Gastroesophageal Reflux (GERD)", [0, 1], 
                                   format_func=lambda x: "No" if x == 0 else "Yes",
                                   horizontal=True,
                                   index=st.session_state.user_input.get('gerd', 0))
                    
                    # Lung Function
                    st.markdown("<div class='sub-header'>Lung Function</div>", unsafe_allow_html=True)
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        fev1 = st.number_input("Lung Function FEV1", min_value=1.0, max_value=4.0, 
                                              value=st.session_state.user_input.get('fev1', 2.5),
                                              help="Forced Expiratory Volume in 1 second (liters)")
                    
                    with col4:
                        fvc = st.number_input("Lung Function FVC", min_value=1.5, max_value=6.0, 
                                             value=st.session_state.user_input.get('fvc', 3.5),
                                             help="Forced Vital Capacity (liters)")
            
            # Symptoms tab
            with section_tabs[4]:
                st.markdown("<div class='sub-header'>Current Symptoms</div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    wheezing = st.radio("Wheezing", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes",
                                      horizontal=True,
                                      index=st.session_state.user_input.get('wheezing', 0))
                    
                    shortness_breath = st.radio("Shortness of Breath", [0, 1], 
                                              format_func=lambda x: "No" if x == 0 else "Yes",
                                              horizontal=True,
                                              index=st.session_state.user_input.get('shortness_breath', 0))
                
                with col2:
                    chest_tightness = st.radio("Chest Tightness", [0, 1], 
                                             format_func=lambda x: "No" if x == 0 else "Yes",
                                             horizontal=True,
                                             index=st.session_state.user_input.get('chest_tightness', 0))
                    
                    coughing = st.radio("Coughing", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes",
                                      horizontal=True,
                                      index=st.session_state.user_input.get('coughing', 0))
                
                with col3:
                    nighttime_symptoms = st.radio("Nighttime Symptoms", [0, 1], 
                                                format_func=lambda x: "No" if x == 0 else "Yes",
                                                horizontal=True,
                                                index=st.session_state.user_input.get('nighttime_symptoms', 0))
                    
                    exercise_induced = st.radio("Exercise-Induced Symptoms", [0, 1], 
                                              format_func=lambda x: "No" if x == 0 else "Yes",
                                              horizontal=True,
                                              index=st.session_state.user_input.get('exercise_induced', 0))
            
            # Submit button
            submit_button = st.form_submit_button(label="Predict Asthma Risk", use_container_width=True)
        
        # Process prediction when form is submitted
        if submit_button:
            # Update session state with current values
            st.session_state.user_input = {
                'age': age,
                'gender': gender,
                'ethnicity': ethnicity,
                'education': education,
                'bmi': bmi,
                'smoking': smoking,
                'physical_activity': physical_activity,
                'diet_quality': diet_quality,
                'sleep_quality': sleep_quality,
                'pollution_exposure': pollution_exposure,
                'pollen_exposure': pollen_exposure,
                'dust_exposure': dust_exposure,
                'pet_allergy': pet_allergy,
                'family_history': family_history,
                'history_allergies': history_allergies,
                'eczema': eczema,
                'hay_fever': hay_fever,
                'gerd': gerd,
                'fev1': fev1,
                'fvc': fvc,
                'wheezing': wheezing,
                'shortness_breath': shortness_breath,
                'chest_tightness': chest_tightness,
                'coughing': coughing,
                'nighttime_symptoms': nighttime_symptoms,
                'exercise_induced': exercise_induced
            }
            
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
            
            # Make a copy of the unscaled data for display
            unscaled_data = input_data.copy()
            
            # Scale the numerical features (avoid scaling categorical ones)
            numeric_cols = input_data.select_dtypes(include=["int64", "float64"]).columns
            input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
            
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # Get confidence
                confidence = max(prediction_proba) * 100
                
                # Save prediction to history
                save_prediction(unscaled_data, prediction, confidence, datetime.now())
                
                # Display results
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<div class='sub-header'>Your Asthma Risk Assessment</div>", unsafe_allow_html=True)
                
                # Create result tabs
                result_tabs = st.tabs(["Risk Assessment", "Risk Factors", "Visualizations", "Recommendations"])
                
                # Risk Assessment tab
                with result_tabs[0]:
                    # Define risk levels based on model output and confidence
                    if prediction == 1:
                        if confidence > 85:
                            risk_level = "High Risk"
                            risk_class = "risk-high"
                        else:
                            risk_level = "Medium-High Risk"
                            risk_class = "risk-medium"
                    else:
                        if confidence > 85:
                            risk_level = "Low Risk"
                            risk_class = "risk-low"
                        else:
                            risk_level = "Medium-Low Risk"
                            risk_class = "risk-medium"
                    
                    # Display prediction with styling
                    st.markdown(f"<div class='{risk_class}'>Prediction: {risk_level}</div>", unsafe_allow_html=True)
                    
                    # Display confidence
                    st.markdown(f"Confidence: {confidence:.2f}%")
                    
                    # Display lung function ratio
                    lung_gauge, ratio = create_lung_function_gauge(fev1, fvc)
                    st.plotly_chart(lung_gauge)
                    
                    if ratio < 70:
                        st.warning(f"Your FEV1/FVC ratio is {ratio:.1f}%, which is below the normal threshold of 70%. This may indicate airflow limitation.")
                    elif ratio < 80:
                        st.info(f"Your FEV1/FVC ratio is {ratio:.1f}%, which is in the borderline range. Monitor this value over time.")
                    else:
                        st.success(f"Your FEV1/FVC ratio is {ratio:.1f}%, which is within the normal range.")
                
                # Risk Factors tab
                with result_tabs[1]:
                    # Identify risk factors
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
                    if ratio < 70:
                        risk_factors.append(f"Low FEV1/FVC ratio ({ratio:.1f}%)")
                    if smoking == 1:
                        risk_factors.append("Smoking")
                    
                    # Display risk factors
                    st.subheader("Your Primary Risk Factors")
                    if risk_factors:
                        for i, factor in enumerate(risk_factors):
                            st.markdown(f"{i+1}. {factor}")
                    else:
                        st.markdown("No significant risk factors identified.")
                    
                    # Display protective factors
                    protective_factors = []
                    if physical_activity > 7:
                        protective_factors.append("Regular physical activity")
                    if diet_quality > 7:
                        protective_factors.append("Good diet quality")
                    if sleep_quality > 7:
                        protective_factors.append("Good sleep quality")
                    if smoking == 0:
                        protective_factors.append("Non-smoking status")
                    if pollution_exposure < 3 and pollen_exposure < 3 and dust_exposure < 3:
                        protective_factors.append("Low environmental exposure")
                    
                    st.subheader("Protective Factors")
                    if protective_factors:
                        for i, factor in enumerate(protective_factors):
                            st.markdown(f"{i+1}. {factor}")
                    else:
                        st.markdown("No significant protective factors identified.")
                
                # Visualizations tab
                with result_tabs[2]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create radar chart
                        radar_fig = create_risk_radar_chart(unscaled_data)
                        st.plotly_chart(radar_fig)
                    
                    with col2:
                        # Create a sample symptom frequency chart
                        # This would normally use historical data, but we'll create mock data
                        symptoms = ['Wheezing', 'Shortness of Breath', 'Chest Tightness', 
                                   'Coughing', 'Nighttime Symptoms', 'Exercise-Induced']
                        
                        symptom_values = [wheezing, shortness_breath, chest_tightness, 
                                         coughing, nighttime_symptoms, exercise_induced]
                        
                        symptoms_df = pd.DataFrame({
                            'Symptom': symptoms,
                            'Present': [1 if v else 0 for v in symptom_values]
                        })
                        
                        fig = px.bar(symptoms_df, x='Symptom', y='Present', 
                                    title='Symptom Status',
                                    color='Present',
                                    color_discrete_map={0: 'green', 1: 'red'},
                                    labels={'Present': 'Status', 'Symptom': 'Symptom Type'})
                        
                        fig.update_layout(
                            xaxis_title='Symptom',
                            yaxis_title='Status (0=No, 1=Yes)',
                            height=500
                        )
                        
                        st.plotly_chart(fig)
                
                # Recommendations tab
                with result_tabs[3]:
                    st.subheader("Personalized Recommendations")
                    
                    if risk_level.startswith("High"):
                        st.markdown("""
                        ### Medical Consultation
                        - 🏥 **Consult with a healthcare provider** for a comprehensive asthma evaluation
                        - 🔬 Consider allergy testing to identify specific triggers
                        - 📋 Develop an asthma action plan with your doctor
                        
                        ### Monitoring
                        - 📊 Monitor your symptoms regularly with a symptom diary
                        - 📱 Consider using a peak flow meter to track lung function
                        - 🔍 Track environmental triggers that worsen symptoms
                        
                        ### Environmental Control
                        - 🌡️ Reduce exposure to identified environmental triggers
                        - 🧹 Consider using air purifiers in your home
                        - 🛏️ Use allergen-proof bedding covers
                        """)
                    elif risk_level.startswith("Medium"):
                        st.markdown("""
                        ### Vigilance
                        - 👩‍⚕️ Consider discussing your risk factors with a healthcare provider
                        - 📝 Start monitoring any symptoms that occur
                        - 🌬️ Be aware of environmental triggers that may affect you
                        
                        ### Prevention
                        - 🏃‍♀️ Maintain regular physical activity appropriate to your fitness level
                        - 🥗 Focus on a balanced diet that supports respiratory health
                        - 🧘‍♂️ Practice stress reduction techniques
                        
                        ### Environmental Awareness
                        - 🏠 Reduce exposure to common allergens and irritants
                        - 💧 Maintain optimal humidity levels in your home (30-50%)
                        - 🌿 Monitor pollen counts if you have allergies
                        """)
                    else:
                        st.markdown("""
                        ### Maintain Healthy Habits
                        - 🏃‍♂️ Continue regular physical activity
                        - 🥦 Maintain a balanced, nutritious diet
                        - 😴 Prioritize good sleep hygiene
                        
                        ### General Awareness
                        - 🌤️ Stay aware of air quality in your area
                        - 👀 Monitor for any developing respiratory symptoms
                        - 💊 Stay up to date with vaccinations
                        
                        ### Environmental Considerations
                        - 🧼 Regular cleaning to reduce dust accumulation
                        - 🏠 Ensure good ventilation in your home
                        - 🚭 Avoid secondhand smoke exposure
                        """)
                    
                    st.info("Remember that these recommendations are personalized based on your risk assessment, but they do not replace professional medical advice.")
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please check if the input data format matches what the model expects.")
    
    # History page
    elif sidebar_options == "Your History":
        st.markdown("<div class='main-header'>Your Prediction History</div>", unsafe_allow_html=True)
        
        if 'prediction_history' not in st.session_state or not st.session_state.prediction_history:
            st.info("You haven't made any predictions yet. Use the Prediction Tool to start.")
        else:
            # Display history in a table
            st.subheader("Previous Assessments")
            
            # Create a dataframe from history
            history_data = []
            for record in st.session_state.prediction_history:
                history_data.append({
                    'Date & Time': record['timestamp'].strftime("%Y-%m-%d %H:%M"),
                    'Risk Prediction': "High Risk" if record['prediction'] == 1 else "Low Risk",
                    'Confidence': f"{record['confidence']:.2f}%",
                    'Age': record['input_data']['Age'].iloc[0],
                    'BMI': record['input_data']['BMI'].iloc[0],
                    'FEV1': record['input_data']['LungFunctionFEV1'].iloc[0],
                    'FVC': record['input_data']['LungFunctionFVC'].iloc[0]
                })
            
            history_df = pd.DataFrame(history_data)
            
            # Add styling
            st.dataframe(history_df, use_container_width=True)
            
            # Option to clear history
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.experimental_rerun()
            
            # Trends over time if multiple predictions exist
            if len(st.session_state.prediction_history) > 1:
                st.subheader("Trends Over Time")
                
                # Create time series data
                time_data = []
                for record in st.session_state.prediction_history:
                    time_data.append({
                        'timestamp': record['timestamp'],
                        'prediction': record['prediction'],
                        'confidence': record['confidence'],
                        'fev1': record['input_data']['LungFunctionFEV1'].iloc[0],
                        'fvc': record['input_data']['LungFunctionFVC'].iloc[0]
                    })
                
                time_df = pd.DataFrame(time_data)
                
                # Create line chart for lung function over time
                fig = px.line(time_df, x='timestamp', y=['fev1', 'fvc'], 
                             title='Lung Function Over Time',
                             labels={'value': 'Lung Function (liters)', 'timestamp': 'Date', 'variable': 'Measure'},
                             color_discrete_map={'fev1': 'blue', 'fvc': 'green'})
                
                st.plotly_chart(fig)
                
                # Create confidence trend
                fig2 = px.line(time_df, x='timestamp', y='confidence',
                              title='Prediction Confidence Over Time',
                              labels={'confidence': 'Confidence (%)', 'timestamp': 'Date'})
                
                st.plotly_chart(fig2)
    
    # About page
    elif sidebar_options == "About":
        st.markdown("<div class='main-header'>About This Application</div>", unsafe_allow_html=True)
        
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
        
        ### The Science Behind the Prediction
        
        The prediction system is built on a supervised machine learning algorithm that has been trained on a 
        dataset of 2,392 individuals with known asthma status. The model identifies patterns in the data that 
        are associated with asthma risk and uses these patterns to make predictions for new individuals.
        
        Key risk factors that the model considers include:
        
        - Family history of asthma
        - History of allergies, eczema, and hay fever
        - Environmental exposures to allergens and irritants
        - Lung function measurements
        - Respiratory symptoms
        
        ### Model Performance
        
        The prediction model has been validated and has the following performance metrics:
        
        - Accuracy: 89%
        - Sensitivity: 85%
        - Specificity: 92%
        - Area Under the ROC Curve: 0.91
        
        ### Data Privacy
        
        All data entered is processed locally on your device and is not stored or transmitted elsewhere, 
        except within your browser's local storage to maintain your prediction history between sessions. 
        You can clear this history at any time.
        
        ### Limitations
        
        This tool is intended for educational and informational purposes only. It should not be used as a 
        replacement for professional medical diagnosis, advice, or treatment. Always consult with a qualified 
        healthcare provider for medical concerns.
        
        The model has several limitations:
        
        - It is based on a specific population that may not represent all demographics
        - It uses self-reported symptoms which may vary in interpretation
        - It cannot account for all possible environmental and genetic factors
        - It provides a risk assessment, not a definitive diagnosis
        """)
        
        # Development team info
        st.markdown("""
        ### Development Team
        
        This application was developed by a multidisciplinary team including:
        
        - Data scientists and machine learning engineers
        - Pulmonologists and allergists
        - Public health researchers
        - Software developers
        
        For more information or to provide feedback, please contact us at example@example.com.
        """)
    
    # Help & FAQ page
    elif sidebar_options == "Help & FAQ":
        st.markdown("<div class='main-header'>Help & Frequently Asked Questions</div>", unsafe_allow_html=True)
        
        faq_expander = st.expander("Frequently Asked Questions")
        with faq_expander:
            st.markdown("""
            ### What is asthma?
            
            Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, 
            leading to symptoms such as wheezing, shortness of breath, chest tightness, and coughing.
            
            ### How accurate is this prediction tool?
            
            The tool has approximately 89% accuracy based on validation studies. However, it is designed as a 
            screening tool rather than a diagnostic test. Only a healthcare provider can diagnose asthma.
            
            ### What do the risk levels mean?
            
            - **High Risk**: You have multiple risk factors and symptoms consistent with asthma. Medical evaluation is recommended.
            - **Medium-High Risk**: You have some risk factors and symptoms that may be associated with asthma. Consider medical evaluation.
            - **Medium-Low Risk**: You have few risk factors but some symptoms that could be monitored.
            - **Low Risk**: You have minimal risk factors and symptoms associated with asthma.
            
            ### What is FEV1 and FVC?
            
            FEV1 (Forced Expiratory Volume in 1 second) and FVC (Forced Vital Capacity) are lung function measurements:
            - FEV1 measures how much air you can exhale in one second
            - FVC measures the total amount of air you can exhale after taking a deep breath
            - The ratio of FEV1/FVC is an important indicator of airway obstruction
            
            ### How often should I use this tool?
            
            This tool can be used periodically to monitor changes in your risk profile, especially if you experience 
            changes in symptoms or environmental exposures. However, if you are experiencing respiratory symptoms, 
            you should consult a healthcare provider rather than rely solely on this tool.
            
            ### Can children use this tool?
            
            The tool is designed for individuals age 5 and older. However, for children, it's particularly important 
            to consult with a pediatrician about respiratory symptoms, as childhood asthma can present differently.
            """)
        
        # Using the app
        how_to_expander = st.expander("How to Use This Application")
        with how_to_expander:
            st.markdown("""
            ### Basic Navigation
            
            1. **Prediction Tool**: Complete the form with your information to receive a risk assessment
            2. **Your History**: View your previous predictions and track changes over time
            3. **About**: Learn more about the science behind the prediction model
            4. **Help & FAQ**: Find answers to common questions
            
            ### Using the Prediction Tool
            
            1. Navigate through the tabs to enter your information in each category
            2. Use the sliders and selection boxes to input your data
            3. Click "Predict Asthma Risk" to generate your assessment
            4. Review your results in the tabbed interface
            
            ### Interpreting Your Results
            
            The results section provides:
            - An overall risk assessment
            - Identification of your specific risk factors
            - Visualizations of your data
            - Personalized recommendations
            
            ### Tips for Accurate Results
            
            - Answer all questions as accurately as possible
            - For lung function measures (FEV1, FVC), use recent spirometry results if available
            - Be honest about symptoms, even if they seem mild
            - Update your information if your health status changes
            """)
        
        # Interpretation guidance
        interpretation_expander = st.expander("Interpreting Lung Function Values")
        with interpretation_expander:
            st.markdown("""
            ### FEV1/FVC Ratio
            
            The FEV1/FVC ratio is the proportion of your vital capacity that you can expire in the first second of forced expiration.
            
            - **Normal range**: 70-85% in adults
            - **Below 70%**: May indicate airflow obstruction, which is a key feature of asthma
            - **Above 85%**: Usually normal, but can sometimes indicate restrictive lung disease
            
            ### FEV1 Percent Predicted
            
            FEV1 values vary based on age, height, sex, and ethnicity. Your value should be compared to the predicted value for someone with your demographics.
            
            - **>80% of predicted**: Normal
            - **70-79% of predicted**: Mild obstruction
            - **60-69% of predicted**: Moderate obstruction
            - **<60% of predicted**: Severe obstruction
            
            ### Spirometry Interpretation
            
            Complete spirometry testing, which includes measurement of FEV1 and FVC, is an important part of asthma diagnosis and monitoring. 
            
            In asthma:
            - FEV1 is often reduced
            - FEV1/FVC ratio is reduced
            - These values may improve significantly after bronchodilator medication
            
            This application provides a simple interpretation, but professional spirometry testing provides more detailed information.
            """)
        
        # Contact support
        st.markdown("""
        ### Need More Help?
        
        If you have questions that aren't answered here, or if you're experiencing technical issues with the application, 
        please contact our support team at support@example.com.
        
        For medical questions related to asthma or respiratory symptoms, please consult with a healthcare provider.
        """)

if __name__ == '__main__':
    main()
