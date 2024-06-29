import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved XGBoost model and scaler
model = joblib.load('best_model_XGBoost.joblib')
scaler = joblib.load('scaler.joblib')

# Load the dataset (replace '122.csv' with your actual dataset file name)
df = pd.read_csv('122.csv')
X = df.drop(columns=['churn'])
y = df['churn']
X_scaled = scaler.transform(X)

# Dropdown mappings and feature descriptions
dropdown_mappings = {
    'new_cell': {'': None, 'U': 1, 'Y': 2, 'N': 0},
    'prizm_social_one': {'': None, 'S': 2, 'U': 4, 'C': 0, 'T': 3, 'R': 1},
    'area': {'': None, 'NEW YORK CITY AREA': 11, 'LOS ANGELES AREA': 8, 'DC/MARYLAND/VIRGINIA AREA': 5,
             'MIDWEST AREA': 9, 'SOUTHWEST AREA': 17, 'ATLANTIC SOUTH AREA': 0,
             'CALIFORNIA NORTH AREA': 1, 'NEW ENGLAND AREA': 10, 'DALLAS AREA': 4,
             'CHICAGO AREA': 3, 'OHIO AREA': 14, 'GREAT LAKES AREA': 6,
             'NORTH FLORIDA AREA': 12, 'NORTHWEST/ROCKY MOUNTAIN AREA': 13,
             'HOUSTON AREA': 7, 'CENTRAL/SOUTH TEXAS AREA': 2, 'SOUTH FLORIDA AREA': 16,
             'TENNESSEE AREA': 18, 'PHILADELPHIA AREA': 15},
    'dualband': {'': None, 'Y': 3, 'N': 0, 'T': 1, 'U': 2},
    'refurb_new': {'': None, 'N': 0, 'R': 1},
    'hnd_webcap': {'': None, 'WCMB': 2, 'WC': 1, 'UNKW': 0},
    'marital': {'': None, 'U': 4, 'M': 2, 'S': 3, 'B': 1, 'A': 0},
    'ethnic': {'': None, 'N': 9, 'H': 5, 'S': 13, 'U': 14, 'G': 4, 'O': 10, 'Z': 16, 'I': 6, 'J': 7,
               'F': 3, 'B': 0, 'R': 12, 'D': 2, 'P': 11, 'C': 1, 'M': 8, 'X': 15},
    'creditcd': {'': None, 'Y': 1, 'N': 0}
}

feature_descriptions = {
    'creditcd': 'Credit card indicator',
    'dualband': 'Dualband',
    'ethnic': 'Ethnicity roll-up code',
    'marital': 'Marital Status',
    'models': 'Number of models issued',
    'months': 'Total number of months in service',
    'new_cell': 'New cell phone user',
    'prizm_social_one': 'Social group letter only',
    'refurb_new': 'Handset: refurbished or new',
    'area': 'Geographic area',
    'hnd_webcap': 'Handset web capability',
    'total_children_0_U': 'Total children',
    'total_truck': 'Total vehicles',
    'uniqsubs': 'Number of unique subscribers in the household',
    'actvsubs': 'Number of active subscribers in household',
    'totcalls': 'Total number of calls over the life of the customer',
    'eqpdays': 'Number of days (age) of current equipment',
    'totmou': 'Total minutes of use over the life of the customer',
    'totrev': 'Total revenue',
    'avgrev': 'Average monthly revenue over the life of the customer',
    'avgmou': 'Average monthly minutes of use over the life of the customer',
    'avgqty': 'Average monthly number of calls over the life of the customer'
}

ranges = {
    'months': [6, 61],
    'uniqsubs': [1, 196],
    'actvsubs': [0, 53],
    'totcalls': [0, 50636],
    'totmou': [0.0, 106760.3067],
    'totrev': [3.65, 14231.32],
    'avgrev': [0.48, 709.97],
    'avgmou': [0.0, 2900.82],
    'avgqty': [0.0, 1488.16],
    'eqpdays': [-5.0, 1823.0],
    'total_truck': [0.0, 2.0],
    'total_children_0_U': [0, 5]
}

# Streamlit app
st.title("Churn Prediction Form")

# Page navigation
if 'page' not in st.session_state:
    st.session_state.page = 1

# Page 1: Dropdown inputs
if st.session_state.page == 1:
    with st.form("churn_form"):
        cols = st.columns(3)
        dropdown_inputs = {}
        for i, (feature, options) in enumerate(dropdown_mappings.items()):
            with cols[i % 3]:
                dropdown_inputs[feature] = st.selectbox(f"{feature_descriptions.get(feature, feature)}",
                                                        list(options.keys()), index=0, key=feature,
                                                        help=feature_descriptions.get(feature, feature))
        if st.form_submit_button("Next"):
            # Convert dropdown values to their corresponding numerical values
            input_data = {key: dropdown_mappings[key].get(value, 0) for key, value in dropdown_inputs.items()}
            
            # Check for invalid selections
            invalid_selection = any(value is None for value in input_data.values())
            if invalid_selection:
                st.error("Invalid selection detected. Please make sure all fields are filled out correctly.")
            else:
                st.session_state.input_data = input_data
                st.session_state.page = 2

# Page 2: Numeric inputs
if st.session_state.page == 2:
    with st.form("page_2_form"):
        cols = st.columns(3)
        other_inputs = {}
        for i, feature in enumerate(['months', 'uniqsubs', 'actvsubs', 'totcalls', 'totmou', 'totrev', 'avgrev', 'avgmou', 'avgqty', 'eqpdays', 'total_truck', 'total_children_0_U']):
            min_val, max_val = ranges[feature]
            with cols[i % 3]:
                other_inputs[feature] = st.number_input(
                    f"{feature_descriptions.get(feature, feature)}",
                    min_value=min_val, max_value=max_val, key=feature,
                    help=f"Range: {min_val} - {max_val}", value=min_val)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.form_submit_button("Back"):
                st.session_state.page = 1
        with col2:
            if st.form_submit_button("Submit"):
                # Process the form data and make predictions
                input_data = st.session_state.input_data
                input_data.update(other_inputs)
                
                # Ensure all features expected by the model are included in the input data
                required_features = list(X.columns)
                for feature in required_features:
                    if feature not in input_data:
                        input_data[feature] = 0  # Default value if the feature is not provided
                
                # Convert input_data to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Reorder columns to match the order used during training
                input_df = input_df[required_features]
                
                # Standardize the input data
                input_scaled = scaler.transform(input_df)
                
                # Make predictions
                predictions = model.predict(input_scaled)
                prediction_probabilities = model.predict_proba(input_scaled)
                
                # Prepare the results
                results = {'Predicted': predictions[0]}
                for i in range(prediction_probabilities.shape[1]):
                    results[f'Probability_Class_{i}'] = prediction_probabilities[0, i]
                
                # Display the results
                st.header("Prediction Results")
                if results['Predicted'] == 0:
                    st.write("### <span style='color:green'>Not Churned</span>", unsafe_allow_html=True)
                else:
                    st.write("### <span style='color:red'>Churned</span>", unsafe_allow_html=True)
                
                st.write("#### Form Data:")
                for key, value in input_data.items():
                    st.write(f"**{feature_descriptions.get(key, key)}**: {value}")

