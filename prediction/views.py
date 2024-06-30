from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import joblib

# Load the saved XGBoost model and scaler
model = joblib.load('best_model_XGBoost.joblib')
scaler = joblib.load('scaler.joblib')

# Load the dataset
df = pd.read_csv('122.csv')
X = df.drop(columns=['churn'])
y = df['churn']
X_scaled = scaler.transform(X)

# Dropdown mappings
dropdown_mappings = {
    'new_cell': {'U': 1, 'Y': 2, 'N': 0},
    'prizm_social_one': {'S': 2, 'U': 4, 'C': 0, 'T': 3, 'R': 1},
    'area': {'NEW YORK CITY AREA': 11, 'LOS ANGELES AREA': 8, 'DC/MARYLAND/VIRGINIA AREA': 5,
             'MIDWEST AREA': 9, 'SOUTHWEST AREA': 17, 'ATLANTIC SOUTH AREA': 0,
             'CALIFORNIA NORTH AREA': 1, 'NEW ENGLAND AREA': 10, 'DALLAS AREA': 4,
             'CHICAGO AREA': 3, 'OHIO AREA': 14, 'GREAT LAKES AREA': 6,
             'NORTH FLORIDA AREA': 12, 'NORTHWEST/ROCKY MOUNTAIN AREA': 13,
             'HOUSTON AREA': 7, 'CENTRAL/SOUTH TEXAS AREA': 2, 'SOUTH FLORIDA AREA': 16,
             'TENNESSEE AREA': 18, 'PHILADELPHIA AREA': 15},
    'dualband': {'Y': 3, 'N': 0, 'T': 1, 'U': 2},
    'refurb_new': {'N': 0, 'R': 1},
    'hnd_webcap': {'WCMB': 2, 'WC': 1, 'UNKW': 0},
    'marital': {'U': 4, 'M': 2, 'S': 3, 'B': 1, 'A': 0},
    'ethnic': {'N': 9, 'H': 5, 'S': 13, 'U': 14, 'G': 4, 'O': 10, 'Z': 16, 'I': 6, 'J': 7,
               'F': 3, 'B': 0, 'R': 12, 'D': 2, 'P': 11, 'C': 1, 'M': 8, 'X': 15},
    'creditcd': {'Y': 1, 'N': 0}
}

# Feature descriptions
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
    'area': 'Geogrpahic area',
    'hnd_webcap': 'Handset web capability',
    'total_children_0_U': 'Total children',
    'total_truck': 'Total vehicles',
    'months': 'Total number of months in service',
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

def form(request):
    return render(request, 'form.html', {'dropdown_mappings': dropdown_mappings, 'feature_descriptions': feature_descriptions})

def predict(request):
    if request.method == 'POST':
        form_data = request.POST.dict()
        input_data = {key: dropdown_mappings[key][form_data[key]] for key in dropdown_mappings}
        
        total_children_0_U = int(form_data['total_children_0_U'])
        total_children_1_Y = 5 - total_children_0_U
        
        input_data['total_children_0_U'] = total_children_0_U
        input_data['total_children_1_Y'] = total_children_1_Y

        other_features = ['months', 'uniqsubs', 'actvsubs', 'totcalls', 'totmou', 'totrev', 'avgrev',
                          'avgmou', 'avgqty', 'eqpdays', 'total_truck']
        for feature in other_features:
            input_data[feature] = float(form_data[feature])
        
        closest_row = df.loc[((df['months'] - input_data['months']).abs() + 
                              (df['uniqsubs'] - input_data['uniqsubs']).abs() + 
                              (df['actvsubs'] - input_data['actvsubs']).abs() + 
                              (df['totcalls'] - input_data['totcalls']).abs() + 
                              (df['totmou'] - input_data['totmou']).abs() + 
                              (df['totrev'] - input_data['totrev']).abs() + 
                              (df['avgrev'] - input_data['avgrev']).abs() + 
                              (df['avgmou'] - input_data['avgmou']).abs() + 
                              (df['avgqty'] - input_data['avgqty']).abs() + 
                              (df['eqpdays'] - input_data['eqpdays']).abs() + 
                              (df['total_truck'] - input_data['total_truck']).abs()).idxmin()]
        
        third_set_features = ['roam_Mean', 'ovrrev_Mean', 'vceovr_Mean', 'ovrmou_Mean', 'hnd_price', 
                              'phones', 'models', 'total_adj_usage', 'total_adj_calls', 'total_adj_revenue', 
                              'avg_monthly_usage', 'avg_monthly_calls', 'avg_monthly_revenue', 'crclscod']
        for feature in third_set_features:
            input_data[feature] = closest_row[feature]
        
        required_features = list(X.columns)
        for feature in required_features:
            if feature not in input_data:
                input_data[feature] = 0

        input_df = pd.DataFrame([input_data])

        input_df = input_df[required_features]

        input_scaled = scaler.transform(input_df)

        predictions = model.predict(input_scaled)
        prediction_probabilities = model.predict_proba(input_scaled)

        results = {'Predicted': predictions[0]}
        for i in range(prediction_probabilities.shape[1]):
            results[f'Probability_Class_{i}'] = prediction_probabilities[0, i]

        return render(request, 'results.html', {'results': results, 'form_data': form_data, 'feature_descriptions': feature_descriptions})

    return JsonResponse({'error': 'Invalid request method'}, status=405)
