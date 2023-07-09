from flask import Flask, request,jsonify
import pandas as pd
import holidays
import joblib

app = Flask(__name__)
 
import pickle
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file, fix_imports=True)

# Define the list of days of the week
DayOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Define the list of seasons
Season = ['Winter', 'Spring', 'Summer', 'Autumn']

# Define the list of public holidays
PublicHoliday = ['False', 'True']

def extract_features(date_str):
    sample = pd.DataFrame({'Date': [date_str]})
    sample['Date'] = pd.to_datetime(sample['Date'])

    # Get the holiday calendar for the specified country or region
    holiday_calendar = holidays.CountryHoliday("UK")

    # Add the new features to the original dataset
    sample['DayOfWeek'] = sample['Date'].dt.day_name()
    sample['Season'] = sample['Date'].apply(get_season)
    sample['PublicHoliday'] = sample['Date'].apply(lambda x: str(x.date()) in holiday_calendar)
    sample['Day'] = sample['Date'].dt.day.astype(int)
    sample['Year'] = sample['Date'].dt.year.astype(int)
    sample['Month'] = sample['Date'].dt.month.astype(int)

    # Convert the DayOfWeek to numeric representation
    sample['DayOfWeek'] = sample['DayOfWeek'].apply(lambda x: DayOfWeek.index(x) + 1)

    # Convert the Season to numeric representation
    sample['Season'] = sample['Season'].apply(lambda x: Season.index(x) + 1)

    # Convert the PublicHoliday to numeric representation
    sample['PublicHoliday'] = sample['PublicHoliday'].apply(lambda x: PublicHoliday.index(str(x)))

    return sample

def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'
 
@app.route('/predict', methods=['GET'])
def predict():
    # Extract the date from the query parameter
    date_str = request.args.get('date')

    # Extract the features
    features = extract_features(date_str)

    # Prepare the input data for prediction
    input_data = features[['DayOfWeek', 'Season', 'PublicHoliday', 'Day', 'Year', 'Month']]

    # Make predictions using the model
    prediction = int(model.predict(input_data))

    # Return the prediction as a JSON response
    prediction_min = prediction - 10
    prediction_max = prediction + 10

    return jsonify({'prediction_min': prediction_min, 'prediction_max': prediction_max})


if __name__ == '__main__':
    app.run()
