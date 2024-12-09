import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained Gradient Boosting model and training data (for resampling)
with open('gradient_boosting_model3.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('X_train.pkl', 'rb') as x_file:
    X_train = pickle.load(x_file)

with open('y_train.pkl', 'rb') as y_file:
    y_train = pickle.load(y_file)

# Define the column names (feature names) used during training
feature_names_from_training = [
    'Sub-Area', 'Property Type', 'Property Area in Sq. Ft.', 'Company Name',
    'TownShip Name/ Society Name', 'ClubHouse', 'School / University in Township ',
    'Hospital in TownShip', 'Mall in TownShip', 'Park / Jogging track', 
    'Swimming Pool', 'Gym'
]

# Home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/", methods=["POST"])
def predict():
    """
    Handle form submission, make predictions, and return results.
    """
    try:
        # Extract values from the form
        sub_area = int(request.form['sub_area'])
        property_type = int(request.form['property_type'])
        property_area = float(request.form['property_area'])
        company_name = int(request.form['company_name'])
        township_name = int(request.form['township_name'])
        clubhouse = int(request.form['clubhouse'])
        school = int(request.form['school'])
        hospital = int(request.form['hospital'])
        mall = int(request.form['mall'])
        park = int(request.form['park'])
        swimming_pool = int(request.form['swimming_pool'])
        gym = int(request.form['gym'])

        feature_names_from_training = X_train.columns.tolist()
        # Prepare the input data as a pandas DataFrame with the correct feature names
        input_data = pd.DataFrame(
            [[sub_area, property_type, property_area, company_name, township_name, 
              clubhouse, school, hospital, mall, park, swimming_pool, gym]],
            columns=feature_names_from_training  # Match the feature names exactly
        )

        # Bootstrap procedure for prediction intervals
        n_iterations = 10
        predictions = np.zeros(n_iterations)
        
        for i in range(n_iterations):
            # Resample training data with replacement
            X_train_resample, y_train_resample = resample(X_train, y_train, n_samples=len(X_train), random_state=i)
            
            # Fit the model on resampled data
            model.fit(X_train_resample, y_train_resample)
            
            # Predict on the input data
            predictions[i] = model.predict(input_data)[0]
        
        # Calculate the point estimate (mean prediction)
        point_estimate = np.mean(predictions)
        
        # Calculate the 0th and 100th percentiles for prediction intervals (lower and upper bounds)
        lower_bound = np.percentile(predictions, 0)
        upper_bound = np.percentile(predictions, 100)

        print(f"Predictions: {predictions}")
        print(f"Point estimate: {point_estimate}")
        print(f"Lower bound: {lower_bound}")
        print(f"Upper bound: {upper_bound}")

        # Render the template with the point estimate and prediction intervals
        return render_template(
            "index.html",
            point_estimate=f"The predicted price of this house is ${point_estimate:.2f}",
            lower_bound=f"The price usually lies between ${lower_bound:.2f} and ${upper_bound:.2f}",
        )
    except Exception as e:
        # Handle exceptions and display an error message
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
