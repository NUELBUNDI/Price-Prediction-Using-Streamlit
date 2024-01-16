import pandas as pd
import joblib
import  numpy as np



def fuel_type_(value_picked):
    fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other']
    result_dict = {}

    for fuel in fuel_types:
        result_dict[fuel] = 1 if fuel == value_picked else 0

    return result_dict


def transmission_(value_picked):
    transmission = ['Automatic', 'Manual', 'Semi-Auto']
    transmission_dict = {}
    for trans in transmission:
        transmission_dict[trans] = 1 if trans == value_picked else 0

    return transmission_dict


def models(value_picked):
    model = [' Fiesta', ' Focus', ' Puma', ' Kuga', ' EcoSport', ' C-MAX',
                        ' Mondeo', ' Ka+', ' Tourneo Custom', ' S-MAX', ' B-MAX', ' Edge',
                        ' Tourneo Connect', ' Grand C-MAX', ' KA', ' Galaxy', ' Mustang',
                        ' Grand Tourneo Connect', ' Fusion', ' Ranger', ' Streetka',
                        ' Escort', ' Transit Tourneo', 'Focus']
    model_dict = {}
    for mo in model:
        model_dict[mo] = 1 if mo == value_picked else 0

    return model_dict
    
    
def import_model(selected_model):
    
    if selected_model == 'RandomForest Regressor':
        model = joblib.load('randomforestregressor_model_1.joblib') 
    if selected_model == 'Gradient Boosting Regressor':
        model = joblib.load('gradientboostingregressor_model_2.joblib') 
        
    return model
    
def predict_function(input_data,model):
    
    prediction = model.predict(input_data)
    return prediction
    


