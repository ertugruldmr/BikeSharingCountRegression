import pickle
import json 
import gradio as gr
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostRegressor


# File Paths
model_path = 'finalized_model.sav'
component_config_path = "component_configs.json"

# predefined
feature_order = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum']

# Loading the files
model = pickle.load(open(model_path, 'rb'))

feature_limitations = json.load(open(component_config_path, "r"))


# Code
examples = [
  [4.0, 0.0, 10.0, 2.0, 0.0, 6.0, 0.0, 1.0, 0.4, 0.62],
  [4.0, 0.0, 10.0, 11.0, 0.0, 1.0, 1.0, 2.0, 0.4, 0.71],
  [4.0, 1.0, 10.0, 1.0, 0.0, 3.0, 1.0, 2.0, 0.62, 0.94],
  [4.0, 1.0, 12.0, 19.0, 0.0, 2.0, 1.0, 1.0, 0.38, 0.46],
  [2.0, 1.0, 4.0, 2.0, 0.0, 5.0, 1.0, 1.0, 0.44, 0.88],
  [4.0, 1.0, 12.0, 14.0, 0.0, 2.0, 1.0, 2.0, 0.38, 0.5]
 ]


# Util function
def predict(*args):

  # preparing the input into convenient form
  features = pd.Series([*args], index=feature_order)
  features = np.array(features).reshape(-1, len(feature_order))

  # prediction
  pred = model.predict(features) #.predict(features)

  return np.round(pred,5)

# Creating the gui component according to component.json file
inputs = list()
for col in feature_order:
  if col in feature_limitations["cat"].keys():
    
    # extracting the params
    vals = feature_limitations["cat"][col]["values"]
    def_val = feature_limitations["cat"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Dropdown(vals, default=def_val, label=col))
  else:
    
    # extracting the params
    min = feature_limitations["num"][col]["min"]
    max = feature_limitations["num"][col]["max"]
    def_val = feature_limitations["num"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Slider(minimum=min, maximum=max, default=def_val, label=col) )

demo_app = gr.Interface(predict, inputs, "number", examples=examples)

# Launching the demo
if __name__ == "__main__":
    demo_app.launch()
