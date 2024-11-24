from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)

app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            bedrooms = request.form.get('bedrooms'),
            bathrooms = request.form.get('bathrooms'),
            sqft_living = request.form.get('sqft_living'),
            sqft_lot = request.form.get('sqft_lot'),
            floors = request.form.get('floors'),
            condition = request.form.get('condition'),
            grade = request.form.get('grade'),
            sqft_basement = request.form.get('sqft_basement'),
            yr_built = request.form.get('yr_built'),
            lat = request.form.get('lat'),
            long = request.form.get('long')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)    


