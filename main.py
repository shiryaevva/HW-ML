import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import joblib
import re

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.get('/')
async def root():
    return {'message': 'Welcome to selling_price prediction API'}


lr_ridge = joblib.load('lr_ridge.pkl')
sc = joblib.load('scaler.pkl')
enc = joblib.load('OHE.pkl')
median = joblib.load('median.pkl')


def remove_um(x):
    return '.'.join(re.findall(r'\d+', x))


def preprocessor(input_data):
    input_data = input_data.drop('name', axis=1)

    input_data['mileage'] = input_data['mileage'].apply(str).apply(remove_um)
    input_data['engine'] = input_data['engine'].apply(str).apply(remove_um)
    input_data['max_power'] = input_data['max_power'].apply(str).apply(remove_um)

    input_data['mileage'] = pd.to_numeric(input_data['mileage'], downcast='float', errors='coerce')
    input_data['engine'] = pd.to_numeric(input_data['engine'], downcast='float', errors='coerce')
    input_data['max_power'] = pd.to_numeric(input_data['max_power'], downcast='float', errors='coerce')

    input_data['mileage_dummy'] = np.where(input_data.mileage.isnull(), 1, 0)
    input_data['engine_dummy'] = np.where(input_data.engine.isnull(), 1, 0)
    input_data['max_power_dummy'] = np.where(input_data.max_power.isnull(), 1, 0)
    input_data['seats_dummy'] = np.where(input_data.seats.isnull(), 1, 0)

    input_data['mileage'] = input_data['mileage'].fillna(median['mileage'])
    input_data['engine'] = input_data['engine'].fillna(median['engine'])
    input_data['max_power'] = input_data['max_power'].fillna(median['max_power'])
    input_data['seats'] = input_data['seats'].fillna(median['seats'])

    input_data['squared_year'] = input_data['year'] ** 2
    input_data['hp_per_v'] = input_data['max_power'] / input_data['engine']
    input_data['log_km_driven'] = np.log(input_data.km_driven)
    input_data['third_above_owner'] = np.where(input_data.owner.isin(['First Owner', 'Second Owner', 'Test Drive Car']), 0, 1)

    assert (input_data.shape[1] == 18, 'features amount missmatch!')
    return input_data


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    d = item.dict().copy()
    input_data = pd.DataFrame([d])
    input_data = preprocessor(input_data)
    input_data_ohe = enc.transform(input_data.select_dtypes(include=object))
    input_data_e = pd.DataFrame(data=input_data_ohe,
                                columns=enc.get_feature_names_out(list(input_data.select_dtypes(include=object))))
    data = input_data.select_dtypes(include=np.number)
    input_data_enc = data.join(input_data_e)

    input_data_s = sc.transform(input_data_enc)
    input_data_scaled = pd.DataFrame(data=input_data_s, columns=list(input_data_enc))

    return lr_ridge.predict(input_data_scaled)[0]


@app.post("/predict_items")
def predict_items(file: UploadFile):
    data_input = pd.read_csv(file.filename)
    input_data = data_input.drop(['selling_price', 'torque'], axis=1)
    input_data = preprocessor(input_data)

    print(input_data.shape)

    input_data_ohe = enc.transform(input_data.select_dtypes(include=object))
    input_data_e = pd.DataFrame(data=input_data_ohe,
                                columns=enc.get_feature_names_out(list(input_data.select_dtypes(include=object))))
    data = input_data.select_dtypes(include=np.number)
    input_data_enc = data.join(input_data_e)

    input_data_s = sc.transform(input_data_enc)
    input_data_scaled = pd.DataFrame(data=input_data_s, columns=list(input_data_enc))

    data_input['prediction'] = lr_ridge.predict(input_data_scaled)
    data_input.to_csv('lib/prediction.csv', index=False)

    return FileResponse('lib/prediction.csv')
