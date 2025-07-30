from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import cloudpickle
from catboost import CatBoostClassifier

RUN_ID = "5627f2b147af4c728171297efc479418"
artifacts_path = f"./models/1/{RUN_ID}/artifacts/"

model_path = f"{artifacts_path}/catboost_model/model.cb"
pipe_path = f"{artifacts_path}/preprocessing/pipe.pkl"
le_path = f"{artifacts_path}/preprocessing/le.pkl"

model = CatBoostClassifier()
model.load_model(model_path)

with open(pipe_path, "rb") as f:
    pipe = cloudpickle.load(f)

with open(le_path, "rb") as f:
    le = cloudpickle.load(f)


app = FastAPI()


class InputData(BaseModel):
    features: dict


@app.get("/")
def read_root():
    return {"message": "API succesfully working"}


@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features])
    df = pipe.transform(df)
    prediction = model.predict(df)
    return {"prediction": le.inverse_transform(prediction)[0]}
