import pandas as pd
import sys
import os
from prefect import flow, task
from google.cloud import storage
import cloudpickle
from mlflow.tracking import MlflowClient
from catboost import CatBoostClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as cfg

@task
def load_data(filename):
    test = pd.read_parquet(filename)
    test = test.rename(columns={"family_history_with_overweight": "overweight_familiar",
                       "FAVC":"eat_HC_food",
                       "FCVC":"eat_vegetables",
                       "NCP":"main_meals",
                       "CAEC":"snack",
                       "CH2O":"drink_water",
                       "SCC":"monitoring_calories",
                       "FAF":"physical_activity",
                       "TUE":"use_of_technology",
                       "CALC":"drink_alcohol",
                       "MTRANS":"transportation_type",
                       "NObeyesdad":"obesity_level"
                       }).rename(columns=str.lower)
    
    test = test.drop(["weight"], axis=1)
    return test


def split_X_y(test, target):
    return test.drop(target, axis=1), test[target]


@task
def prepare_data(test):
    X_test, y_test = split_X_y(test, "obesity_level")
    numeric_cols = X_test.select_dtypes(exclude=["object"]).columns
    X_test[numeric_cols] = ss.transform(X_test[numeric_cols])
    X_test = dv.transform(X_test.to_dict("records"))
    y_test = le.transform(y_test)
    return X_test, y_test


@task
def apply_model(X_test):
    y_pred = model.predict(X_test)
    return y_pred


@task
def make_result(df, y_pred):
    df["prediction"] = y_pred
    df_result = df[["prediction"]].copy()
    return df_result


@task
def save_result(df_result, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = f"{output_folder}/predictions.parquet"
    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    return


def upload_blob(project_id, bucket_name, source_file_name, destination_blob_name):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded as {destination_blob_name}")
    return


@task
def upload2cloud(project_id, bucket_name, output_folder):
    filename = f"{output_folder}/predictions.parquet"
    upload_blob(
        project_id=project_id,
        bucket_name=bucket_name,
        source_file_name=filename,
        destination_blob_name=filename,
    )
    return


@flow(name="Obesity Level Inference Pipeline", retries=1, retry_delay_seconds=300)
def obesity_level_inference_pipeline(project_id, bucket_name, filepath):
    output_folder = f"output"

    df = load_data(filepath)
    X_test, _ = prepare_data(df)
    y_pred = apply_model(X_test)
    df_result = make_result(df, y_pred)
    save_result(df_result, output_folder)
    #upload2cloud(project_id, bucket_name, output_folder)
    return


if __name__ == "__main__":
    alias = sys.argv[1]
    filepath = sys.argv[2]
    
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(cfg.MODEL_NAME, alias)
    RUN_ID = model_version.run_id
    artifacts_path = f"./models/1/{RUN_ID}/artifacts/"

    model_path = f"{artifacts_path}/catboost_model/model.cb"
    ss_path = f"{artifacts_path}/preprocessing/ss.pkl"
    dv_path = f"{artifacts_path}/preprocessing/dv.pkl"
    le_path = f"{artifacts_path}/preprocessing/le.pkl"

    model = CatBoostClassifier()
    model.load_model(model_path)

    with open(ss_path, "rb") as f:
        ss = cloudpickle.load(f)

    with open(dv_path, "rb") as f:
        dv = cloudpickle.load(f)

    with open(le_path, "rb") as f:
        le = cloudpickle.load(f)
        
    project_id = "plucky-haven-463121-j1"
    bucket_name = "plucky-haven-463121-j1-predictions"
    obesity_level_inference_pipeline(project_id, bucket_name, filepath)
