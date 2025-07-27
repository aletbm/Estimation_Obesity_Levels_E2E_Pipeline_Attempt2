import pandas as pd
import numpy as np
import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
import os
from google.cloud import storage
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from catboost import CatBoostClassifier

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
input_folder = "./"
data_folder = input_folder + "data/"
model_folder = input_folder + "models/"

@task
def load_data():
    df = pd.read_csv(data_folder + "ObesityDataSet_raw_and_data_sinthetic.csv")
    df = df.rename(columns={"family_history_with_overweight": "overweight_familiar",
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
    return df


def clean_data(df):
    df = df.drop_duplicates()
    df = df.drop(["weight"], axis=1)
    return df


def split_data(df, target):
    df_full_train, df_test = train_test_split(df, test_size=0.15, random_state=seed_value, stratify=df[target])
    df_train, df_val = train_test_split(df_full_train, test_size=0.15, random_state=seed_value, stratify=df_full_train[target])
    return df_train, df_val, df_test


def get_X_y(df, target):
    X, y = df.drop([target], axis=1), df[target]
    return X, y


def data_scaling(X_train, X_val, X_test, cols):
    ss = StandardScaler().set_output(transform="pandas")

    X_train[cols] = ss.fit_transform(X_train[cols])
    X_val[cols] = ss.transform(X_val[cols])
    X_test[cols] = ss.transform(X_test[cols])
    return X_train, X_val, X_test, ss


def ohe(X_train, X_val, X_test):
    dict_X_train = X_train.to_dict("records")
    dict_X_val = X_val.to_dict("records")
    dict_X_test = X_test.to_dict("records")

    dv = DictVectorizer(sparse=False).set_output(transform="pandas")
    X_train = dv.fit_transform(dict_X_train)
    X_val = dv.transform(dict_X_val)
    X_test = dv.transform(dict_X_test)
    return X_train, X_val, X_test, dv


def label_encoding(y_train, y_val, y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    return y_train, y_val, y_test, le

@task
def prepare_data(df):
    df = clean_data(df)
    df_train, df_val, df_test = split_data(df, target="obesity_level")
    
    X_train, y_train = get_X_y(df_train, target="obesity_level")
    X_val, y_val = get_X_y(df_val, target="obesity_level")
    X_test, y_test = get_X_y(df_test, target="obesity_level")

    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns
    X_train, X_val, X_test, ss = data_scaling(X_train, X_val, X_test, cols=numeric_cols)
    X_train, X_val, X_test, dv = ohe(X_train, X_val, X_test)
    y_train, y_val, y_test, le = label_encoding(y_train, y_val, y_test)

    with open(model_folder + "ss.pkl", "wb") as f:
        cloudpickle.dump(ss, f)

    with open(model_folder + "dv.pkl", "wb") as f:
        cloudpickle.dump(dv, f)

    with open(model_folder + "le.pkl", "wb") as f:
        cloudpickle.dump(le, f)

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_scores(y_true, y_pred, y_pred_proba):
    return {
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "ROC AUC": roc_auc_score(
            y_true, y_pred_proba, average="weighted", multi_class="ovr"
        ),
    }

@task
def training(X_train, y_train, X_val, y_val, X_test, y_test):
    class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(zip(np.unique(y_train), class_weight))

    sample_weights = compute_sample_weight(
        class_weight=class_weight,
        y=y_train
    )
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("obesity_level_experiment")

    with mlflow.start_run():
        cbc = CatBoostClassifier(loss_function='MultiClass',
                         eval_metric='AUC',
                         iterations=5000,
                         depth=6,
                         classes_count=7,
                         class_weights=class_weight,
                         learning_rate=0.1,
                         od_type='Iter',
                         early_stopping_rounds=1000,
                         bootstrap_type='MVS',
                         sampling_frequency='PerTree',
                         random_seed=seed_value,
                         verbose=200)
        cbc.fit(X_train, y_train, sample_weight=sample_weights, eval_set=(X_val, y_val))

        y_pred = cbc.predict(X_test)
        y_pred_proba = cbc.predict_proba(X_test)
        scores = get_scores(y_test, y_pred, y_pred_proba)

        mlflow.log_param("feature_importances_", cbc.feature_importances_)
        mlflow.log_param("best_score", cbc.best_score_)
        mlflow.log_param("best_iteration", cbc.best_iteration_)
        mlflow.log_param("params", cbc.get_all_params())
        mlflow.log_metric("Balanced Accuracy", scores["Balanced Accuracy"])
        mlflow.log_metric("F1 Score", scores["F1 Score"])
        mlflow.log_metric("Precision", scores["Precision"])
        mlflow.log_metric("Recall", scores["Recall"])
        mlflow.log_metric("ROC AUC", scores["ROC AUC"])

        model_info = mlflow.catboost.log_model(
            cb_model=cbc,
            artifact_path="catboost_model",
            registered_model_name="MyCatBoostClassifier",
            signature=mlflow.models.infer_signature(X_train, cbc.predict(X_train)),
            input_example=X_train[0:2],
        )
        ss_path = model_folder + "ss.pkl"
        dv_path = model_folder + "dv.pkl"
        le_path = model_folder + "le.pkl"

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/catboost_model"

        model_info = mlflow.register_model(model_uri, "MyCatBoostClassifier")

        mlflow.log_artifact(ss_path, artifact_path="preprocessing")
        mlflow.log_artifact(dv_path, artifact_path="preprocessing")
        mlflow.log_artifact(le_path, artifact_path="preprocessing")

        os.remove(ss_path)
        os.remove(dv_path)
        os.remove(le_path)

        client = MlflowClient()

        client.transition_model_version_stage(
            name="MyCatBoostClassifier",
            version=model_info.version,
            stage="Staging",
            archive_existing_versions=False,
        )

    return cbc

@flow(name="Obesity Level ML Pipeline", retries=1, retry_delay_seconds=300)
def obesity_level_pipeline():
    train, val, test = load_data()
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = prepare_data(train, val, test)
    training(X_train, y_train, X_val, y_val, X_test, y_test)
    return


if __name__ == "__main__":
    obesity_level_pipeline()

