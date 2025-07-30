import pandas as pd
import numpy as np
import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
import os
import sys
from google.cloud import storage
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as cfg

@task
def load_data():
    df = pd.read_csv(cfg.DATA_FOLDER + "ObesityDataSet_raw_and_data_sinthetic.csv")
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
    df_full_train, df_test = train_test_split(df, test_size=0.15, random_state=cfg.SEED_VALUE, stratify=df[target])
    df_train, df_val = train_test_split(df_full_train, test_size=0.15, random_state=cfg.SEED_VALUE, stratify=df_full_train[target])
    return df_train, df_val, df_test


def get_X_y(df, target):
    X, y = df.drop([target], axis=1), df[target]
    return X, y

@task
def transform_data(X_train, X_val, X_test):
    class MyStandardScaler(BaseEstimator, TransformerMixin):
        def __init__(self, numeric_cols):
            self.ss = StandardScaler().set_output(transform="pandas")
            self.numeric_cols = numeric_cols
            return

        def fit(self, X):
            self.ss.fit(X[self.numeric_cols])
            return self

        def transform(self, X):
            X[self.numeric_cols] = self.ss.transform(X[self.numeric_cols])
            return X.to_dict("records")

    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns
    pipe = Pipeline([('ss', MyStandardScaler(numeric_cols=numeric_cols)), ('dv', DictVectorizer(sparse=False).set_output(transform="pandas"))])

    X_train = pipe.fit_transform(X_train)
    X_val = pipe.transform(X_val)
    X_test = pipe.transform(X_test)
    return X_train, X_val, X_test, pipe


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

    X_train, X_val, X_test, pipe = transform_data(X_train, X_val, X_test)
    y_train, y_val, y_test, le = label_encoding(y_train, y_val, y_test)

    with open(cfg.MODEL_FOLDER + "pipe.pkl", "wb") as f:
        cloudpickle.dump(pipe, f)

    with open(cfg.MODEL_FOLDER + "le.pkl", "wb") as f:
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
def training(X_train, y_train, X_val, y_val, X_test, y_test, model_alias):
    class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(zip(np.unique(y_train), class_weight))

    sample_weights = compute_sample_weight(
        class_weight=class_weight,
        y=y_train
    )

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
                         random_seed=cfg.SEED_VALUE,
                         verbose=200)
        cbc.fit(X_train, y_train, sample_weight=sample_weights, eval_set=(X_val, y_val))

        y_pred = cbc.predict(X_test)
        y_pred_proba = cbc.predict_proba(X_test)
        scores = get_scores(y_test, y_pred, y_pred_proba)

        mlflow.log_param("feature_importances_", cbc.feature_importances_)
        mlflow.log_param("best_score", cbc.best_score_)
        mlflow.log_param("best_iteration", cbc.best_iteration_)
        mlflow.log_param("params", cbc.get_all_params())

        for metric in ["Balanced Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]:
            mlflow.log_metric(metric, scores[metric])

        mlflow.catboost.log_model(
            cb_model=cbc,
            artifact_path="catboost_model",
            registered_model_name=cfg.MODEL_NAME,
            signature=mlflow.models.infer_signature(X_train, cbc.predict(X_train)),
            input_example=X_train[0:2],
        )
        pipe_path = os.path.join(cfg.MODEL_FOLDER, "pipe.pkl")
        le_path = os.path.join(cfg.MODEL_FOLDER, "le.pkl")

        mlflow.log_artifact(pipe_path, artifact_path="preprocessing")
        mlflow.log_artifact(le_path, artifact_path="preprocessing")

        os.remove(pipe_path)
        os.remove(le_path)

        client = MlflowClient()

        latest_mv = client.get_latest_versions(cfg.MODEL_NAME, stages=["None"])[0]
        client.set_registered_model_alias(
            cfg.MODEL_NAME, model_alias, latest_mv.version
        )

        client.set_model_version_tag(
                name=cfg.MODEL_NAME,
                version=latest_mv.version,
                key="task",
                value="classification"
            )

        for metric in ["Balanced Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]:
            client.set_model_version_tag(
                name=cfg.MODEL_NAME,
                version=latest_mv.version,
                key=metric.replace(" ", "_").lower(),
                value=str(scores[metric])
            )

    return cbc

@flow(name="Obesity Level ML Pipeline", retries=1, retry_delay_seconds=300)
def obesity_level_pipeline(model_alias):
    df = load_data()
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = prepare_data(df)
    training(X_train, y_train, X_val, y_val, X_test, y_test, model_alias=model_alias)
    return


if __name__ == "__main__":
    alias = sys.argv[1]
    obesity_level_pipeline(model_alias=alias)

#make run-training ALIAS=champion