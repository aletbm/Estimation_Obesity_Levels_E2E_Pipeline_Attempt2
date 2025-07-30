import pandas as pd
import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from evidently import Report, Dataset, DataDefinition
from evidently.presets import (
    DataDriftPreset,
    ClassificationPreset,
    DataSummaryPreset,
)
from evidently import MulticlassClassification
from sklearn.model_selection import train_test_split
import prefect
from prefect import flow, task
import os
import requests
import sys
import json
from datetime import datetime
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipelines.training_flow import obesity_level_pipeline, load_data, clean_data, prepare_data
import config as cfg

ARTIFACT_DIR = "monitoring/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

@task
def load_model_and_artifacts(model_alias):
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(cfg.MODEL_NAME, model_alias)
    run_id = model_version.run_id

    model = mlflow.pyfunc.load_model(f"models:/{cfg.MODEL_NAME}@{model_alias}")
    pipe_path = client.download_artifacts(run_id, "preprocessing/pipe.pkl", ARTIFACT_DIR)
    le_path = client.download_artifacts(run_id, "preprocessing/le.pkl", ARTIFACT_DIR)

    with open(pipe_path, "rb") as f:
        pipe = cloudpickle.load(f)

    with open(le_path, "rb") as f:
        le = cloudpickle.load(f)

    return model, pipe, le


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
    df = clean_data(df)
    return df


def clean_data(df):
    df = df.drop_duplicates()
    df = df.drop(["weight"], axis=1)
    return df


@task
def split_data(df, target):
    df_full_train, df_test = train_test_split(df, test_size=0.15, random_state=cfg.SEED_VALUE, stratify=df[target])
    df_train, _ = train_test_split(df_full_train, test_size=0.15, random_state=cfg.SEED_VALUE, stratify=df_full_train[target])
    return df_train, df_test


@task
def prepare_datasets(model, pipe, le, train_df, test_df, target_col="obesity_level"):
    feature_cols = train_df.drop(columns=[target_col]).columns.tolist()
    numeric_cols = train_df.drop(columns=[target_col]).select_dtypes(exclude=["object"]).columns.tolist()
    
    def make_dataset(df):
        X = df[feature_cols].copy()
        X_ = pipe.transform(X)
        y = le.transform(df[target_col])
        preds = model.predict(X_)

        df2 = pd.DataFrame(X_, columns=X_.columns)
        df2["target"] = y
        df2["prediction"] = preds

        return Dataset.from_pandas(
            df2,
            data_definition=DataDefinition(
                classification=[
                    MulticlassClassification(
                        target="target", prediction_labels="prediction"
                    )
                ],
                numerical_columns=numeric_cols,
            ),
        )

    ds_train = make_dataset(train_df)
    ds_test = make_dataset(test_df)
    return ds_train, ds_test


@task
def run_monitoring(ds_train, ds_test):
    report = Report(
        metrics=[DataDriftPreset(), DataSummaryPreset(), ClassificationPreset()]
    )
    result = report.run(reference_data=ds_train, current_data=ds_test)
    result.save_html("monitoring/full_monitor_report.html")

    report_dict = result.json()
    report_dict = json.loads(report_dict)

    for metric in report_dict["metrics"]:
        metric_id = metric["metric_id"]
        value = metric["value"]

        if "DriftedColumnsCount" in metric_id:
            drift_score = value["share"]
            break

    print(f"Drift score: {drift_score}")
    return drift_score


@task
def send_slack_alert(message: str, state, flow_name):
    color = {
        "SUCCESS": "#36a64f",
        "FAILURE": "#ff0000",
        "INFO": "#3AA3E3",
        "WARNING": "#FFA500"
    }.get(state, "#cccccc")

    emoji = {
        "SUCCESS": "✅",
        "FAILURE": "❌",
        "INFO": "ℹ️",
        "WARNING": "⚠️"
    }.get(state, "ℹ️")

    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    template = {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{emoji} Prefect Flow Alert - {state}",
                            "emoji": True
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Flow:* `{flow_name}`"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Status:* *{state}*"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Time:* {date}"
                            },
                        ]
                    },
                ]
            }
        ]
    }

    if message:
        template["attachments"][0]["blocks"].append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Message:* `{message}`"
            }
        })

    if not cfg.SLACK_WEBHOOK_URL:
        print("Slack webhook not configured.")
        return

    response = requests.post(cfg.SLACK_WEBHOOK_URL, json=template)
    if response.status_code == 200:
        print("Slack alert sent successfully.")
    else:
        print(f"Failed to send Slack alert: {response.status_code} {response.text}")


@task
def check_drift_and_maybe_retrain(drift_score: float, model_alias: str, flow_name: str):
    if drift_score > cfg.DRIFT_THRESHOLD:
        alert_msg = f"Drift detected (score={drift_score:.3f}) > threshold ({cfg.DRIFT_THRESHOLD})"
        print(alert_msg)
        send_slack_alert(alert_msg, state="WARNING", flow_name=flow_name)

        msg = "Starting model retraining..."
        print(msg)
        send_slack_alert(msg, state="INFO", flow_name=flow_name)

        obesity_level_pipeline(model_alias=model_alias+"_retrain")

        msg = "Retraining completed."
        print(msg)
        send_slack_alert(msg, state="SUCCESS", flow_name=flow_name)
        return True
    else:
        msg = f"Drift is acceptable ({drift_score:.3f} ≤ {cfg.DRIFT_THRESHOLD})"
        print(msg)
        send_slack_alert(msg, state="SUCCESS", flow_name=flow_name)
        return False


@flow(name="Monitoring + Conditional Retraining")
def monitoring_flow(model_alias):
    flow_name = prefect.runtime.flow_run.name
    msg = "Monitoring started."
    send_slack_alert(msg, state="INFO", flow_name=flow_name)
    model, pipe, le = load_model_and_artifacts(model_alias=model_alias)
    df = load_data()
    train_df, test_df = split_data(df, target="obesity_level")
    ds_train, ds_test = prepare_datasets(model, pipe, le, train_df, test_df)
    drift_score = run_monitoring(ds_train, ds_test)
    check_drift_and_maybe_retrain(drift_score, model_alias=model_alias, flow_name=flow_name)


if __name__ == "__main__":
    alias = sys.argv[1]
    monitoring_flow(model_alias=alias)

# prefect server start
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models
