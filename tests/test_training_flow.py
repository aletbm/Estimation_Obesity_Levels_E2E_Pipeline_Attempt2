import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipelines import training_flow
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import config as cfg

input_folder = "./"
data_folder = input_folder + "data/"


def test_transform_data():
    df = pd.read_csv(cfg.DATA_FOLDER + "ObesityDataSet_raw_and_data_sinthetic.csv")
    df = df.rename(
        columns={
            "family_history_with_overweight": "overweight_familiar",
            "FAVC": "eat_HC_food",
            "FCVC": "eat_vegetables",
            "NCP": "main_meals",
            "CAEC": "snack",
            "CH2O": "drink_water",
            "SCC": "monitoring_calories",
            "FAF": "physical_activity",
            "TUE": "use_of_technology",
            "CALC": "drink_alcohol",
            "MTRANS": "transportation_type",
            "NObeyesdad": "obesity_level",
        }
    ).rename(columns=str.lower)
    df = df.drop_duplicates()
    df = df.drop(["weight"], axis=1)
    df_full_train, df_test = train_test_split(
        df, test_size=0.15, random_state=cfg.SEED_VALUE, stratify=df["obesity_level"]
    )
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.15,
        random_state=cfg.SEED_VALUE,
        stratify=df_full_train["obesity_level"],
    )

    X_train, _ = df_train.drop(["obesity_level"], axis=1), df_train["obesity_level"]
    X_val, _ = df_val.drop(["obesity_level"], axis=1), df_val["obesity_level"]
    X_test, _ = df_test.drop(["obesity_level"], axis=1), df_test["obesity_level"]
    act_X_train, act_X_val, act_X_test, _ = training_flow.transform_data(
        X_train.copy(), X_val.copy(), X_test.copy()
    )

    print(act_X_train)

    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns
    X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val.loc[:, numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

    X_train_dict = X_train.to_dict("records")
    X_val_dict = X_val.to_dict("records")
    X_test_dict = X_test.to_dict("records")

    dv = DictVectorizer(sparse=False).set_output(transform="pandas")
    X_train_dict = dv.fit_transform(X_train_dict)
    X_val_dict = dv.transform(X_val_dict)
    X_test_dict = dv.transform(X_test_dict)

    assert np.allclose(np.std(X_train_dict), np.std(act_X_train))
    assert np.allclose(np.std(X_val_dict), np.std(act_X_val))
    assert np.allclose(np.std(X_test_dict), np.std(act_X_test))

    assert len(X_train_dict) == len(act_X_train)
    assert len(X_val_dict) == len(act_X_val)
    assert len(X_test_dict) == len(act_X_test)


def test_label_encoding():
    df = pd.read_csv(cfg.DATA_FOLDER + "ObesityDataSet_raw_and_data_sinthetic.csv")
    df = df.rename(
        columns={
            "family_history_with_overweight": "overweight_familiar",
            "FAVC": "eat_HC_food",
            "FCVC": "eat_vegetables",
            "NCP": "main_meals",
            "CAEC": "snack",
            "CH2O": "drink_water",
            "SCC": "monitoring_calories",
            "FAF": "physical_activity",
            "TUE": "use_of_technology",
            "CALC": "drink_alcohol",
            "MTRANS": "transportation_type",
            "NObeyesdad": "obesity_level",
        }
    ).rename(columns=str.lower)
    df = df.drop_duplicates()
    df = df.drop(["weight"], axis=1)
    df_full_train, df_test = train_test_split(
        df, test_size=0.15, random_state=cfg.SEED_VALUE, stratify=df["obesity_level"]
    )
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.15,
        random_state=cfg.SEED_VALUE,
        stratify=df_full_train["obesity_level"],
    )

    _, y_train = df_train.drop("obesity_level", axis=1), df_train["obesity_level"]
    _, y_val = df_val.drop("obesity_level", axis=1), df_val["obesity_level"]
    _, y_test = df_test.drop("obesity_level", axis=1), df_test["obesity_level"]

    act_y_train, act_y_val, act_y_test, _ = training_flow.label_encoding(
        y_train, y_val, y_test
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    assert np.array_equal(np.unique(y_train), np.unique(act_y_train))
    assert np.array_equal(np.unique(y_val), np.unique(act_y_val))
    assert np.array_equal(np.unique(y_test), np.unique(act_y_test))

    assert len(y_train) == len(act_y_train)
    assert len(y_val) == len(act_y_val)
    assert len(y_test) == len(act_y_test)
