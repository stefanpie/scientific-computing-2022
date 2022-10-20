import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from pprint import pp

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from uritemplate import partial


def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    se = tp / (tp + fn)
    return se

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sp = tn / (tn + fp)
    return sp


if __name__ == "__main__":

    np.random.seed(0)

    heart_df = pd.read_csv("./data/heart.csv")
    # print(heart_df.head())

    x_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]
    y_columns = ["target"]

    numeric_features = ["age", "trestbps", "chol", "fbs", "thalach", "oldpeak"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models = {
        "LogisticRegression": LogisticRegression(),
        "SVC": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
    }

    clf_pipeline_factory = lambda model: Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    classifiers = {name: clf_pipeline_factory(model) for name, model in models.items()}

    metrics = {
        "accuracy": accuracy_score,
        "percision": precision_score,
        "f1": f1_score,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

    x_numpy = heart_df[x_columns]
    y_numpy = heart_df[y_columns]

    x_train, x_test, y_train, y_test = train_test_split(x_numpy, y_numpy, test_size=0.3, random_state=42)

    results_fp = Path("./q01_results.csv")

    # check if results file exists
    if not results_fp.exists():


        results_df = pd.DataFrame(columns=["classifier", "metric", "split", "value"])
        for mode_name, model in classifiers.items():
            print(f"Fitting {mode_name}...")
            model.fit(x_train, y_train.values.ravel())
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            for metric_name, metric in metrics.items():
                metric_train = metric(y_train, y_train_pred)
                metric_test = metric(y_test, y_test_pred)
                results_df = pd.concat([pd.DataFrame({
                    "classifier": mode_name,
                    "metric": metric_name,
                    "split": "train",
                    "value": metric_train,
                }, index=[0]), results_df], ignore_index=True)
                results_df = pd.concat([pd.DataFrame({
                    "classifier": mode_name,
                    "metric": metric_name,
                    "split": "test",
                    "value": metric_test,
                }, index=[0]), results_df], ignore_index=True)

        results_df.to_csv("./q01_results.csv", index=False)
    else:
        results_df = pd.read_csv("./q01_results.csv")
    
    print(results_df)
    g = sns.catplot(
        x="classifier",
        y="value",
        hue="split",
        kind="bar",
        row="metric",
        data=results_df,
        legend_out=True
    )

    fig = g.figure
    fig.set_size_inches(10, 10)

    plt.tight_layout()
    plt.savefig("./q01_results.png")

