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

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


if __name__ == "__main__":

    np.random.seed(0)

    abalone_df = pd.read_csv("./data/abalone.csv")
    # print(abalone_df.head())

    x_columns = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
    ]
    y_columns = ["Rings"]

    numeric_features = [
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
    ]
    categorical_features = ["Sex"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor()
    }

    clf_pipeline_factory = lambda model: Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    classifiers = {name: clf_pipeline_factory(model) for name, model in models.items()}

    metrics = {
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
    }

    x_numpy = abalone_df[x_columns]
    y_numpy = abalone_df[y_columns]

    x_train, x_test, y_train, y_test = train_test_split(x_numpy, y_numpy, test_size=0.3, random_state=42)

    results_fp = Path("./q02_results.csv")

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
        results_df.to_csv("./q02_results.csv", index=False)
    else:
        results_df = pd.read_csv("./q02_results.csv")

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
    axes = g.axes

    fig.set_size_inches(10, 10)

    plt.tight_layout()
    plt.savefig("./q02_results.png")


