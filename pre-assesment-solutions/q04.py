from pprint import pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pp

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, f1_score


if __name__ == "__main__":
    data_df = pd.read_csv("./data/ecoli_data/ecoli.csv")
    x_columns = ["MCG", "GVH", "LIP", "CHG", "AAC", "ALM1", "ALM2"]

    x = data_df[x_columns]
    y = data_df["SITE"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    models = {
        "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000),
        "Support Vector Machine": SVC(
            gamma="auto", kernel="rbf", C=1.0, probability=True
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, min_samples_split=5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=0
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=10),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        ),
        "MLP": MLPClassifier(alpha=1, max_iter=1000),
    }

    for name, model in models.items():
        model.fit(x_train, y_train)

    results = {}
    for name, model in models.items():
        y_pred = model.predict(x_test)
        results[name] = {
            "train": {
                "accuracy": model.score(x_train, y_train),
                "f1": f1_score(y_train, model.predict(x_train), average="macro"),
            },
            "test": {
                "accuracy": model.score(x_test, y_test),
                "f1": f1_score(y_test, y_pred, average="macro"),
            },
        }

    results_df = pd.DataFrame()
    for name, result in results.items():
        splits = ["train", "test"]
        for split in splits:
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "model": name,
                            "split": split,
                            "accuracy": result[split]["accuracy"],
                            "f1": result[split]["f1"],
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    fig, ax = plt.subplots(nrows=2, figsize=(12, 6))

    sns.barplot(
        x="model",
        y="accuracy",
        hue="split",
        data=results_df,
        ax=ax[0],
    )

    sns.barplot(
        x="model",
        y="f1",
        hue="split",
        data=results_df,
        ax=ax[1],
    )

    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Model")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_ylim(0, 1)
    ax[0].set_yticks(np.arange(0, 1.1, 0.1))
    ax[0].set_yticklabels(["{:.1f}".format(x) for x in ax[0].get_yticks()])
    ax[0].yaxis.grid(True)
    ax[0].set_axisbelow(True)
    ax[0].legend(loc="lower right")

    ax[1].set_title("F1 Score")
    ax[1].set_xlabel("Model")
    ax[1].set_ylabel("F1 Score")
    ax[1].set_ylim(0, 1)
    ax[1].set_yticks(np.arange(0, 1.1, 0.1))
    ax[1].set_yticklabels(["{:.1f}".format(x) for x in ax[1].get_yticks()])
    ax[1].yaxis.grid(True)
    ax[1].set_axisbelow(True)
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    max_cols = 4
    num_rows_needed = int(np.ceil(len(models) / max_cols))
    fig, axs = plt.subplots(nrows=num_rows_needed, ncols=max_cols, figsize=(12, 6))
    for i, (name, model) in enumerate(models.items()):
        ax = axs.flatten()[i]
        y_pred = model.predict(x_test)
        labels = model.classes_
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, ax=ax, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_aspect("equal")
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels, rotation=45)

    for i in range(len(models), num_rows_needed * max_cols):
        axs.flatten()[i].axis("off")

    plt.tight_layout()
    plt.show()
