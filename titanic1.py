import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

# Load data

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
submission_df = pd.read_csv("gender_submission.csv")


def preprocess_data(df):
    df = df.copy()  # Handle missing values

    # fill missing age and fare
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # fill missing embarked
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # encode sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # encode embarked

    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    return df


# preprocess datasets
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# features

for df in [train_df, test_df]:
    # delete titles from 'name' column
    df["Title"] = df["Name"].str.extract("([A-Za-z]+)\.", expand=False)
    # unique titles grouped by 'rare'
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    # change titles
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")
    # titles turn into numbers
    df["Title"] = (
        df["Title"]
        .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
        .fillna(0)
    )

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # FamilySize calculating
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)  # If alone 1 else 0

    # bin fare into four quantile_based groups
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)

    # bin age into fixed intervals
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, 80], labels=False)

# graphs
sns.barplot(x="Title", y="Survived", data=train_df)
plt.title("Survival Rate by Title")
plt.show()

sns.barplot(x="FamilySize", y="Survived", data=train_df)
plt.title("Survival Rate by Family Size")
plt.show()

sns.countplot(x="Sex", hue="Survived", data=train_df)
plt.title("Survival Count by Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.xticks([0, 1], ["Male", "Female"])  # sayıları etiketle
plt.tight_layout()
plt.show()

g = sns.catplot(
    x="Embarked",
    col="Pclass",
    hue="Survived",
    kind="count",
    data=train_df,
    height=4,
    aspect=1,
)
g.set_titles("Pclass = {col_name}")
g.set_axis_labels("Embarked", "Count")
g._legend.set_title("Survived")
g._legend.set_bbox_to_anchor((1, 0.5))
plt.tight_layout()
plt.show()

# select features (inputs column)
features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone",
]
X = train_df[features]
y = train_df["Survived"]

# RandomForest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)

# XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learnin_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

model.fit(X, y)

# feature importance graph
plt.figure(figsize=(10, 6))
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values().plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# test data
X_test = test_df[features]
predictions = model.predict(X_test)

# save submission
output_df = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Survived": predictions}
)
output_df.to_csv("my_submission.csv", index=False)
print("Submission file 'my_submission.csv' created successfully.")
