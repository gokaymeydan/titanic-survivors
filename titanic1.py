import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

passenger_ids = test_df["PassengerId"]
# Combine for feature engineering
full_df = pd.concat([train_df, test_df], sort=False)

# Fill missing values
full_df["Embarked"].fillna(full_df["Embarked"].mode()[0], inplace=True)
full_df["Fare"].fillna(full_df["Fare"].median(), inplace=True)

# Extract Title
full_df["Title"] = full_df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
full_df["Title"] = full_df["Title"].replace(["Mlle", "Ms"], "Miss")
full_df["Title"] = full_df["Title"].replace("Mme", "Mrs")
full_df["Title"] = full_df["Title"].replace(
    [
        "Dr",
        "Major",
        "Col",
        "Rev",
        "Capt",
        "Sir",
        "Don",
        "Jonkheer",
        "Lady",
        "Countess",
        "Dona",
    ],
    "Rare",
)

# Fill Age by median of Title group
full_df["Age"] = full_df.groupby("Title")["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Create FamilySize and IsAlone
full_df["FamilySize"] = full_df["SibSp"] + full_df["Parch"] + 1
full_df["IsAlone"] = (full_df["FamilySize"] == 1).astype(int)

# Bin Age and Fare
full_df["AgeBin"] = pd.cut(full_df["Age"], bins=[0, 12, 20, 40, 60, 80], labels=False)
full_df["FareBin"] = pd.qcut(full_df["Fare"], 4, labels=False)

# Drop unused columns
full_df.drop(["Cabin", "Ticket", "Name"], axis=1, inplace=True)

# One-hot encode categorical features
full_df = pd.get_dummies(full_df, columns=["Sex", "Embarked", "Title"], drop_first=True)

# Split back into train and test
train_processed = full_df[: len(train_df)]
test_processed = full_df[len(train_df) :]

X = train_processed.drop("Survived", axis=1)
y = train_processed["Survived"]
X_test = test_processed.drop("Survived", axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Evaluate
train_preds = model.predict(X_scaled)
print("Training Accuracy:", accuracy_score(y, train_preds))


predictions = model.predict(X_test_scaled).astype(int)
submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
submission.to_csv("my_submission.csv", index=False)
print("my_submission.csv created.")

# Visualizations

sns.set_theme(style="darkgrid", palette="Set1")

# Survival rate by Sex
plt.figure(figsize=(7, 5))
sns.barplot(x="Sex", y="Survived", data=train_df, palette="Set1")
plt.title("Survival Rate by Sex", fontsize=14, fontweight="bold")
plt.xlabel("Sex", fontsize=12)
plt.ylabel("Survival Rate", fontsize=12)
plt.tight_layout()
plt.show()

# Survival rate by Pclass
plt.figure(figsize=(7, 5))
sns.barplot(x="Pclass", y="Survived", data=train_df, palette="Set1")
plt.title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold")
plt.xlabel("Pclass", fontsize=12)
plt.ylabel("Survival Rate", fontsize=12)
plt.tight_layout()
plt.show()


# Title distribution
plt.figure(figsize=(9, 5))
title_df = train_df.copy()
title_df["Title"] = title_df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
title_df["Title"] = title_df["Title"].replace(["Mlle", "Ms"], "Miss")
title_df["Title"] = title_df["Title"].replace("Mme", "Mrs")
title_df["Title"] = title_df["Title"].replace(
    [
        "Dr",
        "Major",
        "Col",
        "Rev",
        "Capt",
        "Sir",
        "Don",
        "Jonkheer",
        "Lady",
        "Countess",
        "Dona",
    ],
    "Rare",
)
sns.countplot(
    x="Title",
    data=title_df,
    order=title_df["Title"].value_counts().index,
    palette="Set1",
)
plt.title("Passenger Title Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Title", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Survival by Sex with percentage labels
plt.figure(figsize=(8, 5))
ax = sns.barplot(x="Sex", y="Survived", data=train_df, palette="Set1")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2,
        height + 0.02,
        f"{height*100:.1f}%",
        ha="center",
        fontsize=11,
    )
plt.title("Survival Rate by Sex")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("survival_by_sex.png")
plt.close()

# Survival by Class with labels
plt.figure(figsize=(8, 5))
ax = sns.barplot(x="Pclass", y="Survived", data=train_df, palette="Set1")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2,
        height + 0.02,
        f"{height*100:.1f}%",
        ha="center",
        fontsize=11,
    )
plt.title("Survival Rate by Passenger Class")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.close()

# Title Distribution and Survival Rate by Title
title_df = train_df.copy()
title_df["Title"] = title_df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
title_df["Title"] = title_df["Title"].replace(["Mlle", "Ms"], "Miss")
title_df["Title"] = title_df["Title"].replace("Mme", "Mrs")
title_df["Title"] = title_df["Title"].replace(
    [
        "Dr",
        "Major",
        "Col",
        "Rev",
        "Capt",
        "Sir",
        "Don",
        "Jonkheer",
        "Lady",
        "Countess",
        "Dona",
    ],
    "Rare",
)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(
    x="Title",
    data=title_df,
    ax=axs[0],
    order=title_df["Title"].value_counts().index,
    palette="Set1",
)
axs[0].set_title("Title Distribution")

ax = sns.barplot(
    x="Title",
    y="Survived",
    data=title_df,
    ax=axs[1],
    order=title_df["Title"].value_counts().index,
    palette="Set1",
)
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2,
        height + 0.02,
        f"{height*100:.1f}%",
        ha="center",
        fontsize=10,
    )
axs[1].set_title("Survival Rate by Title")
axs[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig("title_analysis.png")
plt.close()


# Show missing values summary in terminal (if any)
missing_summary = train_df.isnull().sum()
missing_pct = (missing_summary / len(train_df)) * 100
missing_df = pd.DataFrame(
    {
        "Feature": missing_summary.index,
        "Missing Values": missing_summary.values,
        "% Missing": missing_pct.round(2),
    }
)
missing_df = missing_df[missing_df["Missing Values"] > 0]
if not missing_df.empty:
    print("\nMissing Values Summary:")
    print(missing_df.to_string(index=False))

# Feature correlation bar chart
full_corr = full_df.copy()
full_corr["Survived"] = y.tolist() + [np.nan] * (full_df.shape[0] - len(y))
corr = (
    full_corr.corr(numeric_only=True)["Survived"]
    .drop("Survived")
    .sort_values(ascending=False)
    .dropna()
)

plt.figure(figsize=(12, 6))
sns.barplot(x=corr.index, y=corr.values, palette="coolwarm")
plt.title("Feature Correlation with Survival")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
