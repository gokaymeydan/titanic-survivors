# Titanic Survivors

A solution to the Titanic Kaggle competition using XGBoost and feature engineering to predict passenger survival.

## Files
- `titanic1.py`: main script
- `train.csv`, `test.csv`: input data
- `my_submission.csv`: output predictions

## Usage
```bash
pip install -r requirements.txt
python titanic1.py
```

## Model
- Classifier: XGBoost
- Cross-validation mean accuracy: ~0.8249

## Features
- Pclass, Sex, Age, Fare, Embarked, Title, FamilySize, IsAlone