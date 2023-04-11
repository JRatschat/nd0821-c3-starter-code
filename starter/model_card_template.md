# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This repository contains a Random Forest Classifier that classifies whether an
individual has a salary of over USD 50,000 or not based various demographic features.
The model is based on the
(UCI Census Income Dataset)[https://archive.ics.uci.edu/ml/datasets/census+income].

## Intended Use
The classifier has been created as part of the "Performance Testing and Preparing a
Model for Production" task of the Udacity Machine Learning DevOps Engineering
Nanodegree.

## Training Data
Consists of 80 percent of UCI Census Income Dataset.

## Evaluation Data
Consists of 20 percent of UCI Census Income Dataset.

## Metrics
The metrics are based on the model's performance on the evaluation data:
- Precision: 0.7588
- Recall: 0.6265
- Fbeta: 0.6863

## Ethical Considerations
The sliced performance indicates evluation differences (e.g., race and nationality).
Since this model is only used for training purposes of the author, these differences are
not further addressed.

## Caveats and Recommendations
Model performance can probably be improved by using hyperparameter-tuning and choosing
another model (e.g., LightGBM).
