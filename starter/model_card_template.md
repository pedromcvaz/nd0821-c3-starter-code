# Model Card
- Author: Pedro Vaz
- Last updated: March 2022
## Model Details
The algorithm used to train the model was Random Forest Classifier from scikit learn.
The dataset used has around 32k records.
The model was trained using 80% of the available data, the 20% left were using for validation.
The goal of the model is to predict if a person's earnings is above or below $50K.

## Training Data
The model was trained using 80% of the census data provided.

## Evaluation Data
The evaluation dataset consists of the remaining 20% of the census data provided.

## Metrics
precision: 0.9574062301335029
recall: 0.9250614250614251
fbeta: 0.9409559512652297

## Ethical Considerations
The data used was publicly available and was gathered with consent from the individuals.
This model was generated in a strictly academic purpose and should in no way be used outside of this context.
## Caveats and Recommendations
This is a very simplistic model as building a good performing model was out of scope from this project.
Feature engineering and hyper-parameter tuning can be used to increase model performance.