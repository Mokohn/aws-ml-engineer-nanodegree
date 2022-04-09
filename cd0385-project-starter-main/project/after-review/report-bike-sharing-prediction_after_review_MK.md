# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Moritz Kohnen

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
The Kaggle API only accepts positive numbers as result-values. So, I had to convert each predicted value that was smaller 0 (<0) to 0.

### What was the top ranked model that performed?
In all cases - default, feature-engineered, hyperparameters optimized - the WeightedEnsemble_L3 performed best. Though, the performance was quite close to CatBoost and LightGBM.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The EDA found, that we can convert season and weather into categorical features, which can be interpreted by AutoGluon as such. Additionally, I created new features from the datetime column, which gives the models more information for training and made the validation scores a lot better.

### How much better did your model preform after adding additional features and why do you think that is?
As I mentioned above, the model performed a lot better with additional time-features, which is due to the fact that the model has more information per instance.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
The model only performed marginally better with the hyperparameter optimization I did, but I think that is due to limited time I had to finish this project.

### If you were given more time with this dataset, where do you think you would spend more time?
I would definitely focus on optimizing hyperparameters more, because this can be real painstaking but will boost the models performance, if done correctly.
Additionally, I would try some other models and see how they perform, just to have some results to compare to.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

| model         |learning_rate|num_epochs|max_depth|score|
|---------------|----|----|----|-----|
| initial       |0.001|0.001|0.001|1.34025|
| add_features  |10|10|10|0.52779|
| hpo           |5|5|5|0.69743|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

## Summary
This project was a perfect example for how model performance can be optimized with EDA (feature engineering) and hyperparameter optimization. Especially hyperparameter optimization is crucial for finetuning a model and to get the last bit of performance out of it.
Hyperparameter optimization will normally take much more time, but in this case only one state of three hyperparameters was used.
