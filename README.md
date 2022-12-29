# The region-based soybean yield prediction using stochastic, machine learning, and deep learning methods

This project looks into what is the most feasible strategy for soybean yield prediction. We used data combined from the USDA National Statistics Service, Economic Research Service, and NOAA climate data. In our estimation part, we compared three types of estimation models: classic statistical linear regression model (LR), machine learning random forest estimator (RF), and deep learning feed-forward neural network estimator (FFNN). We build the integral model with the whole dataset as well as the region-based model with sub-datasets separated by the USDA agricultural regions with each of the three estimators. We made model evaluations with a cross-validation method on all of our models. For each model, we calculated the model performance on the hold-out test dataset error by getting the mean average error (MAE), root-mean-square deviation (RMSE), and proportion of the variance explained by the model (R^2). We compared region-based estimation models vs. the integral estimation model to see whether the region-based estimation models can facilitate performance. We also compared all the region-based estimation models to see the model performance characteristics for each region. As a result, we noticed that region-based estimators are more accurate than the integral estimator. Among all the models, random forest regression performs the best with the region-based dataset; FFNN performs better on the prediction of the whole dataset but performs worst on the region-based dataset.


## Structure of the repository

### data
├──Soybean_Intensification.csv : dataset used

### scripts
┌──estimator.py
├──evaluator.py
├──my_utils.py
└──project.ipynb

- `estimator.py`: contains the script for establishing different classifiers
- `evaluator.py`: contains the script for evaluating results obtained with each classifier
- `my_utils.py`: contains the script for processing dataset
- `project.ipynb`: contains the main script for soybean yield prediction and visualization

### outputs

all tables and figures used in the article.

## Software Requirement

Python

## Contact

If you have any further questions or suggestions, please contact xuz218@wfu.edu,  langx19@wfu.edu, and lix419@wfu.edu.
