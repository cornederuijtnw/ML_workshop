import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier

class ModelBasedRecommender:

    def __init__(self):
        pass

    def fit_logistic_regression(self, X, y):
        X = review_train_logged

        y = review_train['has_reviewed'].to_numpy().astype(int)

        model = RidgeClassifier().fit(X, y)
        model.score(X, y) # 0.68 accuracy

        model.get_params()