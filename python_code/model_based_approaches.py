import pandas as pd
import numpy as np
from python_code.data_prep.featureconstructor import FeatureConstructor
from sklearn.model_selection import train_test_split
from python_code.recommenders.evaluation import Evaluation

if __name__ == "__main__":
    review_ds = pd.read_csv('./data/reduced_reviews.csv', index_col=False)
    item_ds = pd.read_csv('./data/reduced_items.csv', index_col=False)
    user_ds = pd.read_csv('./data/reduced_users.csv', index_col=False)

    # Fit model-based recommenders:

    # Construct training set:
    seed = 1992
    balanced_train = FeatureConstructor.get_balanced_training_set(review_ds, 25000, seed)

    balanced_train = FeatureConstructor.add_distance(balanced_train, item_ds)
    balanced_train = FeatureConstructor.add_overall_popularity(balanced_train)
    # balanced_train = FeatureConstructor.friend_popularity(balanced_train, user_ds) takes too long to run
    balanced_train = FeatureConstructor.restaurant_type(balanced_train, item_ds)
    balanced_train = FeatureConstructor.is_elite(balanced_train, user_ds)

    norm_train = FeatureConstructor.to_train_matrix(balanced_train)
    y = balanced_train['has_reviewed'].to_numpy().astype(int)

    # Split data set:
    X_train, X_valid, y_train, y_valid = train_test_split(norm_train, y, test_size=0.2, random_state=42)

    np.save('./data/X_train.npy', X_train)
    np.save('./data/X_valid.npy', X_valid)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/y_valid.npy', y_valid)


