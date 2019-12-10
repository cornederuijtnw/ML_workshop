import pandas as pd
import numpy as np
from python_code.data_prep.load_and_split import DataManipulations
from python_code.recommenders.baseline_recommenders import BaselineRecommenders
from python_code.recommenders.evaluation import Evaluation

if __name__ == "__main__":

    prep_ds = False

    user_treshold = 15
    item_treshold = 15

    review_loc = './data/original_ds/yelp_review.csv'
    item_loc = './data/original_ds/yelp_business.csv'
    user_loc = './data/original_ds/yelp_user.csv'

    if prep_ds:
        review_ds = DataManipulations.split_train_test(review_loc, item_loc)

        print("Original review size: " + str(review_ds.shape[0]))

        item_ds = pd.read_csv("./data/original_ds/yelp_business.csv")
        user_ds = pd.read_csv("./data/original_ds/yelp_user.csv")

        review_ds, user_ds, item_ds = DataManipulations.remove_infrequent_users_items(review_ds, user_ds, item_ds,
                                                                                      user_treshold, item_treshold,
                                                                                      sample_size=100000)

        review_ds.to_csv('./data/reduced_reviews.csv', index=False)
        item_ds.to_csv('./data/reduced_items.csv', index=False)
        user_ds.to_csv('./data/reduced_users.csv', index=False)

    else:
        review_ds = pd.read_csv('./data/reduced_reviews.csv', index_col=False)
        item_ds = pd.read_csv('./data/reduced_items.csv', index_col=False)
        user_ds = pd.read_csv('./data/reduced_users.csv', index_col=False)

    # Run recommenders:
    validation_users = pd.read_csv('./data/validation_users.csv', index_col=False).to_numpy()
    validation_users = validation_users.reshape(validation_users.shape[0])

    # Run benchmarks:
    loc_benchmark = BaselineRecommenders.location_baseline(review_ds, validation_users, item_ds)
    most_pop_benchmark = BaselineRecommenders.popular_baseline(review_ds, validation_users)
       

    # Evaluate:
    validation_dat = pd.read_csv('./data/full_validation.csv', index_col=False)

    roc_values_loc = Evaluation.compute_ROC(loc_benchmark, validation_dat, 250)
    roc_values_most_pop = Evaluation.compute_ROC(most_pop_benchmark, validation_dat, 250)

    auc_values_loc = Evaluation.compute_AUC(roc_values_loc)
    auc_values_most_pop = Evaluation.compute_AUC(roc_values_most_pop)

    auc_values_loc.to_csv('./results/val_loc_benchmark.csv', index=False)
    auc_values_most_pop.to_csv('./results/val_most_pop.csv', index=False)

