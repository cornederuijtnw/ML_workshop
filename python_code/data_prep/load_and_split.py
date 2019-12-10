import pandas as pd
import numpy as np
import math

from nltk.tokenize import RegexpTokenizer

class DataManipulations:

    @staticmethod
    def split_train_test(item_loc, review_loc, sample_size=-1):
        print("Loading data set")

        #TODO: Split in windows

        if sample_size > 0:
            reviews = pd.read_csv(review_loc, index_col=False, nrows=sample_size)
        else:
            reviews = pd.read_csv(review_loc, index_col=False)

        items_ds = pd.read_csv(item_loc, index_col=False)

        restaurant_ids = DataManipulations._get_restuarant_ids(items_ds)
        reviews = reviews[reviews['business_id'].isin(restaurant_ids)]

        reviews = reviews.loc[:, ['review_id', 'user_id', 'business_id', 'stars', 'date']]
        reviews['date'] = pd.to_datetime(reviews['date'])

        reviews_over_time = reviews.sort_values('date').groupby('date').size()
        reviews_over_time = (reviews_over_time / sum(reviews_over_time)).cumsum()

        print("Splitting into training, validation and test")
        # Keep 1% for testing:
        test_dates = reviews_over_time[reviews_over_time >= 0.99]
        validation_dates = reviews_over_time[(reviews_over_time >= 0.98) & (reviews_over_time < 0.99)]
        training_dates = reviews_over_time[reviews_over_time < 0.98]

        review_training = reviews[reviews['date'].isin(training_dates.index)]
        review_validation = reviews[reviews['date'].isin(validation_dates.index)]
        review_test = reviews[reviews['date'].isin(test_dates.index)]

        # Keep all users and business in the ds for now, just try avoid using feature which might give clues away
        print("storing data sets")
        review_training.to_csv("./data/full_training.csv", index=False)
        review_validation.to_csv("./data/full_validation.csv", index=False)
        review_test.to_csv("./data/full_test.csv", index=False)

        validation_users = pd.Series(review_validation['user_id'].unique(), name="user_id")
        test_users = pd.Series(review_test['user_id'].unique(), name="user_id")

        validation_users.to_csv("./data/validation_users.csv", index=False)
        test_users.to_csv("./data/test_users.csv", index=False)

        return review_training

    @staticmethod
    def remove_infrequent_users_items(review_training, users_ds, items_ds, user_treshold, item_treshold,
                                      only_keep_restaurants=True, sample_size=-1):
        n_original_users = review_training['user_id'].nunique()
        n_original_items = review_training['business_id'].nunique()
        n_reviews = review_training.shape[0]

        if only_keep_restaurants:
            restaurant_ids = DataManipulations._get_restuarant_ids(items_ds)
            review_training = review_training[review_training['business_id'].isin(restaurant_ids)]

        if 0 < sample_size < review_training.shape[0]:
            review_training = review_training.sample(n=sample_size, random_state=1)

        freq_users = review_training['user_id'].value_counts()
        freq_users = freq_users[freq_users > user_treshold]

        freq_items = review_training['business_id'].value_counts()
        freq_items = freq_items[freq_items > item_treshold]

        review_training = review_training[review_training['user_id'].isin(freq_users.index)]
        review_training = review_training[review_training['business_id'].isin(freq_items.index)]

        remaining_users = review_training['user_id'].unique()
        remaining_items = review_training['business_id'].unique()

        new_users_ds = users_ds[users_ds['user_id'].isin(remaining_users)]
        new_items_ds = items_ds[items_ds['business_id'].isin(remaining_items)]

        user_reduction = round((new_users_ds.shape[0] - n_original_users)/n_original_users * 100, 2)
        item_reduction = round((new_items_ds.shape[0] - n_original_items)/n_original_items * 100, 2)
        review_reduction = round((review_training.shape[0] - n_reviews)/n_reviews * 100, 2)

        print("user reduction: " + str(user_reduction) + ", item reduction: " + str(item_reduction) +
              "review reduction: " + str(review_reduction))

        return review_training, new_users_ds, new_items_ds

    @staticmethod
    def create_ds_preliminary_analysis(review_training, prel_user_sample):
        # sample 10K users for preliminary data analysis
        print("storing data set for preliminary analysis")
        np.random.seed(50)
        user_sample = np.random.choice(review_training['user_id'].unique(), prel_user_sample, replace=False)
        review_training_sample = review_training[review_training['user_id'].isin(user_sample)]
        review_training_sample.to_csv("./data/training_sample"+ str(prel_user_sample) + ".csv", index=False)

    @staticmethod
    def _get_restuarant_ids(item_ds):
        exp_item_cat = pd.concat([item_ds['business_id'], item_ds['categories'].str.split(';', expand=True)],
                                 ignore_index=True, axis=1).rename(columns={0: 'business_id'})

        exp_item_cat = exp_item_cat.melt(id_vars='business_id',
                                         var_name='cat_order',
                                         value_name='cat')

        restaurant_ids = exp_item_cat[exp_item_cat['cat'] == 'Restaurants']['business_id'].unique()

        return restaurant_ids

