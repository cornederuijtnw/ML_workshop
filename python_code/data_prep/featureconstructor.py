import pandas as pd
import numpy as np
import numpy.random as rand
from sklearn import preprocessing


class FeatureConstructor:

    @staticmethod
    def to_train_matrix(review_train):
        keep_cols = ['euc_dist_from_centroid', 'business_pop', 'has_been_elite', 'cat', 'has_reviewed']
        numeric_cols = ['euc_dist_from_centroid', 'business_pop', 'has_been_elite']

        # Replace missing values:
        review_train['euc_dist_from_centroid'] = review_train['euc_dist_from_centroid'].\
            fillna(review_train['euc_dist_from_centroid'].mean())
        review_train['euc_dist_from_centroid'] = review_train['business_pop'].\
            fillna(0)
        #review_train['friend_pop'] = review_train['friend_pop'].\
         #   fillna(0)
        review_train['has_been_elite'] = review_train['has_been_elite'].\
            fillna(0)
        review_train['cat'] = review_train['cat'].\
            replace('', 'other').\
            fillna('other')

        # Store normalization constants:
        norm_table = review_train.loc[:, numeric_cols].agg(['mean', 'std', 'min', 'max'])
        norm_table.to_csv('./models/norm_const.csv', index=False)

        # Take log of count variables and scale:
        scaler = preprocessing.MinMaxScaler()

        review_train_logged = \
            scaler.fit_transform(review_train.loc[:, ['euc_dist_from_centroid', 'business_pop']].\
                apply(lambda x: np.log(x+1)))

        review_train_logged = np.hstack((review_train_logged, review_train['has_been_elite'].to_numpy().reshape(-1, 1)))

        cat_dummy = pd.get_dummies(review_train['cat'], drop_first=True).to_numpy()

        review_train_logged = np.hstack((review_train_logged, cat_dummy))

        return review_train_logged

    @staticmethod
    def get_balanced_training_set(reviews_ds, sample_size=-1, seed=1992):
        balanced_training = pd.DataFrame(columns=['user_id', 'business_id', 'has_reviewed'])

        last_review_id = 0

        rand.seed(seed)

        if sample_size > 0:
            reviews_ds = reviews_ds.sample(sample_size, random_state=1)

        unique_users = reviews_ds['user_id'].unique()

        i = 1
        for u in unique_users:
            if i % 100 == 0:
                print('user ' + str(i) + ' of ' + str(len(unique_users)))
            user_dat = reviews_ds[reviews_ds['user_id'] == u]
            other_reviews = reviews_ds[reviews_ds['user_id'] != u]
            n_new_items = user_dat.shape[0]

            # May include the same item twice:
            review_index_to_add = rand.randint(0, other_reviews.shape[0]-n_new_items, n_new_items)
            reviews_to_add = other_reviews[~other_reviews['review_id'].isin(user_dat['review_id'].unique())].\
                iloc[review_index_to_add, :]
            reviews_to_add = reviews_to_add.drop(['review_id', 'user_id'], axis=1)
            reviews_to_add['user_id'] = u
            reviews_to_add['review_id'] = np.arange(last_review_id, last_review_id + n_new_items)

            user_dat['has_reviewed'] = 1
            reviews_to_add['has_reviewed'] = 0
            new_train = pd.concat([user_dat, reviews_to_add], ignore_index=True)
            balanced_training = pd.concat([balanced_training, new_train], ignore_index=True)

            last_review_id += n_new_items
            i+=1

        return balanced_training

    @staticmethod
    def is_elite(review_train, user_ds):
        user_ds['has_been_elite'] = (user_ds['elite'] != 'None').astype(int)

        review_train = review_train.\
            set_index('user_id').\
            join(user_ds.loc[:, ['user_id', 'has_been_elite']].set_index(['user_id']), on='user_id').\
            reset_index()

        return review_train

    @staticmethod
    def restaurant_type(review_train, item_ds):
        item_ds['categories'] = item_ds['categories'].str.lower()
        item_ds['categories'] = item_ds['categories'].str.replace('restaurants;', '')
        exp_item_cat = pd.concat([item_ds['business_id'], item_ds['categories'].str.split(';', expand=True)],
                                 ignore_index=True, axis=1).rename(columns={0: 'business_id'})

        exp_item_cat = exp_item_cat.melt(id_vars='business_id',
                                         var_name='cat_order',
                                         value_name='cat')

        # Only keep top category:
        exp_item_cat = exp_item_cat[exp_item_cat['cat_order'] == 1]
        exp_item_cat = exp_item_cat[~(exp_item_cat['cat'] == 'None')]

        top_10_cat_freq = exp_item_cat['cat'].value_counts().index[0:9].to_numpy()
        not_in_top10 = exp_item_cat['cat'][~exp_item_cat['cat'].isin(top_10_cat_freq)].unique()
        exp_item_cat['cat'] = exp_item_cat['cat'].replace(not_in_top10, 'other')

        exp_item_cat = exp_item_cat.drop(['cat_order'], axis=1)

        review_train = review_train.\
            set_index('business_id').\
            join(exp_item_cat.set_index('business_id'), on='business_id').\
            reset_index()

        return review_train

    @staticmethod
    def friend_popularity(review_train, users_ds):
        friends_per_user = pd.concat([users_ds['user_id'], users_ds['friends'].str.split(', ', expand=True)],
                                 ignore_index=True, axis=1).rename(columns={0: 'user_id'})

        friends_per_user = friends_per_user.melt(id_vars='user_id',
                                                 var_name='friends_order',
                                                 value_name='friend')
        friends_per_user = friends_per_user[~friends_per_user['friend'].isnull()]

        unique_users = review_train['user_id'].unique()

        friend_pop_df = pd.DataFrame(columns=['user_id', 'business_id', 'friend_pop'])

        print('Computing business popularity amongst friends')
        i = 1

        for u in unique_users:
            if i % 10 == 0:
                print('i: ' + str(i) + ' of ' + str(unique_users.shape[0]))
            friends = friends_per_user[friends_per_user['user_id'] == u]['friend'].to_numpy()
            business_pop_among_friends = \
                review_train[review_train['user_id'].isin(friends)]['business_id'].\
                value_counts().\
                reset_index().\
                rename(columns={'business_id': 'friend_pop',
                                'index': 'business_id'})
            business_pop_among_friends['user_id'] = u
            friend_pop_df = pd.concat([friend_pop_df, business_pop_among_friends], ignore_index=True)
            i += 1

        review_train = review_train.\
            set_index(['user_id', 'business_id']).\
            join(friend_pop_df.set_index(['user_id', 'business_id']), on=['user_id', 'business_id']).\
            reset_index()

        return review_train

    @staticmethod
    def add_overall_popularity(review_train):
        item_pop = review_train['business_id'].\
            value_counts().\
            reset_index().\
            rename(columns={'business_id': 'business_pop',
                            'index': 'business_id'})

        review_train = review_train.\
            set_index('business_id').\
            join(item_pop.set_index('business_id'), on='business_id').\
            reset_index()

        return review_train

    @staticmethod
    def add_distance_valid(valid_ds, items_ds):
        centroid = pd.read_csv('./models/centroids.csv', index_col=False)

        valid_ds = valid_ds.\
            set_index('business_id').\
            join(items_ds.loc[:, ['business_id', 'latitude', 'longitude']].set_index('business_id'), on='business_id').\
            reset_index()

        valid_ds = valid_ds.\
            set_index('user_id').\
            join(centroid.set_index('user_id'), on='user_id').reset_index()

        valid_ds['euc_dist_from_centroid'] = \
            np.sqrt((valid_ds['latitude'] - valid_ds['longitude']) ** 2 +\
                (valid_ds['longitude'] - valid_ds['centroid_longitude']) ** 2)

        return valid_ds

    @staticmethod
    def add_distance(review_train, item_ds):
        review_train = review_train.\
            set_index('business_id').\
            join(item_ds.loc[:, ['business_id', 'latitude', 'longitude']].set_index('business_id'), on='business_id').\
            reset_index()

        centroid = review_train.groupby('user_id').\
            agg({'latitude': 'mean',
                 'longitude': 'mean'}).\
            rename(columns={'latitude': 'centroid_latitude',
                            'longitude': 'centroid_longitude'})

        # centroid.reset_index().to_csv('./models/centroids.csv', index=False)

        review_train = review_train.\
            set_index('user_id').\
            join(centroid, on='user_id').reset_index()

        review_train['euc_dist_from_centroid'] = \
            np.sqrt((review_train['latitude'] - review_train['longitude']) ** 2 + \
                (review_train['longitude'] - review_train['centroid_longitude']) ** 2)

        return review_train

