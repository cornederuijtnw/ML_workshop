import pandas as pd
import numpy as np


class BaselineRecommenders:

    @staticmethod
    def popular_baseline(training_set, val_users, list_size=250):
        most_popular = training_set['business_id'].value_counts().reset_index().\
            rename(columns={'index': 'business_id', 'business_id': 'freq'}).head(list_size)

        most_popular['rank'] = np.arange(list_size)

        all_users = pd.Series(val_users, name='user_id').to_frame()
        all_users['tmp'] = 1
        most_popular['tmp'] = 1

        recommendation = all_users.merge(most_popular, how='outer', on=['tmp'])

        recommendation = recommendation.drop(['tmp', 'freq'], axis=1)

        return recommendation

    @staticmethod
    def location_baseline(training_set, val_users, items_ds, list_size=250):
        # Find most likely city for all users

        items_ds['city'] = items_ds['city'].str.lower()

        train_users = training_set['user_id'].unique()

        # Compute most likely city per user
        user_city = pd.DataFrame(columns=['user_id', 'city'])

        training_set = training_set.set_index('business_id').join(items_ds.loc[:, ['business_id', 'city']].
                                                                  set_index('business_id'), on='business_id').reset_index()

        for user in val_users:
            if user in train_users:
                most_likely_city = training_set[training_set['user_id'] == user]['city'].value_counts().index[0]
            else:
                most_likely_city = 'most_pop'
            u_c = pd.DataFrame.from_dict({'user_id': [user], 'city': [most_likely_city]})
            user_city = pd.concat([user_city, u_c], ignore_index=True)

        # Compute most popularity per city:
        city_pop = training_set.groupby('city')['business_id'].value_counts().rename().reset_index(). \
                rename(columns={0: 'freq'}).groupby('city').head(list_size)

        # Is already sorted
        overall_pop = training_set['business_id'].value_counts().reset_index().rename(columns={'index': 'business_id',
                                                                                               'business_id': 'freq'})

        city_recommendation = pd.DataFrame(columns=['city', 'business_id', 'rank'])

        unique_cities = user_city['city'].unique()

        # find the top 250 for each city, if a city has less businesses then fill with most popular overall
        for city in unique_cities:
            if city == 'most_pop':
                cur_city_rec = overall_pop.head(250)
                cur_city_rec['city'] = 'most_pop'
            else:
                city_top = city_pop[city_pop['city'] == city]
                business_to_add = list_size - city_top.shape[0]
                if business_to_add > 0:
                    city_top_ids = city_top['business_id'].unique()
                    remaining_rec = overall_pop[~overall_pop['business_id'].isin(city_top_ids)]
                    cur_city_rec = pd.concat([city_top['business_id'], remaining_rec['business_id']], ignore_index=True).\
                        head(list_size).to_frame()
                    cur_city_rec['city'] = city
                else:
                    cur_city_rec = city_top

            cur_city_rec['rank'] = np.arange(list_size)
            cur_city_rec = cur_city_rec.loc[:, ['city', 'business_id', 'rank']]
            city_recommendation = pd.concat([city_recommendation, cur_city_rec], ignore_index=True, sort=False)

        # Create the recommendation list
        temp_rank_pd = pd.DataFrame.from_dict({'tmp': np.repeat(1, list_size), 'rank': np.arange(list_size)})
        user_city['tmp'] = 1
        pop_loc_recommendation = user_city.merge(temp_rank_pd, how='outer', on=['tmp'])

        pop_loc_recommendation = \
            pop_loc_recommendation.set_index(['city', 'rank']).join(city_recommendation.set_index(['city', 'rank']),
                                                                on=['city', 'rank']).reset_index()

        pop_loc_recommendation = pop_loc_recommendation.drop(['city', 'tmp'], axis=1)
        pop_loc_recommendation = pop_loc_recommendation.loc[:, ['user_id', 'business_id', 'rank']]

        return pop_loc_recommendation



