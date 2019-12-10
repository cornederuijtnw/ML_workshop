import pandas as pd
import numpy as np
from sklearn import metrics

class Evaluation:

    @staticmethod
    def compute_AUC(roc_curve):
        AUC = np.zeros(roc_curve.shape[0])

        i = 1
        for k in roc_curve[roc_curve['k'] > 0]['k'].unique():
            cur_dat = roc_curve[roc_curve['k'] <= k]
            AUC[i] = metrics.auc(cur_dat['fpr'], cur_dat['recall'])
            i+=1
        roc_curve['AUC'] = AUC

        return roc_curve

    @staticmethod
    def compute_ROC(recommendation, ground_truth, max_K):
        res = pd.DataFrame(columns=['k', 'fpr', 'recall'])

        for k in range(0, max_K, 10):
            print("Computing for k:" + str(k))
            recall, fpr = Evaluation._get_rates(recommendation, ground_truth, k)
            res = pd.concat([res, pd.DataFrame.from_dict({'k': [k], 'fpr': [fpr], 'recall': [recall]})])

        return res

    @staticmethod
    def _get_rates(recommendation, ground_truth, K):
        recommendation_K = recommendation[recommendation['rank'] <= K]
        recommendation_K = recommendation_K.drop(['rank'], axis=1)

        recommendation_K['in_rec'] = 1

        ground_truth = ground_truth.loc[:, ['user_id', 'business_id']]

        recall = Evaluation._comp_recall(recommendation_K, ground_truth)
        fpr = Evaluation._comp_fpr(recommendation_K, ground_truth)

        return recall, fpr

    @staticmethod
    def _comp_recall(recommendation_K, ground_truth):
        tp_set = \
            ground_truth.loc[:, ['user_id', 'business_id']].set_index(['user_id', 'business_id']).\
                join(recommendation_K.set_index(['user_id', 'business_id']), on=['user_id', 'business_id']).reset_index()

        tp_set['in_rec'] = tp_set['in_rec'].fillna(0)

        tp_set = \
            tp_set.groupby(['user_id']).agg({'in_rec': 'sum', 'business_id': 'count'}).reset_index().\
                rename(columns={'business_id': 'n_positives'})

        tp_set['recall'] = tp_set['in_rec'] / tp_set['n_positives']

        recall = tp_set['recall'].mean()

        return recall

    @staticmethod
    def _comp_fpr(recommendation_K, ground_truth):
        gt_per_user = ground_truth['user_id'].value_counts().reset_index().rename(columns={'index': 'user_id',
                                                                                           'user_id': 'positives'})

        all_valid_items = ground_truth['business_id'].nunique()

        ground_truth['in_gt'] = 1

        fp_set = \
            recommendation_K.loc[:, ['user_id', 'business_id']].set_index(['user_id', 'business_id']).\
            join(ground_truth.set_index(['user_id', 'business_id']), on=['user_id', 'business_id']).reset_index()

        fp_set['in_gt'] = fp_set['in_gt'].fillna(0)
        fp_set['n_in_gt'] = 1 - fp_set['in_gt']
        fp_set = fp_set.drop(['in_gt'], axis=1)

        fp_set = fp_set.groupby('user_id').agg({'n_in_gt': 'sum'}).reset_index().set_index('user_id').\
            join(gt_per_user.set_index('user_id'), on='user_id').reset_index()

        fp_set['fpr'] = fp_set['n_in_gt'] / (all_valid_items - fp_set['positives'])

        fpr = fp_set['fpr'].mean()

        return fpr
