import numpy as np
import pandas as pd
import scipy
from copy import deepcopy
import re
import os

tmp_csv = "output.csv"


def fleiss_kappa(M):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = (
        category_sum / tot_annotations
    )  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = (
        np.sum(P) / N
    )  # add all observed agreement chances per item and divide by amount of items

    return round_2dp(round((Pbar - PbarE) / (1 - PbarE), 4))


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a - b) ** 2


def ratio_metric(a, b):
    return ((a - b) / (a + b)) ** 2


def round_2dp(a):
    return np.round(a, 2)


def krippendorff_alpha(
    data,
    metric=interval_metric,
    force_vecmath=False,
    convert_items=float,
    missing_items=None,
):
    """
    Calculate Krippendorff's alpha (inter-rater reliability):

    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items

    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    """

    # number of coders
    m = len(data)

    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)

    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)

        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))

    units = dict(
        (it, d) for it, d in units.items() if len(d) > 1
    )  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values

    if n == 0:
        raise ValueError("No items to compare.")

    np_metric = (np is not None) and (
        (metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath
    )

    Do = 0.0
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du / float(len(grades) - 1)
    Do /= float(n)

    if Do == 0:
        return 1.0

    De = 0.0
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n * (n - 1))

    return round_2dp(1.0 - Do / De if (Do and De) else 1.0)


def get_df(df):
    cols = [col for col in df.columns if col.startswith("model_")]
    metrics = set([re.sub("model_[0-9]_", "", col) for col in cols])
    df_new = pd.DataFrame(columns=list(metrics))
    for metric in metrics:
        cols_for_metric = [col_name for col_name in df.columns if metric in col_name]
        scores = []
        for col_for_metric in cols_for_metric:
            scores.append(list(df[col_for_metric]))
        scores = [item for sublist in scores for item in sublist]
        df_new.loc[:, metric] = scores
    df_new = df_new.reset_index(drop=True)
    df_new.to_csv(tmp_csv, index=None)

    return


def kendal_tau_matrix(df):
    df_new = get_df(df)
    df_new = pd.read_csv(tmp_csv)
    df_new = df_new.rename(
        columns={
            "ldfacts_src_hyp": "LongDocFACTScore",
            "rouge_rouge_1_f_score": "ROUGE-1",
            "bertscore": "BERTScore",
            "factuality": "Human",
            "rouge_rouge_2_f_score": "ROUGE-2",
            "bartscore_src_hyp": "BARTScore",
            "rouge_rouge_l_f_score": "ROUGE-L",
            "questeval": "QuestEval",
            "factcc": "FactCC",
        }
    )
    df_new = df_new.drop(columns=["coherence", "fluency"])
    cols = [
        "Human",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "BERTScore",
        "QuestEval",
        "FactCC",
        "BARTScore",
        "LongDocFACTScore",
    ]
    df_new = df_new[cols]
    columns = df_new.columns
    kendalltau_matrix = np.zeros(shape=(len(columns), len(columns)))
    for idx1, col_n_1 in enumerate(df_new.columns):
        for idx2, col_n_2 in enumerate(df_new.columns):
            df_new_2 = deepcopy(df_new)
            col_1 = np.array(df_new_2[col_n_1])
            col_2 = np.array(df_new_2[col_n_2])
            kendalltau_matrix[idx1, idx2] = np.round(
                scipy.stats.kendalltau(col_1, col_2)[0], 2
            )  # 0 is the correlation
    os.remove(tmp_csv)
    return df_new, kendalltau_matrix
