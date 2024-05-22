import pandas as pd
import numpy as np
from utils.correlations import (
    fleiss_kappa,
    krippendorff_alpha,
    interval_metric,
    kendal_tau_matrix,
)
from utils.metrics import test_huggingface_rouge, test_questeval
import argparse
from longdocfactscore.ldfacts import LongDocFACTScore
from utils.preprocess import clean_abstract
from BARTScore.bart_score import BARTScorer
from bert_score.scorer import BERTScorer
import json
from nltk import sent_tokenize
import subprocess
import time
import os
import seaborn as sb
import torch


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def timeit(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res, func.__name__

    return wrapper


# load models
ldfacts_scorer = LongDocFACTScore(device=device)
bart_scorer = BARTScorer(checkpoint="facebook/bart-large", device=device)
bert_scorer = BERTScorer("bert-base-uncased", device=device)

# args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["pubmed_longdocfactscore", "arxiv_longdocfactscore", "pubmed_longeval"],
    help="data set to run evaluation on",
)
args = parser.parse_args()


def calculate_inter_annotator_agreement_factuality(data_frames):
    num_subjects = len(data_frames[0])
    model_nums = ["1", "2", "3"]
    sent_nums = ["1", "2", "3"]
    M = np.zeros(shape=(135, 2))

    # each annotator has a dataframe
    for df in data_frames:
        start_idx = 0

        #  we treat each model as a new sample
        for model_num in model_nums:
            # we treat each sentence as a new sample
            for sent_num in sent_nums:
                for idx, row in df.iterrows():
                    score = row[f"model_{model_num}_sent_{sent_num}_factuality"]
                    overall_idx = idx + start_idx
                    if np.isnan(score):
                        print("nan!")
                    else:
                        M[overall_idx, int(score)] += 1

                start_idx += num_subjects
    return fleiss_kappa(M)


def calculate_inter_annotator_agreement_fluency_coherence(data_frames, col="coherence"):
    num_subjects = len(df_1)
    num_categories = np.max(df_1[f"model_1_{col}"])
    model_nums = ["1", "2", "3"]
    M = np.zeros(shape=(num_subjects * len(model_nums), num_categories))
    for df in data_frames:
        start_idx = 0
        for model_num in model_nums:
            for idx, row in df.iterrows():
                rank = row[f"model_{model_num}_{col}"]
                M[idx + start_idx, rank - 1] += 1
            start_idx += num_subjects
    return M


def calculate_inter_annotator_agreement_fluency_coherence_krippendorff(
    data_frames, col="coherence"
):
    model_nums = ["1", "2", "3"]
    all_scores = []
    for df in data_frames:
        annotator_scores = []
        for model_num in model_nums:
            for idx, row in df.iterrows():
                rank = row[f"model_{model_num}_{col}"]
                annotator_scores.append(rank)
        all_scores.append(annotator_scores)
    return krippendorff_alpha(all_scores, interval_metric)


def calculate_inter_annotator_agreement_factuality_krippendorff(data_frames):
    model_nums = ["1", "2", "3"]
    sent_nums = ["1", "2", "3"]
    all_scores = []
    for df in data_frames:
        annotator_scores = []
        for model_num in model_nums:
            for sent_num in sent_nums:
                for idx, row in df.iterrows():
                    score = row[f"model_{model_num}_sent_{sent_num}_factuality"]
                    if not np.isnan(score):
                        annotator_scores.append(score)
        all_scores.append(annotator_scores)
    return krippendorff_alpha(all_scores, interval_metric)


def load_dataframes(dataset):
    if dataset == "arxiv_longdocfactscore":
        df_1 = pd.read_csv(
            "./data/human_eval_results/LongSciVerify/arxiv_reviewer_1.csv"
        )
        df_2 = pd.read_csv(
            "./data/human_eval_results/LongSciVerify/arxiv_reviewer_2.csv"
        )
        df_3 = pd.read_csv(
            "./data/human_eval_results/LongSciVerify/arxiv_reviewer_3.csv"
        )
        raw_path = "./data/raw_data/LongSciVerify/arxiv_test.json"
        return [df_1, df_2, df_3], raw_path
    if dataset == "pubmed_longdocfactscore":
        df_1 = pd.read_csv(
            "./data/human_eval_results/LongSciVerify/pubmed_reviewer_1.csv"
        )
        df_2 = pd.read_csv(
            "./data/human_eval_results/LongSciVerify/pubmed_reviewer_2.csv"
        )
        df_3 = pd.read_csv(
            "./data/human_eval_results/LongSciVerify/pubmed_reviewer_3.csv"
        )
        raw_path = "./data/raw_data/LongSciVerify/pubmed_test.json"
        return [df_1, df_2, df_3], raw_path
    if dataset == "pubmed_longeval":
        with open(
            "./data/raw_data/LongEval/pubmed_summary_src_doc_data.json", "r"
        ) as f:
            summaries = json.load(f)
        with open(
            "./data/human_eval_results/LongEval/pubmed_fine_scores.json", "r"
        ) as f:
            scores = json.load(f)
        results = []
        article_ids = set([key.split("_")[1] for key in summaries.keys()])
        summary_methods = ["bigbird_pegasus", "longt5"]
        for article_id in article_ids:
            result = {"id": article_id}
            for idx, summary_method in enumerate(summary_methods):
                article_key = f"article_{article_id}_{summary_method}"
                summary = summaries[article_key]["summary"]
                result[summary_method] = clean_abstract(summary)
                result[f"model_{idx}_factuality"] = np.mean(
                    [
                        score_array["score"]
                        for score_array in scores
                        if score_array["story"] == article_key
                    ][0]
                )
            result["article"] = clean_abstract(summaries[article_key]["source_doc"])
            result["ref"] = clean_abstract(
                summaries[f"article_{article_id}_human"]["summary"]
            )
            results.append(result)
        df = pd.DataFrame(results)
        return df


def average_scores_accross_annnotators(dfs):
    for idx, df in enumerate(dfs):
        for model in ["1", "2", "3"]:
            dfs[idx][f"model_{model}_factuality"] = (
                df[f"model_{model}_sent_1_factuality"]
                + df[f"model_{model}_sent_2_factuality"]
                + df[f"model_{model}_sent_3_factuality"]
            ) / 3
    df_all_annotators = dfs[0][
        ["id", "gencomparesumabs", "dancersum", "section_conditional"]
    ]
    cols = [
        "model_1_coherence",
        "model_2_coherence",
        "model_3_coherence",
        "model_1_fluency",
        "model_2_fluency",
        "model_3_fluency",
        "model_1_factuality",
        "model_2_factuality",
        "model_3_factuality",
    ]
    array = np.zeros(shape=(len(dfs[0]), len(cols)))
    for jj, col in enumerate(cols):
        for ii, row in dfs[0].iterrows():
            array[ii, jj] = (
                dfs[0].loc[ii, col] + dfs[1].loc[ii, col] + dfs[2].loc[ii, col]
            ) / 3
    scores = pd.DataFrame(array, columns=cols)
    df_all_annotators = df_all_annotators.join(scores)
    return df_all_annotators



def get_rouge_row(df, col_input, col_output, tgt_col):
    for idx, row in df.iterrows():
        rouge = test_huggingface_rouge([row[col_input]], [row[tgt_col]])
        for k, v in rouge.items():
            df.loc[idx, f"{col_output}_{k}"] = v
    return df


def get_bert_score_row(df, col_input, col_output, tgt_col):
    inputs = list(df[col_input])
    outputs = list(df[tgt_col])
    P_sci, R_sci, F1_sci = bert_scorer.score(inputs, outputs)
    df.loc[:, col_output] = F1_sci
    return df


def get_bartscore_row(df, col_input, col_output, tgt_col):
    inputs = list(df[col_input])
    outputs = list(df[tgt_col])
    df.loc[:, col_output] = np.array(bart_scorer.score(inputs, outputs))
    return df


def get_ldfacts_row(df, col_input, col_output, tgt_col):
    inputs = list(df[col_input])
    outputs = list(df[tgt_col])
    df.loc[:, col_output] = np.array(ldfacts_scorer.score_src_hyp_long(inputs, outputs))
    return df


def get_questeval_row(df, col_input, col_output, tgt_col):
    inputs = list(df[col_input])
    outputs = list(df[tgt_col])
    df.loc[:, col_output] = test_questeval(hypothesis=inputs, sources=outputs)
    return df


def get_factcc_row(df, col_input, col_output, tgt_col):
    return create_data_file_factcc(df, col_input, col_output)


def create_data_file_factcc(df, model, col_output):
    path = f"./data/formatted_for_factcc/{args.dataset}/{col_output}"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "data-dev.jsonl"), "w") as f:
        data_gen = []
        for ii, row in df.iterrows():
            pred_sents = sent_tokenize(row[model])
            for sent in pred_sents:
                # data needs to be in this format for factCC scripts - CORRECT will be ignores.
                dict_ = {
                    "claim": sent,
                    "id": row["id"],
                    "label": "CORRECT",
                    "text": row["article"],
                }
                data_gen.append(dict_)
                f.write(f"{json.dumps(dict_)}\n")
        f.close()
    print(path)
    subprocess.run(
        [
            "python",
            "src/factCC/modeling/run.py",
            "--task_name",
            "factcc_annotated",
            "--do_eval",
            "--eval_all_checkpoints",
            "--do_lower_case",
            "--max_seq_length",
            "512",
            "--per_gpu_train_batch_size",
            "12",
            "--model_type",
            "bert",
            "--model_name_or_path",
            "factcc-checkpoint",
            "--data_dir",
            path,
            "--output_dir",
            "./factcc-checkpoint",
            "--tokenizer_name",
            "bert-base-uncased",
            "--config_name",
            "bert-base-uncased",
            "--no_cuda",
        ]
    )
    with open(os.path.join(path, "data-dev.jsonl"), "r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    with open(os.path.join(path, "results.json"), "r") as f:
        results = json.load(f)
    for ii, row in enumerate(data):
        data[ii][col_output] = 1 - results[ii]  # in their results, 0 means 'entailed'
    df_results = pd.DataFrame(data)
    df_results = df_results[[col_output, "id"]].groupby("id").mean()
    df = df.join(df_results, on="id")
    return df


def load_json_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_raw_data(df, raw_path):
    data = load_json_data(raw_path)
    for idx, row in df.iterrows():
        article = [
            a
            for a in data
            if (a["article_id"] == row["id"].strip() or (row["id"] in a["article_id"]))
        ][0]
        df.loc[idx, "ref"] = clean_abstract(article["abstract_text"])
        df.loc[idx, "article"] = clean_abstract(article["article_text"])
    return df


def update_df_with_metric_scores(
    df,
    metric_function,
    src_cols=["gencomparesumabs", "dancersum", "section_conditional"],
    metric_name="rouge",
    tgt_col="ref",
    switch_tgt=False,
):
    print(f"calculating metrics for {metric_name}")
    output_cols = [f"model_{idx+1}_{metric_name}" for idx, model in enumerate(src_cols)]
    for src, output_col in zip(src_cols, output_cols):
        if switch_tgt == True:
            df = metric_function(df, tgt_col, output_col, src)
        else:
            df = metric_function(df, src, output_col, tgt_col)
    return df


def get_scores_from_metrics(df, src_cols):
    df = update_df_with_metric_scores(
        df, get_rouge_row, metric_name="rouge", src_cols=src_cols, tgt_col="article"
    )
    df = update_df_with_metric_scores(
        df,
        get_bert_score_row,
        metric_name="bertscore",
        src_cols=src_cols,
        tgt_col="article",
    )
    df = update_df_with_metric_scores(
        df,
        get_bartscore_row,
        metric_name="bartscore_src_hyp",
        tgt_col="article",
        switch_tgt=True,
        src_cols=src_cols,
    )
    df = update_df_with_metric_scores(
        df,
        get_ldfacts_row,
        metric_name="ldfacts_src_hyp",
        tgt_col="article",
        switch_tgt=True,
        src_cols=src_cols,
    )
    df = update_df_with_metric_scores(
        df,
        get_questeval_row,
        metric_name="questeval",
        tgt_col="article",
        switch_tgt=True,
        src_cols=src_cols,
    )
    df = update_df_with_metric_scores(
        df, get_factcc_row, metric_name="factcc", src_cols=src_cols, tgt_col="article"
    )
    return df


if __name__ == "__main__":

    if "longdocfactscore" in args.dataset:

        # load data
        dfs, raw_path = load_dataframes(args.dataset)

        # average scores across annotators for each example
        df_all = average_scores_accross_annnotators(dfs)

        # get raw data
        df_all = load_raw_data(df_all, raw_path)
        src_cols = ["gencomparesumabs", "dancersum", "section_conditional"]

    else:
        df_all = load_dataframes(args.dataset)
        src_cols = ["bigbird_pegasus", "longt5"]

    # get scores from metrics
    df_all = get_scores_from_metrics(df_all, src_cols)

    df_all.to_csv(f"{args.dataset}.csv", index=None)

    # get correlations
    df_new, kt = kendal_tau_matrix(df_all)

    plot = sb.heatmap(
        kt,
        cmap="Blues",
        annot=True,
        xticklabels=df_new.columns,
        yticklabels=df_new.columns,
        annot_kws={"fontsize": 8},
    )
    plot = plot.get_figure()
    plot.savefig(f"{args.dataset}_correlation.png", bbox_inches="tight")
