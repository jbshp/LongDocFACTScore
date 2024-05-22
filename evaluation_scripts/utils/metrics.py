# from pyrouge import Rouge155
import time
import shutil
import os
from IPython.utils import io
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import evaluate
import sys
from bert_score.scorer import BERTScorer
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk
from nltk.translate import meteor
from nltk import word_tokenize

sys.path.append("./evaluation_scripts/QuestEval/")
from questeval.questeval_metric import QuestEval

nltk.download("punkt")
nltk.download("wordnet")

questeval = QuestEval(no_cuda=False)


def test_huggingface_rouge(predicted_summaries, gold_summaries):
    metric = evaluate.load("rouge")
    results = metric.compute(predictions=predicted_summaries, references=gold_summaries)
    return {
        "rouge_1_f_score": ss(results["rouge1"]),
        "rouge_2_f_score": ss(results["rouge2"]),
        "rouge_l_f_score": ss(results["rougeL"]),
    }


def ss(x):
    return np.round(x * 100, decimals=2)


def test_questeval(hypothesis, sources):
    return list(
        questeval.corpus_questeval(hypothesis=hypothesis, sources=sources)[
            "ex_level_scores"
        ]
    )


def test_meteor(predicted_summaries, gold_summaries):
    """
    Calculate METEOR metric fro predicted and gold summaries
    """
    scores = []
    for g, p in zip(gold_summaries, predicted_summaries):
        a = [word_tokenize(g)]
        b = word_tokenize(p)
        scores.append(meteor(a, b))
    return scores


bert_scorer = BERTScorer("bert-base-uncased", device="cpu")


def test_bert_score(predicted_summaries, gold_summaries):
    with io.capture_output() as captured:
        P_sci, R_sci, F1_sci = bert_scorer.score(predicted_summaries, gold_summaries)
    return F1_sci


def test_frugal_score(
    predicted_summaries,
    gold_summaries,
    device,
    batch_size=32,
    pretrained_model_name_or_path="moussaKam/frugalscore_medium_bert-base_bert-score",
):
    with io.capture_output() as captured:

        def tokenize_function(data, max_length=512):
            return tokenizer(
                data["sentence1"],
                data["sentence2"],
                max_length=max_length,
                truncation=True,
                padding=True,
            )

        # load models
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if device == "cpu":
            fp16 = False
        else:
            fp16 = True
        training_args = TrainingArguments(
            "trainer",
            fp16=fp16,
            per_device_eval_batch_size=batch_size,
            report_to=None,
            no_cuda=(device == "cpu"),
            log_level="warning",
        )
        trainer = Trainer(model, training_args, tokenizer=tokenizer)

        # tokenize data
        dataset = {"sentence1": predicted_summaries, "sentence2": gold_summaries}
        raw_datasets = datasets.Dataset.from_dict(dataset)
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2"]
        )

        # make predictions
        predictions = trainer.predict(tokenized_datasets)

    return list(predictions.predictions.squeeze(-1))


def format_rouge_results(results):
    return f"{results['rouge_1_f_score']}/{results['rouge_2_f_score']}/{results['rouge_l_f_score']}"


def rounded_mean(array):
    return np.round(float(np.mean(array)), decimals=4) * 100


def calulate_all_metrics(our_predictions, gold_summaries, device, wandb_log=False):
    with io.capture_output() as captured:
        our_pred = test_rouge(our_predictions, gold_summaries)
        hf_rouge_pred = test_huggingface_rouge(our_predictions, gold_summaries)
        meteor_pred = test_meteor(our_predictions, gold_summaries)
        bert_score_pred = test_bert_score(our_predictions, gold_summaries, device)
        # frugal_score_pred = test_frugal_score(our_predictions,gold_summaries, device)
        meteor_pred = test_meteor(our_predictions, gold_summaries)
        dancersum = test_huggingface_rouge_dancersum_implementation(
            our_predictions, gold_summaries
        )
    if wandb_log:
        import wandb

        wandb_results = {
            "rouge_155": our_pred,
            "hf_rouge": hf_rouge_pred,
            "bert_score": rounded_mean(bert_score_pred),
        }
        wandb.log(wandb_results)
    print(
        f"""
    ROUGE-F results: (1/2/l)/ ROUGE-F results google-hf implementation (1/2/l)/ ROUGE-F dancersum implementation (1/2/l) / BERTScore/ METEOR:
    {format_rouge_results(our_pred)}/{format_rouge_results(hf_rouge_pred)}/{format_rouge_results(dancersum)}/{rounded_mean(bert_score_pred)}/{rounded_mean(meteor_pred)}
    """
    )
    return
