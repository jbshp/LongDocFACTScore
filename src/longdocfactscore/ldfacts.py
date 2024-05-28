# Suppress annoying warnings from this issue which cannot be solved: https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md and transformers packages
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util

class LongDocFACTScore():
    def __init__(self, device="cuda:0", model="BARTScore"):
        self.sent_model = SentenceTransformer("bert-base-nli-mean-tokens")
        self.sent_model.to(device)
        self.device = device
        if model == "BARTScore":
            self.metric = BARTScore(device=self.device)
            self.metric_function = self.metric.bart_score
        else:
            raise ValueError("LongDocFACTScore currently only supports BARTScore")


    def get_surrounding_sentences(self, sentence_array, ii):
        if ii > 0 and ii < len(sentence_array) - 1:
            sents = " ".join(np.array(sentence_array)[ii - 1 : ii + 1])
        elif ii == 0:
            sents = " ".join(np.array(sentence_array)[:2])
        elif ii == len(sentence_array) - 1:
            sents = " ".join(np.array(sentence_array)[ii - 1 :])
        return sents

    def group_into_sections(self, sentence_array, num_sent):
        sectioned_sents = []
        for ii in range(0, len(sentence_array), num_sent):
            sectioned_sents.append(" ".join(sentence_array)[ii : ii + num_sent])
        return sectioned_sents

    def score_src_hyp_long(self, srcs, hyps):
        all_scores = []
        # src is a list containing source documents.
        # hyps is a list containing predicted documents
        for src, hyp in zip(srcs, hyps):
            src_sents = sent_tokenize(src)
            sentence_embeddings_src = self.sent_model.encode(
                src_sents, show_progress_bar=False
            )
            doc_scores = []
            hyp_array = sent_tokenize(hyp)
            for idx, hyp_sentence in enumerate(hyp_array):
                # for each sentence in summary, calculate the most similar sentence in the source article
                sentence_embeddings_hyp = self.sent_model.encode(
                    hyp_sentence, show_progress_bar=False
                )
                scores = util.cos_sim(sentence_embeddings_hyp, sentence_embeddings_src)[
                    0
                ]
                sorted_idxs = np.argsort(-1 * scores)
                similar_src_sentences = []
                #  get sentences surrounding the most similar sentences in the source article
                for ii in sorted_idxs[0:3]:
                    similar_sents = self.get_surrounding_sentences(src_sents, ii)
                    similar_src_sentences.append(similar_sents)
                # calculate metric for 3 most similar sections of source article
                scores = self.metric_function(
                    similar_src_sentences,
                    [hyp_sentence for i in range(0, len(similar_src_sentences))],
                )
                # Take the max scoring section to use
                score = np.max(scores)
                doc_scores.append(score)

            # calculate average score over whole doc
            doc_score = np.mean(doc_scores)
            all_scores.append(doc_score)
        return all_scores


# code taken from https://github.com/neulab/BARTScore/blob/main/bart_score.py
class BARTScore():
    def __init__(self, device="cuda:0", checkpoint="facebook/bart-large"):
        # Set up model
        self.device = device
        self.max_length = 1024
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=1)
        

    def bart_score(self, srcs, tgts, batch_size=4):
        ### Taken from 
        """Score a batch of examples"""
        score_list = []

        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i : i + batch_size]
            tgt_list = tgts[i : i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    src_tokens = encoded_src["input_ids"].to(self.device)
                    src_mask = encoded_src["attention_mask"].to(self.device)

                    tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                    tgt_mask = encoded_tgt["attention_mask"]
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f"source: {src_list}")
                print(f"target: {tgt_list}")
                exit(0)
        return score_list
