### Overview

This folder contains our LongSciVerify data set, and a pre-processed version of the [LongEval (Krishna et al., 2023)](https://aclanthology.org/2023.eacl-main.121) PubMed data set.


### LongSciVerify

The raw articles and summaries from the PubMed and ArXiv data sets included in this study can be found in the `./data/raw_data/LongSciVerify` in this repo. 

The human annotations (one csv per expert annotator) can be found in `./data/human_eval_results/LongSciVerify`.

The mappings of the model ids to the models used to generate the abstractive summaries are as follows. Details can be found in the appendix of the [paper](https://aclanthology.org/2024.lrec-main.941/).
```
model_1 = GenCompareSum-abs
model_2 = DANCERSumm
model_3 = ZONESumm 
```

### LongEval

We have pre-processed the LongEval PubMed data set to made it work with our scripts. This data was originally published as part of [LONGEVAL: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization by Krishna et al., (2023)](https://aclanthology.org/2023.eacl-main.121/). Links to their original data can be found [here](https://github.com/martiansideofthemoon/longeval-summarization). The  Similarly to LongSciVerify, the raw articles can be found in `./data/raw_data/LongEval` and the human annotations in `./data/human_eval_results/LongEval`.


### Inter-Annotator Agreement calculations

A notebook to calculate the IAA within the data sets can be found in `./evaluation_scripts/IAA_datasets.ipynb`.
