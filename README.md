### Usage

To run on a piece of text:
```
from src.ldfacts import LDFACTS

predict_summary = "INSERT PREDICTED SUMMARY HERE"
src_doc = "INSERT SOURCE DOCUMENT HERE"

ldfacts_scorer = LDFACTS(device='cpu')

scores = ldfacts_scorer.score_src_hyp_long([src_doc],[predict_summary])
```

To run with some example data:
```bash
pip install -r requirements.txt
python run_example.py
```