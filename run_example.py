from src.ldfacts import LDFACTS

text_1 = " On 24th October, there were 120 students attending the class about maths."
text_2 = "On 23rd October, there were 120 students attending the class about physics."


with open('./example_text_file.txt') as f:
    source_text = f.read()

ldfacts_scorer = LDFACTS(device='cpu')

print(f"\nThe source text is: \n\n{source_text}\n")

scores = ldfacts_scorer.score_src_hyp_long([source_text,source_text],[text_1,text_2])

print(f"\nSummary 1 is: {text_1} The score is: {scores[0]}\n")
print(f"\nSummary 2 is: {text_2} The score is: {scores[1]}\n")
