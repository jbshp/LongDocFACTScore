import json


def get_raw_data(filepath):
    with open(filepath) as f:
        data = f.readlines()
    docs = [json.loads(doc) for doc in data]
    return docs


def clean_sent(sentence):
    sentence = sentence.replace("<S>", "")
    sentence = sentence.replace("</S>", "")
    sentence = sentence.replace("<pad>", "")
    sentence = sentence.replace("<br />", "")
    return sentence
    
def clean_abstract(text_array):
    if type(text_array)==str:
        cleaned = clean_sent(text_array)
    else:
        cleaned = ""
        for sentence in text_array:
            sentence = clean_sent(sentence)
            cleaned += f" {sentence} "
    return cleaned
