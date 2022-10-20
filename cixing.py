import requests
import nltk
import numpy as np
response = requests.get("https://gist.githubusercontent.com/nzhukov/b66c831ea88b4e5c4a044c952fb3e1ae/raw/7935e52297e2e85933e41d1fd16ed529f1e689f5/A%2520Brief%2520History%2520of%2520the%2520Web.txt")
text = response.text
text_tokenized = nltk.tokenize.word_tokenize(text)
tagged_words = nltk.pos_tag(text_tokenized)
tags = [word_tag[1] for word_tag in tagged_words]
cixing = {
    "Существительное": ['NN', 'NNP', 'NNPS', 'NNS'],
    "Прилагательное": ['JJ', 'JJR', 'JJS'],
    "Глаголы": ['VB','VBP', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    "Наречия": ["RBR", "RBS"],
    "Междометия": ['IN'],
    "Предлоги": ["PRP", "PRPS"],
}

count = []

for name in cixing:
    tag = list(filter(lambda x : x in cixing[name], tags))
    count.append((name, len(tag)))

count = np.array(count)
count = count[count[:, 1].argsort()]
print(count)
