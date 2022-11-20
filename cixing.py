import requests
import nltk
import numpy as np
response = requests.get("https://gist.githubusercontent.com/pp7777myku/d2597707854dd8d96db736bdedcce00b/raw/b29a9d860604e2fe8231ab38ad91538cc9e19a38/gistfile1.txt")
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
