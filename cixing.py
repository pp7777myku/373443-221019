import requests
import nltk
import numpy as np

response = requests.get("https://gist.githubusercontent.com/nzhukov/b66c831ea88b4e5c4a044c952fb3e1ae/raw/7935e52297e2e85933e41d1fd16ed529f1e689f5/A%2520Brief%2520History%2520of%2520the%2520Web.txt")
text = response.text
text_tokenized = nltk.tokenize.word_tokenize(text)
tagged_words = nltk.pos_tag(text_tokenized)
tags = [word_tag[1] for word_tag in tagged_words]
name_tag_group_map = {
    "Существительное": ['NN', 'NNP', 'NNPS', 'NNS'],
    "Прилагательное": ['JJ', 'JJR', 'JJS'],
    "Глаголы": ['VB','VBP', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    "Наречия": ["RBR", "RBS"],
    "Междометия": ['IN'],
    "Предлоги": ["PRP", "PRPS"],
}
name_count_map = []

for name in name_tag_group_map:
    tags_from_group = list(filter(lambda x : x in name_tag_group_map[name], tags))
    name_count_map.append((name, len(tags_from_group)))
name_count_map = np.array(name_count_map)
name_count_map = name_count_map[name_count_map[:, 1].argsort()]
print(name_count_map)


