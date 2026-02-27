from gensim import corpora
from gensim.models import LdaModel

def perform_lda(texts, num_topics=4):
    tokenized = [t.split() for t in texts if t]

    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=3, no_above=0.7)

    corpus = [dictionary.doc2bow(t) for t in tokenized]

    lda = LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=15
    )

    return lda, dictionary