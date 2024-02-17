from collections import defaultdict

from nltk.corpus import brown
import nltk

corpus_type = nltk.corpus.reader.tagged.CategorizedTaggedCorpusReader


def hmm(corpus: corpus_type):
    """
    This function is an implementation of various HMMs (Hidden Markov Models).

    1st HMM: ?
    - Split data set into training and test set with a ratio of 0.9:0.1
    - For each word, compute the tag that maximizes the posterior P(tag|word).

    :param corpus: the text on which the model is trained and tested.
    :return:
    """

    # Split data set
    train_ratio = 0.9
    train_len = round(train_ratio * len(corpus))
    train, test = corpus[train_len], corpus[train_len:]

    '''compute max posterior tag for each word'''
    # Define the vocabulary
    vocabulary = set([word for word, _ in corpus])

    # Calculate the frequency of each word in the corpus, and store them in a dictionary
    frequency_dict = defaultdict(lambda: 0)
    for word, _ in corpus:
        frequency_dict[word] += 1

    # Create a dictionary of words where each word is a dictionary of tags and their posteriors
    posterior_dict = defaultdict(lambda: defaultdict(lambda: 0))

    for word, tag in corpus:
        posterior_dict[word][tag] += 1/frequency_dict[word]

    # And extract max posterior tags
    max_post_tag = {word: max(posterior_dict[word]) for word in vocabulary}
    return max_post_tag


if __name__ == '__main__':
    # Download the brown corpus form nltk, and extract from it the "news" text.
    news_corpus = brown.tagged_words(categories='news')
    hmm(news_corpus)
