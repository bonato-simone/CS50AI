import nltk
import sys
import glob
import os.path
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    res = {}
    for path in glob.glob(os.path.join(directory, "*.txt")):
        with open(path, encoding="utf-8") as f:
            data = f.read()
            filename = os.path.basename(path)
            res[filename] = data
    return res


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = document.translate(str.maketrans('', '', string.punctuation)).lower()
    return [x for x in nltk.word_tokenize(document) if x not in nltk.corpus.stopwords.words("english")]


def compute_idfs(documents: dict):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    for document_words in documents.values():
        words |= set(document_words)

    res = {}
    for word in words:
        res[word] = 0
        for document_words in documents.values():
            if word in document_words:
                res[word] += 1

    documents_number = len(documents)
    for word in words:
        res[word] = math.log(documents_number / res[word])
    return res


def top_files(query: set, files: dict, idfs: dict, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    res = []
    for file in files:
        tfidf_sum = 0
        for word in query:
            if word in files[file]:
                tfidf_sum += files[file].count(word) * idfs[word]
        res.append([file, tfidf_sum])

    return [x for x, _ in sorted(res, key=lambda pair: pair[1], reverse=True)][:n]


def top_sentences(query: set, sentences: dict, idfs: dict, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    res = []
    for sentence, sentence_words in sentences.items():
        idf_sum = 0
        for word in query:
            if word in sentence_words:
                idf_sum += idfs[word]
        words_in_query = 0
        for word in sentence_words:
            if word in query:
                words_in_query += 1
        res.append([sentence, (idf_sum, words_in_query / len(sentence_words))])

    return [x for x, _ in sorted(res, key=lambda pair: pair[1], reverse=True)][:n]


if __name__ == "__main__":
    main()
