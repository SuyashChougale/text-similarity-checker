import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import string

# Download NLTK data for sentence tokenization and word lemmatization
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the Porter Stemmer and WordNet Lemmatizer
stemmer = PorterStemmer()
lemmer = WordNetLemmatizer()

def preprocess(text1, text2):
    # Convert text to lowercase
    text1 = text1.lower()
    text2 = text2.lower()

    # Tokenize sentences in text1
    sents1 = sent_tokenize(text1)
    for i in range(len(sents1)):
        # Remove punctuation from each sentence
        sents1[i] = sents1[i].translate(str.maketrans('', '', string.punctuation))

        # Tokenize words in each sentence
        words = word_tokenize(sents1[i])

        # Stem and lemmatize words in each sentence
        for j in range(len(words)):
            words[j] = stemmer.stem(words[j])
            words[j] = lemmer.lemmatize(words[j])

        # Reconstruct the sentence after preprocessing
        sents1[i] = ' '.join(words)

    # Tokenize sentences in text2 and perform the same preprocessing
    sents2 = sent_tokenize(text2)
    for i in range(len(sents2)):
        sents2[i] = sents2[i].translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(sents2[i])
        for j in range(len(words)):
            words[j] = stemmer.stem(words[j])
            words[j] = lemmer.lemmatize(words[j])
        sents2[i] = ' '.join(words)

    return sents1, sents2

def check_similarity(text1, text2):
    # Preprocess the input texts
    a, b = preprocess(text1, text2)

    # Combine preprocessed sentences for TF-IDF vectorization
    c = []
    c.extend(a)
    c.extend(b)

    # Create a TF-IDF vectorizer and calculate vectors for the sentences
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(c)
    vec1 = tfidf.transform(a)
    vec2 = tfidf.transform(b)

    # Calculate the cosine similarity between the vectors
    similarity = cosine_similarity(vec1, vec2)
    return similarity[0][0]
