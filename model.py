import pandas as pd



from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import punkt

# Remove puncuations
# lowercase
# lemmatization
# stemming
from nltk.stem import PorterStemmer,WordNetLemmatizer
stemmer = PorterStemmer()
lemmer = WordNetLemmatizer()
import string
def preprocess(text1,text2):
    text1 = text1.lower()
    sents1= sent_tokenize(text1)
    for i in range(len(sents1)):
        sents1[i] = sents1[i].translate(str.maketrans('','',string.punctuation))
        words = word_tokenize(sents1[i])
        for j in range(len(words)):
            words[j] = stemmer.stem(words[j])
            words[j] = lemmer.lemmatize(words[j])
        sents1[i] = ' '.join(words)
    text2 = text2.lower()
    sents2= sent_tokenize(text2)
    for i in range(len(sents2)):
        sents2[i] = sents2[i].translate(str.maketrans('','',string.punctuation))
        words = word_tokenize(sents1[i])
        for j in range(len(words)):
            words[j] = stemmer.stem(words[j])
            words[j] = lemmer.lemmatize(words[j])
        sents1[i] = ' '.join(words)
    return sents1,sents2




    
def check_similarity(text1,text2):
    a,b = preprocess(text1,text2)
    c = []
    c.extend(a)
    c.extend(b)
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(c)
    vec1 = tfidf.transform(a)
    vec2 = tfidf.transform(b)
    similarity = cosine_similarity(vec1,vec2)
    return (similarity[0][0])




