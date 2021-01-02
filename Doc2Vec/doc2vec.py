

""" Source https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4 """





import re
import sklearn
from gensim.models import Doc2Vec
import numpy as np
import pandas as pd
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from tqdm import tqdm

df = pd.read_csv("/home/mayank/Downloads/pmeans/IIITH_Codemixed_new.txt",sep = "\t",names=["sentence","label"]).dropna()

#Data Cleaning
#Removing all the punctuation marks and converting to lowercase


def clean_str(string):
    """Tokenization/string cleaning for all datasets except for SST.

    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"[.,#!$%&;:{}=_`~()/\\]", "", string)
    return string.strip().lower()

def preprocessing(df):

    df['sentence'] = df['sentence'].astype(str).map(clean_str,na_action=None)
    df['label'] = df['label'].astype(str)
    
    
    """Removing most frequent words,rare words and stop words"""
    
    from collections import Counter
    cnt = Counter()
    for text in df["sentence"].values:
        for word in text.split():
            cnt[word] += 1
            
    cnt.most_common(10)
    
    FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
    
    def remove_freqwords(text):
        """custom function to remove the frequent words"""
        return " ".join([word for word in str(text).split() if word not in FREQWORDS])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_freqwords(text))
    
    n_rare_words = 10
    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
    
    def remove_rarewords(text):
        """custom function to remove the rare words"""
        return " ".join([word for word in str(text).split() if word not in RAREWORDS])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_rarewords(text))
    
    def remove_shorter_words(text):
      return " ".join([word for word in str(text).split() if len(word) >= 2])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_shorter_words(text))
    
    def remove_longer_words(text):
      """ remove words longer than 12 char """
      return " ".join([word for word in str(text).split() if len(word) <= 12])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_longer_words(text))
    
    def remove_words_digits(text):
      """ remove words with digits """
      return " ".join([word for word in str(text).split() if not any(c.isdigit() for c in word) ])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_words_digits(text))
    
    STOPWORDS = set(stopwords.words('english'))
    def remove_stopwords(text):
        """custom function to remove the stopwords"""
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_stopwords(text))
    return df


df = preprocessing(df)

# import Word2Vec loading capabilities
from gensim.models import KeyedVectors

# Creating the model
print("Loading the pre-trained embeddings")
embed_lookup = KeyedVectors.load_word2vec_format('/home/mayank/Downloads/GoogleNews-vectors-negative300.bin', 
                                                 binary=True)
print("DONE")

#store pretrained vocab
pretrained_words = []
for word in embed_lookup.vocab:
    pretrained_words.append(word)

row_idx = 1

# get word/embedding in that row
word = pretrained_words[row_idx] # get words by index
embedding = embed_lookup[word] # embeddings by word

# vocab and embedding info
print("Size of Vocab: {}\n".format(len(pretrained_words)))
print('Word in vocab: {}\n'.format(word))
print('Length of embedding: {}\n'.format(len(embedding)))
#print('Associated embedding: \n', embedding)

# print a few common words
for i in range(100):
    print(pretrained_words[i])

train, test = train_test_split(df, test_size=0.3, random_state=42)



def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r['sentence']), tags=[r.label]), axis=1)
test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['sentence']), tags=[r.label]), axis=1)

print(train_tagged.values[1])

model= Doc2Vec(dm=1, vector_size=300,min_count=1,window = 5,workers = 4,neagtive = 6,epochs = 2,sample=1e-4)
model.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model.alpha -= 0.002
    model.min_alpha = model.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors



y_train, X_train = vec_for_learning(model, train_tagged)
y_test, X_test = vec_for_learning(model, test_tagged)


X_train = np.array(X_train)
X_test = np.array(X_test)
Y_test = np.array(y_test)
Y_train = np.array(y_train)


from keras.utils import to_categorical

    

# Y_train = to_categorical(Y_train,num_classes=3)
# Y_test = to_categorical(Y_test,num_classes=3)
# X_train=X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
# X_test=X_test.reshape(X_test.shape[0],X_test.shape[1], 1)




 
# def train_classifier(X,y):
#     """ To perform grid search"""
#     param_grid = {'C': [0.1, 1, 10,100,1000],  
#               'gamma': [1, 0.1, 0.01], 
#               'kernel': ['rbf']} 
  
#     clf = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
#     clf.fit(X,y)
#     return clf 


# classifier = train_classifier(X_train,y_train)
# print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
# print (classifier.score(X_test,y_test))
# print(classifier.best_params_)


SVM = svm.SVC(kernel='rbf',C=1000, gamma=0.1)
SVM.fit(X_train,y_train)

pred_y = SVM.predict(X_test)

print(sklearn.metrics.classification_report(y_test, pred_y))






