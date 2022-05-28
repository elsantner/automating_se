from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import pandas as pd
import numpy as np
import math
import joblib
import warnings

tokenizer_bert = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model_bert = AutoModel.from_pretrained("microsoft/codebert-base")

# Use a standard vectorization approach by creating a feature matrix.
# Loses semantic and structural information, but it is FAST (>> anything BERT embedding related)
# Pre-configured to use 750 features 
class FuncEncoderCountVectorizer:
    # df_data required for vocabulary creation
    def __init__(self, df_data=None, vocabulary_path=None):
        if (df_data is None and vocabulary_path is None):
            raise ValueError("df_data or vocabulary_path must be set.")
            
        if (vocabulary_path is not None):
            try:
                # load vocab from file
                vocab = self.__load_vocabulary(vocabulary_path)
                # Show warning if both params were set and loading succeeded
                if (df_data is not None and vocabulary_path is not None):
                    warnings.warn("Both df_data and vocabulary_path are set. Ignoring df_data.")
            except:
                # if file not found, try to extract from data ...
                if (df_data is not None):
                    warnings.warn("Loading vocabulary form '{0}' failed. Falling back to df_data.".format(vocabulary_path))
                    vocab = self.__extract_vocabulary(df_data)
                    if (vocabulary_path is not None):
                        self.__save_vocabulary(vocab, vocabulary_path)
                # ... otherwise fail
                else:
                    raise ValueError("Cannot load vocabulary form '{0}'.".format(vocabulary_path))
        else:
            vocab = self.__extract_vocabulary(df_data)
            if (vocabulary_path is not None):
                self.__save_vocabulary(vocab, vocabulary_path)
            
        
        # Create another CountVectorizer with the vocabulary from the CV above
        # (ensures to have consistent token features for test, training, validation)
        self.vectorizer = CountVectorizer(tokenizer = tokenizer_bert.tokenize,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             lowercase=False, \
                             vocabulary=vocab)
    
    def __load_vocabulary(self, vocabulary_path):
        return joblib.load(vocabulary_path)
        
    def __save_vocabulary(self, vocabulary, vocabulary_path):
        joblib.dump(vocabulary, vocabulary_path)
    
    def __extract_vocabulary(self, df_data):
        all_funcs = df_data['processed_func'].tolist()
        
        # Create a count vectorizer on the whole dataset and include the most occuring n tokens
        # tokenizer takes CodeBERT as function
        newVec = CountVectorizer(tokenizer = tokenizer_bert.tokenize,    \
                                     preprocessor = None, \
                                     stop_words = None,   \
                                     max_features=1500, \
                                     lowercase=False)
        newVec.fit_transform(all_funcs)
        return newVec.vocabulary_
    
    # Return feature matrix for list_of_strings
    # columns ... features
    # index ... row index of input list
    def __vectorizer_encode(self, list_of_strings):
        features = self.vectorizer.fit_transform(list_of_strings)
        return pd.DataFrame(features.toarray(), columns=self.vectorizer.get_feature_names_out())

    # Return a df of features
    def encode(self, list_funcs):
        # get feature matrix (as a df)
        df_enc_funcs = self.__vectorizer_encode(list_funcs)
        # return df
        return df_enc_funcs
        

# Use a standard vectorization approach by creating a feature matrix.
# Loses semantic and structural information, but it is FAST (>> anything BERT embedding related)
# Pre-configured to use 750 features 
class FuncEncoderTFIDFVectorizer:
    # df_data required for vocabulary creation
    def __init__(self, df_data=None, vocabulary_path=None):
        if (df_data is None and vocabulary_path is None):
            raise ValueError("df_data or vocabulary_path must be set.")
            
        if (vocabulary_path is not None):
            try:
                # load vocab from file
                vocab = self.__load_vocabulary(vocabulary_path)
                # Show warning if both params were set and loading succeeded
                if (df_data is not None and vocabulary_path is not None):
                    warnings.warn("Both df_data and vocabulary_path are set. Ignoring df_data.")
            except:
                # if file not found, try to extract from data ...
                if (df_data is not None):
                    warnings.warn("Loading vocabulary form '{0}' failed. Falling back to df_data.".format(vocabulary_path))
                    vocab = self.__extract_vocabulary(df_data)
                    if (vocabulary_path is not None):
                        self.__save_vocabulary(vocab, vocabulary_path)
                # ... otherwise fail
                else:
                    raise ValueError("Cannot load vocabulary form '{0}'.".format(vocabulary_path))
        else:
            vocab = self.__extract_vocabulary(df_data)
            if (vocabulary_path is not None):
                self.__save_vocabulary(vocab, vocabulary_path)
            
        
        # Create another CountVectorizer with the vocabulary from the CV above
        # (ensures to have consistent token features for test, training, validation)
        self.vectorizer = TfidfVectorizer(tokenizer = tokenizer_bert.tokenize,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             lowercase=False, \
                             vocabulary=vocab)
    
    def __load_vocabulary(self, vocabulary_path):
        return joblib.load(vocabulary_path)
        
    def __save_vocabulary(self, vocabulary, vocabulary_path):
        joblib.dump(vocabulary, vocabulary_path)
    
    def __extract_vocabulary(self, df_data):
        all_funcs = df_data['processed_func'].tolist()
        
        # Create a count vectorizer on the whole dataset and include the most occuring n tokens
        # tokenizer takes CodeBERT as function
        newVec = CountVectorizer(tokenizer = tokenizer_bert.tokenize,    \
                                     preprocessor = None, \
                                     stop_words = None,   \
                                     max_features=1500, \
                                     lowercase=False)
        newVec.fit_transform(all_funcs)
        return newVec.vocabulary_
    
    # Return feature matrix for list_of_strings
    # columns ... features
    # index ... row index of input list
    def __vectorizer_encode(self, list_of_strings):
        features = self.vectorizer.fit_transform(list_of_strings)
        return pd.DataFrame(features.toarray(), columns=self.vectorizer.get_feature_names_out())

    # Return a df of features
    def encode(self, list_funcs):
        # get feature matrix (as a df)
        df_enc_funcs = self.__vectorizer_encode(list_funcs)
        # return df
        return df_enc_funcs  
        