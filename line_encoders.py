from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
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
class EncoderCountVectorizer:
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
        all_lines = df_data['line'].tolist() + df_data['prev_line'].tolist() + df_data['next_line'].tolist()
        
        # Create a count vectorizer on the whole dataset and include the most occuring n tokens
        # tokenizer takes CodeBERT as function
        newVec = CountVectorizer(tokenizer = tokenizer_bert.tokenize,    \
                                     preprocessor = None, \
                                     stop_words = None,   \
                                     max_features=1500, \
                                     lowercase=False,\
                                     min_df=0.01,
                                     max_df=0.2)
        newVec.fit_transform(all_lines)
        return newVec.vocabulary_
    
    # Return feature matrix for list_of_strings
    # columns ... features
    # index ... row index of input list
    def __vectorizer_encode(self, list_of_strings):
        features = self.vectorizer.fit_transform(list_of_strings)
        return pd.DataFrame(features.toarray(), columns=self.vectorizer.get_feature_names_out())

    # Return a df of features
    # Please note that this DF contains 3x the columns as features, 
    # since features are distinctly represened for line, prev_line and next_line 
    def encode(self, list_lines, list_prev_lines, list_next_lines):
        # get feature matrices (as a dfs)
        df_enc_lines = self.__vectorizer_encode(list_lines)
        df_enc_prev_lines = self.__vectorizer_encode(list_prev_lines)
        df_enc_next_lines = self.__vectorizer_encode(list_next_lines)
        
        # join feature matrices, but preserve distinction by adding suffix
        df_joined = df_enc_lines.join(df_enc_prev_lines, rsuffix="_prev")
        df_joined = df_joined.join(df_enc_next_lines, rsuffix="_next")
        
        # return joined df
        return df_joined
        
    # Return a df of features
    def encodeSingle(self, list_lines):
        # get feature matrices (as a dfs)
        df_enc_lines = self.__vectorizer_encode(list_lines)
        
        # return df
        return df_enc_lines
        
        
# Most precise since the embeddings of all 3 features are preserved
# On average 2x slower than EncoderBERTStringConcat
class EncoderBERTVectorConcat:
    # avg_embeddings ... If true, then the encoded embeddings are averaged, otherwise the first embedding vector is returned
    def __init__(self, avg_embeddings=True):
        self.avg_embeddings = avg_embeddings

    # Condense multiple embeddings vectors to ONE single vector using element-wise averaging 
    def __condenseEmbeddings(self, context_embeddings):
        np_array = context_embeddings.detach().numpy()
        
        if (self.avg_embeddings):
            avg_embedding = np.mean(np_array[0].tolist(), axis=0)
            return avg_embedding
        else:
            # just take the first embedding vector
            return np_array[0][0]

    # Return n 768 vectors
    # This vector is created by element-wise averaging of all token vectors of a string 
    def __bert_encode(self, list_of_strings):
        embeddings = []
        for s in list_of_strings:
            # tokenize
            code_tokens=tokenizer_bert.tokenize(s)
            # add special tokens
            tokens=[tokenizer_bert.cls_token]+code_tokens+[tokenizer_bert.sep_token]
            # convert to IDs
            tokens_ids=tokenizer_bert.convert_tokens_to_ids(tokens)
            # create embedding
            context_embeddings=model_bert(torch.tensor(tokens_ids)[None,:])[0]
            # condense embedding to a single vector of fixed size
            embeddings.append(self.__condenseEmbeddings(context_embeddings))

        return embeddings

    # Return n 2304 vectors by encoding and element-wise concatenation of encoded inputs
    # n ... length of input lists
    def encode(self, list_lines, list_prev_lines, list_next_lines):
        encoded_lines = self.__bert_encode(list_lines)
        encoded_prev_lines = self.__bert_encode(list_prev_lines)
        encoded_next_lines = self.__bert_encode(list_next_lines)

        # concat 3x768 vectors to 1x2304 vector 
        return [np.concatenate((l, p, n)) for l, p, n in zip(encoded_lines, encoded_prev_lines, encoded_next_lines)]
    
    # Return a df of features
    def encodeSingle(self, list_lines):
        encoded_lines = self.__bert_encode(list_lines)
        # return 768 vector
        return encoded_lines


# Less precise since all 3 features are concatenated before embedding creation
# On average 2x faster than EncoderBERTVectorConcat
class EncoderBERTStringConcat:
    # avg_embeddings ... If true, then the encoded embeddings are averaged, otherwise the first embedding vector is returned
    def __init__(self, avg_embeddings=True):
        self.avg_embeddings = avg_embeddings
    
    # Condense multiple embeddings vectors to ONE single vector using element-wise averaging 
    def __condenseEmbeddings(self, context_embeddings):
        np_array = context_embeddings.detach().numpy()
        
        if (self.avg_embeddings):
            avg_embedding = np.mean(np_array[0].tolist(), axis=0)
            return avg_embedding
        else:
            # just take the first embedding vector
            return np_array[0][0]

    # Return n 768 vectors
    # This vector is created by element-wise averaging of all token vectors of a string 
    def __bert_encode(self, list_lines, list_prev_lines, list_next_lines):
        embeddings = []
        for i in range(0, len(list_lines)):
            # tokenize
            code_tokens_lines=tokenizer_bert.tokenize(list_lines[i])
            code_tokens_prev_lines=tokenizer_bert.tokenize(list_prev_lines[i])
            code_tokens_next_lines=tokenizer_bert.tokenize(list_next_lines[i])
            
            # add special tokens
            tokens=[tokenizer_bert.cls_token]+code_tokens_lines+[tokenizer_bert.sep_token]+ \
                  code_tokens_prev_lines+[tokenizer_bert.sep_token]+code_tokens_next_lines+[tokenizer_bert.sep_token]
            # convert to IDs
            tokens_ids=tokenizer_bert.convert_tokens_to_ids(tokens)
            # create embedding
            context_embeddings=model_bert(torch.tensor(tokens_ids)[None,:])[0]
            # condense embedding to a single vector of fixed size
            embeddings.append(self.__condenseEmbeddings(context_embeddings))

        return embeddings

    # Return n 768 vectors by encoding element-wise concatenated input strings
    # n ... length of input lists
    def encode(self, list_lines, list_prev_lines, list_next_lines):
        encoded_lines = self.__bert_encode(list_lines, list_prev_lines, list_next_lines)
        return encoded_lines
        
    # not supported (use EncoderBERTVectorConcat instead)
    def encodeSingle(self, list_lines):
        raise NotImplementedError("use EncoderBERTVectorConcat instead")
