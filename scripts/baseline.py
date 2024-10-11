from collections import Counter
import random
import editdistance
import pandas as pd
import re
import math
random.seed(42)

def order_string(word: str) -> str:
    word = ''.join(sorted(word))
    return word

def get_most_frequent_word(typo_word: str, fre_dict: dict) -> str:
    typo_word = order_string(typo_word)
    if typo_word in fre_dict.keys():
        return fre_dict[typo_word]
    else:
        return '[UNK]'

def sentence_tokennizer(sentence: str) -> list:
    #Remove all non-alphabet chars
    regex = re.compile('[^a-zA-Z ]')
    sentence = regex.sub('', sentence)
    sentence = sentence.lower()
    sentence = sentence.split(' ')
    #Remove empty strings
    sentence = [i for i in sentence if len(i) != 0]
    return sentence

def calculate_edit_distance(pre_word: str, word: str):
    if pre_word == '[UNK]':
        return 1
    else:
        #Penalize for too long predictions e.g. f('ba', 'bane')/len('ba') = 1
        #And keep propotional score if too short or same len, e.g. f('bane', 'ba')/len('bane') = 0.5
        
        """#To get percentage of word that needs edit
        score = editdistance.eval(word, pre_word)/len(word)
        #Multiply by log
        return score*math.log10(len(word))"""

        return editdistance.eval(word, pre_word)

def create_frequent_dict(word_list: list) -> dict:
    #Get word counts
    word_count = Counter(word_list)
    #Make frequent dict of words
    fre_dict = {}
    #score dict 
    scores = {}
    #Assign most frequent word
    for word, count in word_count.items():
        word_sorted = order_string(word)
        #Make keys if not exists
        if word_sorted not in fre_dict.keys():
            fre_dict[word_sorted] = word
            scores[word_sorted] = count
            continue
        #If new word is more frequent
        if scores[word_sorted] < count:
            fre_dict[word_sorted] = word
            scores[word_sorted] = count
            continue
        #If new word has same frequent - replace with 50% chance
        if scores[word_sorted] == count and random.random() >= 0.5:
            fre_dict[word_sorted] = word
            scores[word_sorted] = count
            continue
    return fre_dict

def get_predictions(list_of_sen: list, fre_dict: dict) -> list:
    predictions = []

    for sen in list_of_sen:
        sen = [get_most_frequent_word(i, fre_dict) for i in sen]
        predictions.append(sen)
    
    return predictions

def get_score(predictions: list, ground_truth: list):
    total_score = 0

    #Word
    total_token = 0
    total_score_word = 0
    
    #For every sentence
    for sen_idx in range(len(predictions)):
        y = [i for i in ground_truth[sen_idx] if len(i) > 3]
        y_hat = [i for i in predictions[sen_idx] if len(i) > 3]
        #Total editdistance for sentence
        sentence_score = 0
        for w_idx in range(len(y)):
            sentence_score += calculate_edit_distance(pre_word = y_hat[w_idx], word = y[w_idx])
            
        #on sentence level
        if sentence_score != 0:
            total_score += sentence_score/len(y)
        
        #on word level
        total_token += len(y)
        total_score_word += sentence_score
    return total_score/len(predictions), total_score_word/total_token

def get_base_line_score(train: pd.DataFrame, test: pd.DataFrame, type: str) -> None:
    #Get stats
    word_list = []
    for sen in train[type].to_list():
        word_list += sentence_tokennizer(sen)
    fre_dict = create_frequent_dict(word_list)
    
    #Prepare test data
    test_data = [sentence_tokennizer(sen) for sen in test[type+'_Typo'].to_list()]
    y_test = [sentence_tokennizer(sen) for sen in test[type].to_list()]

    #Predict test data
    predictions = get_predictions(test_data, fre_dict)

    #Calculate score
    sen_score, word_score = get_score(predictions = predictions, ground_truth = y_test)
    print(f"The base line has a mean editdistance of {round(sen_score,3)} pr. sentence, and {round(word_score,3)} pr. word")