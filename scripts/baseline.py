from collections import Counter
import random
import editdistance
import pandas as pd
import re
import math

from scripts.preprocessing import sentence_tokennizer

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



def calculate_edit_distance(pre_word: str, word: str):
    if pre_word == '[UNK]':
        return 1
    else:
        #Penalize for too long predictions e.g. f('ba', 'bane')/len('ba') = 1
        #And keep propotional score if too short or same len, e.g. f('bane', 'ba')/len('bane') = 0.5
        
        return editdistance.eval(word, pre_word)/len(word)

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

    word_stats = {}
    score_data = []
    total_score_words = 0
    num_of_words = 0 
    #For every sentence
    for sen_idx in range(len(predictions)):
        y = [i for i in ground_truth[sen_idx]]
        y_hat = [i for i in predictions[sen_idx]]
        #Total editdistance for sentence
        sentence_score = 0
        words = []
        for w_idx in range(len(y)):
            if y[w_idx] not in word_stats:
                word_stats[y[w_idx]] = {'count':0, 'score_sum': 0}
            #get edit distance
            ed = calculate_edit_distance(pre_word = y_hat[w_idx], word = y[w_idx])
            #add edit distance
            sentence_score += ed
            word_stats[y[w_idx]]['score_sum'] = ed + word_stats[y[w_idx]]['score_sum']
            #update count
            word_stats[y[w_idx]]['count'] = 1 + word_stats[y[w_idx]]['count']
            #Add to sentence level
            words.append(ed)
            #Add to word level
            total_score_words += ed
            num_of_words += 1
            
        #on sentence level
        if sentence_score != 0:
            sentence_score = sentence_score/len(y)
        score_data.append([sentence_score, words])

    avg_word = total_score_words/num_of_words
    return score_data, word_stats, avg_word

def get_base_line_score(train: pd.DataFrame, test: pd.DataFrame, type: str) -> None:
    #Get stats
    word_list = []
    for sen in train[type].to_list():
        word_list += sentence_tokennizer(sen)
    fre_dict = create_frequent_dict(word_list)
    
    #Prepare test data
    test_data = [sentence_tokennizer(sen) for sen in test['typoglycemia'].to_list()]
    y_test = [sentence_tokennizer(sen) for sen in test[type].to_list()]

    #Predict test data
    predictions = get_predictions(test_data, fre_dict)
    print(predictions)
    #Calculate score
    score_data, word_stats, avg_word = get_score(predictions = predictions, ground_truth = y_test)
    score_data = pd.DataFrame(score_data, columns= ['Avg sentence', 'Words'])
    #Multiply with 100 to get percentage
    print(f"The {type} baseline has a mean editdistance of {round(score_data['Avg sentence'].mean()*100,3)}% pr. sentence")
    print(f"The {type} baseline has a mean editdistance of {round(avg_word*100,3)}% pr. word")
    word_performance = []
    for key, value in word_stats.items():
        word_performance.append([key, len(key), value['count'], value['score_sum']])
    word_performance = pd.DataFrame(word_performance, columns= ['word', 'len', 'freq', 'total_score'])
    word_performance['avg'] = word_performance['total_score']/word_performance['freq']
    word_performance = word_performance[['word', 'len', 'freq', 'avg']]
    word_performance.to_csv(f'analysis/{type}_baseline_stats.csv')