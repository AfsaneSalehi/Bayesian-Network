import math
import pandas as pd
import numpy as np
from itertools import chain, combinations
from decimal import Decimal


def equation(N, alphas, r, length, len_labels):
    res1 = 0
    res2 = 0
    for n in N:
        temp_value = Decimal((math.factorial(r - 1)) / math.factorial(n + r - 1))
        if temp_value != 0:
            res1 = res1 * Decimal(temp_value)

    for alpha in alphas:
        if len(alpha) != 0:
            res2 = Decimal(res2) * Decimal(math.factorial(len(alpha)))
    res = res1*res2
    
    for n in N:
        vl = Decimal((math.factorial(r - 1)) / math.factorial(n + r - 1))
        if vl != 0:
             res1 = res1 + Decimal(math.log(vl))

    for alpha in alphas:    
        if len(alpha) != 0:
            res2 = Decimal(res2) + Decimal(math.log(math.factorial(len(alpha))))

    res = -Decimal(2*(res1+res2)) + Decimal((2*len_labels) * math.log(length))   
    return res


def fill_parents_map(data):
    labels = list(data.columns)
    parents_map = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])),index = labels, columns = labels)  
    features_num = data.shape[1]
    att_values = {}
    for i in range(1, features_num): 
        if i == 0:
            parents = []
        else:
            if i > 2:
                parents = labels[i-2:i]  
                parents.insert(0, labels[0])
            else:
                parents = labels[:i]            
        parents = list(chain.from_iterable(combinations(parents,n) for n in range(len(parents)+1)))
        att_values[labels[i]] = list(set(data.iloc[:, i]))
        r = len(att_values[labels[i]])
        mxm = -math.inf
        best_parents = []
        for prs in parents:
            alphas, N = n_alpha(i, prs, data)
            length = len(data)
            len_labels = len(labels)
            result = equation(N, alphas, r, length, len_labels)
            if result >= mxm:
                mxm = result
                best_parents = prs
        for p in best_parents:
            parents_map[labels[i]][p] = 1
    return parents_map.T
            


def n_alpha(index, parents, data):
    n_tmp = 0
    att_values = {}
    alpha_parts, n = [], []
    labels = list(data.columns)
    att_values[labels[index]] = list(set(data.iloc[:, index]))
    if len(parents) == 0:
        for value in att_values[labels[index]]:
            alpha = len(data[data[labels[index]] == value])
            alpha_parts.append(data[data[labels[index]] == value])
            n_tmp = n_tmp + alpha
        n.append(n_tmp)
        return alpha_parts, n 
    else:
        alphas, _ = n_alpha(labels.index(parents[0]), parents[1:], data)
        for c in alphas:
            n_tmp = 0
            for value in att_values[labels[index]]:
                alpha = len(c[c[labels[index]] == value])
                alpha_parts.append(c[c[labels[index]] == value])
                n_tmp = n_tmp + alpha
            n.append(n_tmp)
            
    return alpha_parts, n


def compute_Prob(feature, value, data):
    overall = len(data)
    fit = data[data.iloc[:, feature] == value].count()
    if fit[0] == 0:
        return 0
    p = fit[0] / overall
    return p


def probs(train, test, parents_map, value, idx):
    features = list(train.columns)
    for i in range(0, parents_map.shape[0]):
        temp = 0
        c_train = train.copy()
        if i == 0:
            res = compute_Prob(i, value, train)
        else:
            for m in range(0, parents_map.shape[1]):
                if parents_map[features[m]][features[i]] == 1:
                    temp = 1
                    if m == 0:
                        c_train = c_train[c_train.iloc[:, m] == value]
                    else:
                        c_train = c_train[c_train.iloc[:, m] == test.iloc[idx, m]]
            # If has no parents
            if temp == 0:
                res = res * compute_Prob(i, test.iloc[idx, i], train)
            else:
                res = res * compute_Prob(i, test.iloc[idx, i], c_train)
    return res


def test(train, test, parents_map, itr):
    tp, tn = 0, 0
    fp, fn = 0, 0
    label = test.iloc[:, 0]
    class_values = list(set(test.iloc[:, 0]))

    for i in range(0, test.shape[0]):
        fmax = -math.inf
        for class_vl in class_values:
            result = probs(train, test, parents_map, class_vl, i)
            if result > fmax:
                fmax = result
                predicted = class_vl
        if predicted == label.iloc[i]:
            if predicted == 1:
                tn = tn + 1
            elif predicted == 2:
                tp = tp + 1
        elif predicted != label.iloc[i]:
            if predicted == 2:
                fp = fp + 1
            elif predicted == 1:
                fn = fn + 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precison = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2*precison*recall/(precison + recall)
    print('Accuracy' + str(itr) + ': ' + str(accuracy))
    print('Precison' + str(itr) + ': ' + str(precison))
    print('Recall' + str(itr) + ': ' + str(recall))
    print('F1-Score' + str(itr) + ': ' + str(fscore) + '\n')
    return accuracy, precison, recall, fscore

def cross_validation(data, folds):
    # K-fold cross validation
    bound = int(len(data)/folds)
    accuracies = []
    precisions = []
    recalls = []
    f_scores = []
    for i in range(folds):
        data_tmp = data.copy()
        test_set = data.iloc[i*bound:(i+1)*bound, :]
        train_set = data_tmp.drop(test_set.index)       

        depends = fill_parents_map(data)
        itr = i+1
        acc, precision, recall, f_score = test(train_set, test_set, depends,itr)
        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
    print('Final Results: ')
    print('averag accuracy: ', np.mean(accuracies))
    print('averag precision: ', np.mean(precisions))
    print('averag recalls: ', np.mean(recalls))
    print('averag F: ', np.mean(f_scores))

def main():
    data = pd.read_csv(r'D:\learning\University\AI_DrHarati\HWs\HW6\data.csv', index_col = 0)
    data = data.sample(frac=1).reset_index(drop=True)
    labels = list(data.columns)
    labels.reverse()
    data = data[labels]
    cross_validation(data, 5)
	
if __name__ == "__main__":
	main()