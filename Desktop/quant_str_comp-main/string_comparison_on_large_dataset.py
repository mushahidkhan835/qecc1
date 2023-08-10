import math

import pandas as pd
from string_comparison import StringComparator
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

class StringComparisonOnLargeDataset:
    def __init__(self):
        self.balanced_scale_df = pd.read_csv('./datasets/balance_scale.csv')
        self.breast_cancer_df = pd.read_csv('./datasets/breast_cancer.csv')
        self.lymphography_df = pd.read_csv('./datasets/lymphography.csv')
        self.zoo_df = pd.read_csv('./datasets/zoo.csv')
        self.votes_df = pd.read_csv('./datasets/votes.csv')
        self.tictactoe_df = pd.read_csv('./datasets/tictactoe.csv')

    def balance_scale_data_set(self):
        self.balanced_scale_df['1'] = self.balanced_scale_df['1'] - 1
        self.balanced_scale_df['1.1'] = self.balanced_scale_df['1.1'] - 1
        self.balanced_scale_df['1.2'] = self.balanced_scale_df['1.2'] - 1
        self.balanced_scale_df['1.3'] = self.balanced_scale_df['1.3'] - 1

        for c in self.balanced_scale_df.columns:
            if c == '1' or c == '1.2' or c == '1.1' or c == '1.3':
                self.balanced_scale_df[c] = self.balanced_scale_df[c].apply(lambda x: self.changeToBinary(x, 3))

        data_as_list = []
        for index, row in self.balanced_scale_df.iterrows():
            data_as_list.append([row['B'], row['1'] + row['1.1'] + row['1.2'] + row['1.3']])

        data_as_list = data_as_list[0: 620]
        data_as_10_fold = []
        temp = []
        l = len(data_as_list) // 10

        for d in data_as_list:
            if len(temp) == l:
                data_as_10_fold.append(temp)
                temp = [d]
            else:
                temp.append(d)

        if len(temp) == l:
            data_as_10_fold.append(temp)

        return self.run_ten_fold_test(3, data_as_10_fold)

    def cancer_data_set(self):
        self.breast_cancer_df.drop(['1000025'], 1, inplace=True)

        for column in self.breast_cancer_df.columns:
            m = self.breast_cancer_df[column].mode()[0]
            self.breast_cancer_df[column] = self.breast_cancer_df[column].replace(['?'], m)
            self.breast_cancer_df[column].fillna(m)
            self.breast_cancer_df[column].replace(np.nan, m)

        X = self.breast_cancer_df.iloc[:, 0: 9]
        Y = self.breast_cancer_df.iloc[:, -1]

        from sklearn.feature_selection import f_classif
        ffs = SelectKBest(score_func=f_classif, k=4)#the k can change
        X_selected = ffs.fit_transform(X, Y)
        cols = ffs.get_support(indices=True)
        X = X.iloc[:, cols]

        for c in X:
           X[c] = X[c].apply(lambda x: self.changeToBinary(x, 4))
        X["output"] = Y
        print(X.head())
        data_as_list = []
        for index, row in X.iterrows():
            s = ""
            for c in X:
                if c != "output":
                    s = s + str(row[c])
            data_as_list.append([row["output"], s])

        data_as_list = data_as_list[0:690]
        data_as_10_fold = []
        temp = []
        l = len(data_as_list) // 10

        for d in data_as_list:
            if len(temp) == l:
                data_as_10_fold.append(temp)
                temp = [d]
            else:
                temp.append(d)

        if len(temp) == l:
            data_as_10_fold.append(temp)

        return self.run_ten_fold_test(4, data_as_10_fold)

    def run_ten_fold_test(self, s_l, data_as_10_fold):
        acc = 0
        for i in range(len(data_as_10_fold)):
            print(i)
            test_set = data_as_10_fold[i]
            training_set = []
            training_set_with_classes = {}
            for j in range(len(data_as_10_fold)):
                if i != j:
                    for d in data_as_10_fold[j]:
                        training_set.append(d)
            for t in training_set:
                if t[0] in training_set_with_classes:
                    training_set_with_classes[t[0]].append(t[1])
                else:
                    training_set_with_classes[t[0]] = [t[1]]
            for test in test_set:
                prob_of_zero_test = -math.inf
                class_test_belongs_to = ""
                for key in training_set_with_classes:
                    print(key)
                    x = StringComparator(test[1], training_set_with_classes[key], symbol_length=s_l)
                    results = x.run()
                    prob_c_zero_l = results['prob_of_measuring_register_c_as_0']
                    if prob_c_zero_l > prob_of_zero_test:
                        class_test_belongs_to = key
                        prob_of_zero_test = prob_c_zero_l
                if test[0] == class_test_belongs_to:
                    acc += 1
        return acc / (len(data_as_10_fold) * 10)

    def votes_data_set(self):
        self.votes_df.rename(columns={'?': 'c'}, inplace=True)
        for c in self.votes_df.columns:
            self.votes_df.loc[self.votes_df[c] == '?', c] = self.votes_df.mode()[c][0]
        for c in self.votes_df.columns:
            self.votes_df.loc[self.votes_df[c] == 'y', c] = 1
            self.votes_df.loc[self.votes_df[c] == 'n', c] = 0


        X = self.votes_df.iloc[:, 1:]
        Y = self.votes_df.iloc[:, 0]

        ffs = SelectKBest(score_func=f_classif, k=6) #k can be changed
        X_selected = ffs.fit_transform(X, Y)
        cols = ffs.get_support(indices=True)
        X = X.iloc[:, cols]

        for c in X:
           X[c] = X[c].apply(lambda x: self.changeToBinary(x, 4))
        X["output"] = Y

        data_as_list = []
        for index, row in X.iterrows():
            s = ""
            for c in X:
                if c != "output":
                    s = s + str(row[c])
            data_as_list.append([row["output"], s])

        data_as_list = data_as_list[0:430]
        data_as_10_fold = []
        temp = []
        l = len(data_as_list) // 10

        for d in data_as_list:
            if len(temp) == l:
                data_as_10_fold.append(temp)
                temp = [d]
            else:
                temp.append(d)

        if len(temp) == l:
            data_as_10_fold.append(temp)

        return self.run_ten_fold_test(1, data_as_10_fold)

    def zoo_data_set(self):
        self.zoo_df.drop(['4', 'aardvark'], axis=1, inplace=True)
        X = self.zoo_df.iloc[:, 0:-1]
        Y = self.zoo_df.iloc[:, -1]

        ffs = SelectKBest(score_func=chi2, k=12)
        X_selected = ffs.fit_transform(X, Y)
        cols = ffs.get_support(indices=True)
        X = X.iloc[:, cols]

        X["output"] = Y

        data_as_list = []
        for index, row in X.iterrows():
            s = ""
            for c in X:
                if c != "output":
                    s = s + str(row[c])
            data_as_list.append([row["output"], s])

        data_as_10_fold = []
        temp = []
        l = len(data_as_list) // 10

        for d in data_as_list:
            if len(temp) == l:
                data_as_10_fold.append(temp)
                temp = [d]
            else:
                temp.append(d)

        if len(temp) == l:
            data_as_10_fold.append(temp)

        return self.run_ten_fold_test(1, data_as_10_fold)

    def changeToBinary(self, n, maxLen):
        binaryForm = bin(int(n)).replace("0b", "")
        while len(binaryForm) < maxLen:
            binaryForm = "0" + binaryForm
        return binaryForm



if __name__ == "__main__":
    x = StringComparisonOnLargeDataset()
    accuracies = x.tictactoe_data_set()
    print(accuracies)


