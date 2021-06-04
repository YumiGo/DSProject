import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
'''
function: scaling_encoding_cases()
input: X_train_num: Numerical input data for train set
       X_train_cat: Categorical input data for train set
       y_train: Target Data for Train Set
       X_test_num: Numerical input data for test set
       X_test_cat: Categorical input data for test set
       y_test: Target Data for Test Set
       s1: Scaling method 1 (Default: 'MinMax')
       s2: Scaling method 2 (Default: None)
       s3: Scaling method 3 (Default: None)
       s4: Scaling method 4 (Default: None)
       e1: Encoding method 1 (Default: 'Label')
       e2: Encoding method 2 (Default: None)
output: Datarame that represents the accuracy of each scaling and encoding combination by rank
description: This function receives input from data, scaling methods, and encoding methods
             then preprocess the data according to each scaling and encoding combination, trains the model
             using Bagging Classifier, and shows its accuracy by rank.
'''


def scaling_encoding_cases(X_train_num, X_train_cat, y_train, X_test_num,  X_test_cat, y_test, s1='MinMax', s2=None,
                     s3=None, s4=None, e1='Label', e2=None):
    s = [] # List to store scalers
    e = [] # List to store encoders
    scaler = [] # List to store information about the scaler.
    num_attribs = list(X_train_num) # A list that stores information in numerical features
    cat_attribs = list(X_train_cat) # A list that stores information in categorical features
    cat_len = len(X_train_cat.columns) # Number of categorical features

    count_scaler = 0 # Number of scalers
    count_encoder = 0 # Number of encoders

    # Set up the scaler based on the input
    if s1 == 'Standard':
        scaler.append(StandardScaler())
        s.append(s1)
        count_scaler = count_scaler + 1
    if s1 == 'MinMax':
        scaler.append(MinMaxScaler())
        s.append(s1)
        count_scaler = count_scaler + 1
    if s1 == 'Robust':
        scaler.append(RobustScaler())
        s.append(s1)
        count_scaler = count_scaler + 1
    if s1 == 'MaxAbs':
        scaler.append(MaxAbsScaler())
        s.append(s1)
        count_scaler = count_scaler + 1
    if s2 == 'Standard':
        scaler.append(StandardScaler())
        s.append(s2)
        count_scaler = count_scaler + 1
    if s2 == 'MinMax':
        scaler.append(MinMaxScaler())
        s.append(s2)
        count_scaler = count_scaler + 1
    if s2 == 'Robust':
        scaler.append(RobustScaler())
        s.append(s2)
        count_scaler = count_scaler + 1
    if s2 == 'MaxAbs':
        scaler.append(MaxAbsScaler())
        s.append(s2)
        count_scaler = count_scaler + 1
    if s3 == 'Standard':
        scaler.append(StandardScaler())
        s.append(s3)
        count_scaler = count_scaler + 1
    if s3 == 'MinMax':
        scaler.append(MinMaxScaler())
        s.append(s3)
        count_scaler = count_scaler + 1
    if s3 == 'Robust':
        scaler.append(RobustScaler())
        s.append(s3)
        count_scaler = count_scaler + 1
    if s3 == 'MaxAbs':
        scaler.append(MaxAbsScaler())
        s.append(s3)
        count_scaler = count_scaler + 1
    if s4 == 'Standard':
        scaler.append(StandardScaler())
        s.append(s4)
        count_scaler = count_scaler + 1
    if s4 == 'MinMax':
        scaler.append(MinMaxScaler())
        s.append(s4)
        count_scaler = count_scaler + 1
    if s4 == 'Robust':
        scaler.append(RobustScaler())
        s.append(s4)
        count_scaler = count_scaler + 1
    if s4 == 'MaxAbs':
        scaler.append(MaxAbsScaler())
        s.append(s4)
        count_scaler = count_scaler + 1
    if e1 == 'Ordinal':
        e.append(e1)
        # Encoding - fit_transform
        X_train_cat_encoded = OrdinalEncoder().fit_transform(X_train_cat)
        X_test_cat_encoded = OrdinalEncoder().fit_transform(X_test_cat)
        # Convert encoded data to Data Frame
        X_train_cat_encoded = pd.DataFrame(X_train_cat_encoded, columns=list(X_train_cat))
        X_test_cat_encoded = pd.DataFrame(X_test_cat_encoded, columns=list(X_test_cat))
        count_encoder = count_encoder + 1
    if e1 == 'Label':
        e.append(e1)
        encoder = LabelEncoder()
        count_encoder = count_encoder + 1
        # Since label encoder can encode one column at a time,
        # repeat by the number of categorical features
        for i in range(0, cat_len):
            globals()['X_train_cat_encoded{}'.format(i)] = encoder.fit_transform(X_train_cat[cat_attribs[i]])
            globals()['X_train_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_train_cat_encoded{}'.format(i)],
                                                                        columns=[cat_attribs[i]])
            globals()['X_test_cat_encoded{}'.format(i)] = encoder.fit_transform(X_test_cat[cat_attribs[i]])
            globals()['X_test_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_test_cat_encoded{}'.format(i)],
                                                                       columns=[cat_attribs[i]])
            # Concat encoded data
            X_train_cat_encoded = pd.concat([globals()['X_test_cat_encoded{}'.format(i)]], axis=1)
            X_test_cat_encoded = pd.concat([globals()['X_test_cat_encoded{}'.format(i)]], axis=1)
    if e2 == 'Ordinal':
        e.append(e2)
        X_train_cat_encoded = OrdinalEncoder().fit_transform(X_train_cat)
        X_test_cat_encoded = OrdinalEncoder().fit_transform(X_test_cat)
        X_train_cat_encoded = pd.DataFrame(X_train_cat_encoded, columns=list(X_train_cat))
        X_test_cat_encoded = pd.DataFrame(X_test_cat_encoded, columns=list(X_test_cat))
        count_encoder = count_encoder + 1
    if e2 == 'Label':
        e.append(e2)
        encoder = LabelEncoder()
        count_encoder = count_encoder + 1
        list_of_train = []
        list_of_test = []
        for i in range(0, cat_len):
            globals()['X_train_cat_encoded{}'.format(i)] = encoder.fit_transform(X_train_cat[cat_attribs[i]])
            globals()['X_train_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_train_cat_encoded{}'.format(i)],
                                                                        columns=[cat_attribs[i]])
            globals()['X_test_cat_encoded{}'.format(i)] = encoder.fit_transform(X_test_cat[cat_attribs[i]])
            globals()['X_test_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_test_cat_encoded{}'.format(i)],
                                                                       columns=[cat_attribs[i]])
            list_of_train.append(globals()['X_train_cat_encoded{}'.format(i)])
            list_of_test.append(globals()['X_test_cat_encoded{}'.format(i)])
        X_train_cat_encoded = pd.concat(list_of_train, ignore_index=True, axis=1)
        X_test_cat_encoded = pd.concat(list_of_test, ignore_index=True, axis=1)

    count_score = 0 # Number of scores
    scaler_encoder = [] # List to contain scaling, encoding combination information
    score = [] # List to contain score
    for i in range(0, count_scaler): # Repeat by the number of scalers
        for j in range(0, count_encoder): # Repeat by the number of encoders
            string = s[i] + '&' + e[j] # scaling, encoding combination information
            scaler_encoder.append(string) # append information to 'scaler_encoder' list
            # Scaling - fit_transform
            X_train_num_scaled = scaler[i].fit_transform(X_train_num) # Scale numerical train data
            X_test_num_scaled = scaler[i].fit_transform(X_test_num) # Scale numerical test data
            # Scaling - Convert scaled data to Data Frame
            X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=num_attribs)
            X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=num_attribs)
            # Concat scaled, encoded data
            X_train_prepared = pd.concat([X_train_num_scaled, X_train_cat_encoded], axis=1)
            X_test_prepared = pd.concat([X_test_num_scaled, X_test_cat_encoded], axis=1)
            # Bagging Classifier Declaration
            bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=300, max_samples=100,
                                        bootstrap=True,
                                        max_features=9, n_jobs=-1)
            # Bagging Classifier - fit
            bag_clf.fit(X_train_prepared, y_train)
            # Predict target
            y_pred = bag_clf.predict(X_test_prepared)
            # Put the accuracy score on the score list.
            score.append(accuracy_score(y_test, y_pred))
            count_score = count_score + 1 # Increase the number of scores
   # Selection sort - sort the scores by descending order
    for i in range(count_score - 1):  # Repeat by (size-1) of the list
        for j in range(i + 1, len(score)):  # From that (index+1), repeat by list size
            if score[i] < score[j]:  # If the value of score[j] is greater than score[i]
                score[i], score[j] = score[j], score[i]  # swap
                scaler_encoder[i], scaler_encoder[j] = scaler_encoder[j], scaler_encoder[i]
    # Create an empty list to contain results
    result = []
    # Repeat as number of score
    for i in range(0, count_score):
        a = [] # Create another empty list to contain results
        # append data to list a
        a.append(i + 1) # rank of the combination
        a.append(scaler_encoder[i]) # information abount the scaling & encoding combination
        a.append(score[i]) # accuracy score of the combination
        # append list 'a' to list 'result' -> 2D list created
        result.append(a)
    # Make the 'result' list a data frame.
    table = pd.DataFrame(result, columns=['Rank', 'Scaler & Encoder', 'Accuracy'])
    # Return the data frame
    return table
