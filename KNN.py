import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Data read ##
### The Vaccine Adverse Event Reporting System (VAERS) ###
dataFrame = pd.read_csv('2021VAERSDATA.csv',encoding='ISO-8859-1', low_memory=False)
symptom = pd.read_csv('2021VAERSSYMPTOMS.csv')
symptom.drop_duplicates(subset=['VAERS_ID'], keep='first', inplace=True, ignore_index=False)
vax = pd.read_csv('2021VAERSVAX.csv')
vax.drop_duplicates(subset=['VAERS_ID'], keep='first', inplace=True, ignore_index=False)
dataFrame1 = pd.merge(dataFrame, symptom, on = 'VAERS_ID')
data = pd.merge(dataFrame1, vax, on = 'VAERS_ID')
data = data[data.VAX_TYPE == 'COVID19']
data = data[data.VAX_NAME.str.contains('COVID19')]
data = data[['STATE', 'AGE_YRS', 'SEX', 'RECOVD', 'NUMDAYS', 'OTHER_MEDS', 'CUR_ILL', 'ALLERGIES','VAX_MANU', 'SYMPTOM1']]

from collections import Counter

def outliers_iqr(df, feature):
    out_indexer = []
    for i in feature:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)

        IQR = Q3-Q1

        alt_sinir = Q1 - 1.5 * IQR
        ust_sinir = Q3 + 1.5 * IQR

        out = ((df[i]<alt_sinir) | (df[i]>ust_sinir))

        out_index = df[i][out].index
        out_indexer.extend(out_index)

    out_indexer = Counter(out_indexer)

    outlier_index = [i for i, v in out_indexer.items() if v > 0]
    return outlier_index


age_outlier_index = outliers_iqr(data, ['AGE_YRS'])
data = data.drop(age_outlier_index, axis=0).reset_index(drop=True)  # 행 3개 드랍
median = data['AGE_YRS'].median()
data['AGE_YRS'].fillna(median, inplace=True)
num_outlier_index = outliers_iqr(data, ['NUMDAYS'])
data = data.drop(num_outlier_index, axis=0).reset_index(drop=True)  # 행 5884개 드랍
median = data['NUMDAYS'].median()
data['NUMDAYS'].fillna(median, inplace=True)
########################
######## STATE #########
########################
data['STATE'].fillna(method='ffill', inplace=True)
########################
###### OTHER_MEDS ######
####### CUR_ILL ########
###### ALLERGIES #######
########################

data['OTHER_MEDS'] = data['OTHER_MEDS'].str.upper()
data['CUR_ILL'] = data['CUR_ILL'].str.upper()
data['ALLERGIES'] = data['ALLERGIES'].str.upper()

data['OTHER_MEDS'].replace(
    ['NaN', '', 'NONE', 'N/A', 'NONE.', 'NA', 'NO', 'UNKNOWN', 'NONE KNOWN', 'NKA', 'NKDA', 'NONE KNOWN',
     'NONE REPORTED'], np.NaN, inplace=True)
data['CUR_ILL'].replace(
    ['NaN', 'NONE', 'N/A', 'NONE.', 'NA', 'NO', 'UNKNOWN', 'NONE KNOWN', 'NKA', 'NKDA', 'NONE KNOWN', 'NONE REPORTED'],
    np.NaN, inplace=True)
data['ALLERGIES'].replace(
    ['NaN', '', 'NONE', 'N/A', 'NONE.', 'NA', 'NO', 'UNKNOWN', 'NONE KNOWN', 'NKA', 'NKDA', 'NO KNOWN ALLERGIES',
     'NONE KNOWN', 'NONE REPORTED'], np.NaN, inplace=True)

data['OTHER_MEDS'].fillna('None', inplace=True)
data['CUR_ILL'].fillna('None', inplace=True)
data['ALLERGIES'].fillna('None', inplace=True)

# Allergies를 가지고 있는 개수로 변경
# ALL_COUNT 행 만들어서 넣고
# 이후에 ALLERGIES 삭제

data['ALL_COUNT'] = 0
for key, value in data['ALLERGIES'].iteritems():
    count = 0
    words = value.replace(' ', '')
    words = words.replace('AND', ',')
    split = words.split(',')
    for j in split:
        count += 1;
    if (value == 'None'):
        data['ALL_COUNT'].loc[key] = 0
    else:
        data['ALL_COUNT'].loc[key] = count

data.drop('ALLERGIES', axis=1, inplace=True)

# OTHER_MEDS 처리
# 있으면 1 없으면 0
for key, value in data['OTHER_MEDS'].iteritems():
    if (value == 'None'):
        data['OTHER_MEDS'][key] = 0
    else:
        data['OTHER_MEDS'][key] = 1

# CUR_ILL 처리
# 있으면 1 없으면 0
for key, value in data['CUR_ILL'].iteritems():
    if (value == 'None'):
        data['CUR_ILL'][key] = 0
    else:
        data['CUR_ILL'][key] = 1

# SYMPTOM1 categories
data['SYMPTOM1'].astype('category').cat.categories
counts = data['SYMPTOM1'].value_counts()
pd.DataFrame(counts, columns = ['symptom', 'case'])

for key, value in data['SYMPTOM1'].iteritems():
    if counts[value] < 100:
        data.drop(key, axis = 0, inplace =True)

########################
######### SEX ##########
########################

#Fill null value using method = 'ffill'
data['SEX'].replace('U', np.nan, inplace = True)
data['SEX'].fillna(method = 'ffill' , inplace=True)

########################
####### RECOVD #########
########################

data['RECOVD'].replace(['U',' '], np.nan, inplace = True)
data['RECOVD'].fillna(method = 'ffill' , inplace=True)

data = data[data.SYMPTOM1 != 'Product administered to patient of inappropriate age']

data = data[data.SYMPTOM1 != 'Incorrect dose administered']

# 부작용을 단계별로 분류하기
# 1. 매핑을 위한 딕셔너리 생성
symptom_to_levels = {
    'No adverse event': 0,
    'Unevaluable event': 0,
    'Chills': 1,
    'Dizziness': 1,
    'Fatigue': 1,
    'Headache': 1,
    'Asthenia': 1,
    'Injection site erythema': 1,
    'Erythema': 1,
    'Chest discomfort': 1,
    'Blood pressure increased': 1,
    'Anxiety': 1,
    'Body temperature increased': 1,
    'Injection site pain': 1,
    'Back pain': 1,
    'Body temperature': 1,
    'Blood test': 1,
    'Myalgia': 1,
    'Ageusia': 1,
    'Flushing': 1,
    'Atrial fibrillation': 1,
    'Feeling hot': 1,
    'Paraesthesia': 1,
    'Dysgeusia':1,
    'Arthralgia': 2,
    'Diarrhoea': 2,
    'Abdominal pain': 2,
    'Rash': 2,
    'Pruritus': 2,
    'Abdominal pain upper': 2,
    'Chest pain': 2,
    'Abdominal discomfort': 2,
    'Axillary pain': 2,
    'Condition aggravated':2,
    'Burning sensation': 2,
    'Pyrexia': 2,
    'Pyrexia': 2,
    'Nausea': 2,
    'Feeling abnormal': 2,
    'Cough': 2,
    'Pain': 2,
    'Dyspnoea': 3,
    'Hypoaesthesia': 3,
    'Pain in extremity': 3,
    'Lymphadenopathy': 3,
    'Facial paralysis': 3,
    'Aphasia': 3,
    'Death': 4,
    'COVID-19': 4,
    'Anaphylactic reaction': 4,
    'Cerebrovascular accident':4,
    'SARS-CoV-2 test positive':4
}

data['LEVEL'] = data['SYMPTOM1'].apply(lambda x: symptom_to_levels[x])

data.drop('SYMPTOM1', axis = 1, inplace = True)

############################### Spilt ###############################
from sklearn.model_selection import train_test_split, GridSearchCV

train, test = train_test_split(data, test_size=0.2, random_state=42)
############################### Scaling and Encoding ###############################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

train_num = train[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
train_cat = train.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis=1)

test_num = test[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
test_cat = test.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis=1)


############################### 1. Standard Scaler, Ordinal Encoder ###############################

def scaling(s, e):
    if s == 'StandardScaler':
        scaler = StandardScaler()
    if s == 'MinMaxScaler':
        scaler = MinMaxScaler()
    if s == 'RobustScaler':
        scaler = RobustScaler()
    if s == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    if e == 'OrdinalEncoder':
        encoder = OrdinalEncoder()
        # 인코딩
        train_cat_encoded = encoder.fit_transform(train_cat)
        test_cat_encoded = encoder.fit_transform(test_cat)
        # 데이터 프레임으로 변환
        train_cat_encoded = pd.DataFrame(train_cat_encoded, columns=list(train_cat))
        test_cat_encoded = pd.DataFrame(test_cat_encoded, columns=list(test_cat))
    if e == 'LabelEncoder':
        encoder = LabelEncoder()
        # train_cat인코딩
        train_cat_encoded_0 = encoder.fit_transform(train_cat['STATE'])
        train_cat_encoded_1 = encoder.fit_transform(train_cat['SEX'])
        train_cat_encoded_2 = encoder.fit_transform(train_cat['RECOVD'])
        train_cat_encoded_3 = encoder.fit_transform(train_cat['OTHER_MEDS'])
        train_cat_encoded_4 = encoder.fit_transform(train_cat['CUR_ILL'])
        train_cat_encoded_5 = encoder.fit_transform(train_cat['VAX_MANU'])
        train_cat_encoded_6 = encoder.fit_transform(train_cat['LEVEL'])
        # train_cat 데이터 프레임으로 변환
        train_cat_encoded_0 = pd.DataFrame(train_cat_encoded_0, columns=['STATE'])
        train_cat_encoded_1 = pd.DataFrame(train_cat_encoded_1, columns=['SEX'])
        train_cat_encoded_2 = pd.DataFrame(train_cat_encoded_2, columns=['RECOVD'])
        train_cat_encoded_3 = pd.DataFrame(train_cat_encoded_3, columns=['OTHER_MEDS'])
        train_cat_encoded_4 = pd.DataFrame(train_cat_encoded_4, columns=['CUR_ILL'])
        train_cat_encoded_5 = pd.DataFrame(train_cat_encoded_5, columns=['VAX_MANU'])
        train_cat_encoded_6 = pd.DataFrame(train_cat_encoded_6, columns=['LEVEL'])
        # train_cat 데이터프레임 합치기
        train_cat_encoded = pd.concat([train_cat_encoded_0, train_cat_encoded_1,
                                       train_cat_encoded_2, train_cat_encoded_3,
                                       train_cat_encoded_4,
                                       train_cat_encoded_5, train_cat_encoded_6], axis=1)
        # test_cat인코딩
        test_cat_encoded_0 = encoder.fit_transform(test_cat['STATE'])
        test_cat_encoded_1 = encoder.fit_transform(test_cat['SEX'])
        test_cat_encoded_2 = encoder.fit_transform(test_cat['RECOVD'])
        test_cat_encoded_3 = encoder.fit_transform(test_cat['OTHER_MEDS'])
        test_cat_encoded_4 = encoder.fit_transform(test_cat['CUR_ILL'])
        test_cat_encoded_5 = encoder.fit_transform(test_cat['VAX_MANU'])
        test_cat_encoded_6 = encoder.fit_transform(test_cat['LEVEL'])
        # test_cat 데이터 프레임으로 변환
        test_cat_encoded_0 = pd.DataFrame(test_cat_encoded_0, columns=['STATE'])
        test_cat_encoded_1 = pd.DataFrame(test_cat_encoded_1, columns=['SEX'])
        test_cat_encoded_2 = pd.DataFrame(test_cat_encoded_2, columns=['RECOVD'])
        test_cat_encoded_3 = pd.DataFrame(test_cat_encoded_3, columns=['OTHER_MEDS'])
        test_cat_encoded_4 = pd.DataFrame(test_cat_encoded_4, columns=['CUR_ILL'])
        test_cat_encoded_5 = pd.DataFrame(test_cat_encoded_5, columns=['VAX_MANU'])
        test_cat_encoded_6 = pd.DataFrame(test_cat_encoded_6, columns=['LEVEL'])
        # test_cat 데이터프레임 합치기
        test_cat_encoded = pd.concat([test_cat_encoded_0, test_cat_encoded_1,
                                      test_cat_encoded_2, test_cat_encoded_3,
                                      test_cat_encoded_4,
                                      test_cat_encoded_5, test_cat_encoded_6], axis=1)
    # 스케일링
    train_num_scaled = scaler.fit_transform(train_num)
    test_num_scaled = scaler.fit_transform(test_num)
    # 스케일링 - 데이터 프레임으로 변환
    train_num_scaled = pd.DataFrame(train_num_scaled, columns=['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'])
    test_num_scaled = pd.DataFrame(test_num_scaled, columns=['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'])
    # 스케일링, 인코딩 완료한 데이터들을 합친다
    train_prepared = pd.concat([train_num_scaled, train_cat_encoded], axis=1)
    test_prepared = pd.concat([test_num_scaled, test_cat_encoded], axis=1)
    # 스케일링, 인코딩 완료한 데이터들을 X와 y로 나눈다
    train_X = train_prepared.drop('LEVEL', axis=1)
    train_y = train_prepared['LEVEL'].copy()
    test_X = test_prepared.drop('LEVEL', axis=1)
    test_y = test_prepared['LEVEL'].copy()

    # Bagging
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True,
                                n_jobs=-1, oob_score=True)
    bag_clf.fit(train_X, train_y)
    y_pred = bag_clf.predict(test_X)
    print('#######', s, ' & ', e, '######')
    print('oob score: ', bag_clf.oob_score_)
    print('accuarcy score: ', accuracy_score(test_y, y_pred))

    some_data = test_X.iloc[:5]
    some_labels = test_y.iloc[:5]
    print('실제: ', some_labels)
    print('예측: ', bag_clf.predict(some_data))


#     bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier())
#     grid_param = {'n_estimators': [100, 300, 500, 100], 'max_features': [3, 5, 7, 9],
#               'base_estimator__max_depth': [1, 2, 3],
#               'base_estimator__criterion': ['mse', 'mae']}

#     grid_search = GridSearchCV(bag_reg, grid_param, cv=3, scoring='neg_mean_squared_error')
#     grid_search.fit(train_X, train_y.values.ravel())
#     best_parameters = grid_search.best_params_
#     best_score = grid_search.best_score_
#     print('Best parameter: ', best_parameters)
#     print('Best score: ', best_score)


scaling('StandardScaler', 'OrdinalEncoder')
scaling('MinMaxScaler', 'OrdinalEncoder')
scaling('RobustScaler', 'OrdinalEncoder')
scaling('MaxAbsScaler', 'OrdinalEncoder')
scaling('StandardScaler', 'LabelEncoder')
scaling('MinMaxScaler', 'LabelEncoder')
scaling('RobustScaler', 'LabelEncoder')
scaling('MaxAbsScaler', 'LabelEncoder')