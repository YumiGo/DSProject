import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

## Data read ## 
### The Vaccine Adverse Event Reporting System (VAERS) ###
dataFrame = pd.read_csv('2021VAERSDATA.csv')
#print("VAERSDATA.csv :\n", data.head())

symptom = pd.read_csv('2021VAERSSYMPTOMS.csv')
#print("VAERSSYMPTOM.csv :\n", symptom.head())

vax = pd.read_csv('2021VAERSVAX.csv')
vax = vax[vax.VAX_DOSE_SERIES == '1']
#print("VAERSVAX.csv :\n", vax.head())

## Merge 3 datasets into 1 dataset based on 'VAERS_ID'
dataFrame1 = pd.merge(dataFrame, symptom, on = 'VAERS_ID')
data = pd.merge(dataFrame1, vax, on = 'VAERS_ID')
data = data[data.VAX_TYPE == 'COVID19']
data=data[data.VAX_NAME.str.contains('COVID19')]

print(data.shape)

# Columns of Dataset
#print(focus1.columns)

#Cheak columns we don't use because of too many null values and duplicated values
#Check null values in each column

# missing_values_train = data.isnull().sum()
# missing_values_train = missing_values_train.to_frame(name='num_missing')
# missing_values_train['perc_missing'] = (missing_values_train['num_missing']/data.shape[0])*100
# for index, row in missing_values_train.iterrows():
#     if (row['num_missing'] > 0):
#         print ("For \"%s\" the number of missing values are: %d (%.0f%%)" %  (index,row['num_missing'],row['perc_missing']))
# print()

# Remain useful columns
# 'NUMDAYS'는 백신을 맞은 후 부작용이 일어나기 까지 시간
# 'ONSET_DATE' - 'VAX_DATE' 이다.
# 15개의 column만 사용
data = data[['STATE', 'AGE_YRS', 'SEX', 'RECOVD', 'NUMDAYS', 'OTHER_MEDS', 'CUR_ILL', 'ALLERGIES', 'SYMPTOM1','VAX_MANU']]
#print(data)

# Find columns that have non numeric values
# To check if they are useful or not
# Print object columns names
# 2: AGE_YRS 
# 5: NUMDAYS
# 두개 제외하고 전부 non numeric value

#outlier 지워주는 함수
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

########################
####### AGE_YRS ########
########################

#outlier
plt.figure(figsize=(8,6))
sns.boxplot(data = data['AGE_YRS'], color = 'orange')
#plt.show()

age_outlier_index = outliers_iqr(data, ['AGE_YRS'])
#print(data.loc[age_outlier_index, 'AGE_YRS'])

#Drop outlier data ['AGE_YRS']
data = data.drop(age_outlier_index, axis = 0).reset_index(drop=True)

median = data['AGE_YRS'].median()
data['AGE_YRS'].fillna(median, inplace = True)


########################
####### NUMDAYS ########
########################

#outlier
sns.boxplot(data = data['NUMDAYS'], color = 'blue')
#plt.show()

num_outlier_index = outliers_iqr(data, ['NUMDAYS'])
#print(data.loc[num_outlier_index, 'NUMDAYS'])

#Drop outlier data ['NUMDAYS']
data = data.drop(num_outlier_index, axis = 0).reset_index(drop=True)

median = data['NUMDAYS'].median()
data['NUMDAYS'].fillna(median, inplace = True)

########################
######## STATE #########
########################

#Fill null value using method = 'ffill'
data['STATE'].fillna(method = 'ffill' , inplace=True)

########################
###### OTHER_MEDS ######
####### CUR_ILL ########
###### ALLERGIES #######
########################

data['OTHER_MEDS'] = data['OTHER_MEDS'].str.upper()
data['CUR_ILL'] = data['CUR_ILL'].str.upper()
data['ALLERGIES'] = data['ALLERGIES'].str.upper()

data['OTHER_MEDS'].replace(['','NONE','N/A','NONE.','NA','NO','UNKNOWN','NONE KNOWN','NKA','NKDA','NONE KNOWN','NONE REPORTED'], np.nan, inplace = True)
data['CUR_ILL'].replace(['','NONE','N/A','NONE.','NA','NO','UNKNOWN','NONE KNOWN','NKA','NKDA','NONE KNOWN','NONE REPORTED'], np.nan, inplace = True)
data['ALLERGIES'].replace(['','NONE','N/A','NONE.','NA','NO','UNKNOWN','NONE KNOWN','NKA','NKDA','NO KNOWN ALLERGIES','NONE KNOWN','NONE REPORTED'], np.nan, inplace = True)

data['OTHER_MEDS'].fillna('None', inplace = True)
data['CUR_ILL'].fillna('None', inplace = True)
data['ALLERGIES'].fillna('None', inplace = True)


# Allergies를 가지고 있는 개수로 변경
# ALL_COUNT 행 만들어서 넣고 
# 이후에 ALLERGIES 삭제

data['ALL_COUNT'] = 0
for key, value in data['ALLERGIES'].iteritems():
    count = 0
    words = value.replace(' ','')
    words = words.replace('AND',',')
    split = words.split(',')

    for j in split:
        count += 1
    
    if(value == 'None'):
        data['ALL_COUNT'].loc[key] = 0
    else:
        data['ALL_COUNT'].loc[key] = count
    
data.drop('ALLERGIES', axis = 1, inplace = True)

#Drop outlier data ['ALL_COUNT']
count_outlier_index = outliers_iqr(data, ['NUMDAYS'])

data = data.drop(count_outlier_index, axis = 0).reset_index(drop=True)

median = data['ALL_COUNT'].median()
data['ALL_COUNT'].fillna(median, inplace = True)

# OTHER_MEDS 처리 
# 있으면 1 없으면 0
for key, value in data['OTHER_MEDS'].iteritems():
    if (value == 'None'):
        data['OTHER_MEDS'][key] = 0
    else:
        data['OTHER_MEDS'][key] = 1
print(data['OTHER_MEDS'].value_counts())

        
# CUR_ILL 처리 
# 있으면 1 없으면 0
for key, value in data['CUR_ILL'].iteritems():
    if (value == 'None'):
        data['CUR_ILL'][key] = 0
    else:
        data['CUR_ILL'][key] = 1
print(data['CUR_ILL'].value_counts())

######################### SYMPTOM1 카테고리 정리########################
data['SYMPTOM1'].astype('category').cat.categories
counts = data['SYMPTOM1'].value_counts()
pd.DataFrame(counts, columns = ['symptom', 'case'])

for key, value in data['SYMPTOM1'].iteritems():
    if counts[value] < 100:
        data.drop(key, axis = 0, inplace =True)
# 이 2개는 부작용이 아니라 의료진 실수임 -> 드랍
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

########################
######### SEX ##########
########################

#Fill null value using method = 'ffill'
data['SEX'].replace('U', np.nan, inplace = True)
data['SEX'].fillna(method = 'ffill' , inplace=True)
#print(data['SEX'].value_counts())

########################
####### RECOVD #########
########################

data['RECOVD'].replace(['U',' '], np.nan, inplace = True)
data['RECOVD'].fillna(method = 'ffill' , inplace=True)
#print(data['RECOVD'].value_counts())

## SYMPTOM1 가보자 ##


#-------------------- 이후에는 인코딩 및 스케일링 -----------------------

ot = pd.DataFrame(data.dtypes == 'object').reset_index()
object_type = ot[ot[0] == True]['index']
print("object type: ",object_type)
print()

# Find column that have numeric values
# To check if they are useful or not
num_type = pd.DataFrame(data.dtypes != 'object').reset_index().rename(columns={0: 'yes/no'})
num_type = num_type[num_type['yes/no'] == True]['index']
print("num_type",num_type)

############################### Scaling and Encoding ######################
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
data_label = data.copy()
data_onehot = data.copy()

# Label encoder
lable_encoder = LabelEncoder()
for x in data_label:
    if data_label[x].dtypes == 'object':
        data_label[x] = lable_encoder.fit_transform(data_label[x])
   
print(data_label.head())

X = data_label[['STATE','AGE_YRS','SEX','RECOVD','NUMDAYS','OTHER_MEDS','CUR_ILL','VAX_MANU','ALL_COUNT']]
y = data_label['SYMPTOM1']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# -------------------------------------------- StandardScaler ----------------
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train_scale = ss.fit_transform(X_train)
X_test_scale = ss.transform(X_test)

# Set HyperParameters of KNeighborsClassifier
grid_params_knn = {
    'n_neighbors': np.arange(3, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

gs_knn = GridSearchCV(KNeighborsClassifier(), grid_params_knn, verbose=1, cv=5, n_jobs=-1)
gs_knn.fit(X_train_scale, y_train)

# Show the model performance of train set
print("Standard Scaler, KNN Classifier")
print("best_parameter: ", gs_knn.best_params_)
print("best_train_score: ", gs_knn.best_score_)


grid_params_dt = {
    'min_samples_split': [2, 3, 4],
    'max_features': [3, 5, 7],
    'max_depth': [3, 5, 7],
    'max_leaf_nodes': list(range(7, 100))
}

# Make GridSearchCV with DecisionTreeClassifier
# make model by fit train dataset
gs_dt = GridSearchCV(DecisionTreeClassifier(), grid_params_dt, verbose=1, cv=3, n_jobs=-1)
gs_dt.fit(X_train_scale, y_train)

# Show the model performance of train set
print("Standard Scaler, DecisionTree Classifier")
print("best_parameter: ", gs_dt.best_params_)
print("best_train_score: %.2f" % gs_dt.best_score_)

# Show the score of model from test set
dt_score = gs_dt.score(X_test, y_test)
print("test_score: %.2f" % dt_score)
print()

# -------------------------------------------- RobustScaler ----------------

# Robust Scaling
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()
X_train_scale = rb.fit_transform(X_train)
X_test_scale = rb.transform(X_test)

grid_params_knn = {
    'n_neighbors': np.arange(3, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

gs_knn = GridSearchCV(KNeighborsClassifier(), grid_params_knn, verbose=1, cv=5, n_jobs=-1)
gs_knn.fit(X_train_scale, y_train)

# Show the model performance of train set
print("Robust Scaler, KNN Classifier")
print("best_parameter: ", gs_knn.best_params_)
print("best_train_score: %.2f" % gs_knn.best_score_)

# Show the score of model from test set
knn_score = gs_knn.score(X_test, y_test)
print("test_score: %.2f" % knn_score)
print()