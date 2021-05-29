import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## Data read ## 
### The Vaccine Adverse Event Reporting System (VAERS) ###
from sklearn.model_selection import train_test_split

dataFrame = pd.read_csv('2021VAERSDATA.csv')
#print("VAERSDATA.csv :\n", data.head())

symptom = pd.read_csv('2021VAERSSYMPTOMS.csv')
#print("VAERSSYMPTOM.csv :\n", symptom.head())

vax = pd.read_csv('2021VAERSVAX.csv')
#print("VAERSVAX.csv :\n", vax.head())

## Merge 3 datasets into 1 dataset based on 'VAERS_ID'
dataFrame1 = pd.merge(dataFrame, symptom, on = 'VAERS_ID')
data = pd.merge(dataFrame1, vax, on = 'VAERS_ID')
print(data.shape)
data=data[data.VAX_TYPE == 'COVID19']
print(data.shape)
data=data[data.VAX_NAME.str.contains('COVID19')]
print(data.shape)

data = data[['STATE', 'AGE_YRS', 'SEX', 'RECOVD', 'NUMDAYS', 'OTHER_MEDS', 'CUR_ILL', 'ALLERGIES', 'SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5','VAX_MANU']]
#print(data)

# Find columns that have non numeric values
# To check if they are useful or not
# Print object columns names
# 0: VAERS_ID
# 2: AGE_YRS
# 5: NUMDAYS
# 세개 제외하고 전부 non numeric value

ot = pd.DataFrame(data.dtypes == 'object').reset_index()
object_type = ot[ot[0] == True]['index']
#print(object_type)
#print()

# Find column that have numeric values
# To check if they are useful or not
num_type = pd.DataFrame(data.dtypes != 'object').reset_index().rename(columns={0: 'yes/no'})
num_type = num_type[num_type['yes/no'] == True]['index']
#print(num_type)
#print(data[num_type]['AGE_YRS'].value_counts())

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
print(data.loc[age_outlier_index, 'AGE_YRS'])

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
print(data.loc[num_outlier_index, 'NUMDAYS'])

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

########################
######### SEX ##########
########################

#Fill null value using method = 'ffill'
data['SEX'].replace('U', np.nan, inplace = True)
data['SEX'].fillna(method = 'ffill' , inplace=True)
print(data['SEX'].value_counts())

########################
####### RECOVD #########
########################

data['RECOVD'].replace(['U',' '], np.nan, inplace = True)
data['RECOVD'].fillna(method = 'ffill' , inplace=True)
print(data['RECOVD'].value_counts())
print(data['SYMPTOM1'].value_counts())

x_data = ['AGE_YRS', 'SEX', 'RECOVD', 'NUMDAYS', 'OTHER_MEDS', 'CUR_ILL', 'ALLERGIES']
numeric_x = ['AGE_YRS', 'NUMDAYS']
categorical_x= ['SEX', 'RECOVD', 'OTHER_MEDS', 'CUR_ILL', 'ALLERGIES']
target = 'SYMPTOM1'

from sklearn import preprocessing

minMaxScaler = preprocessing.MinMaxScaler()
minMaxScaler.fit(data[numeric_x])
data[numeric_x].iloc[:,0:9] = minMaxScaler.transform(data[numeric_x].iloc[:,0:9])
# categorical data encoding
for i in categorical_x :
    enc= preprocessing.OrdinalEncoder()
    enc.fit(data[i].to_numpy().reshape(-1, 1))
    data[i] = enc.transform(data[i].to_numpy().reshape(-1, 1))
enc = preprocessing.OrdinalEncoder()
enc.fit(data[target].to_numpy().reshape(-1, 1))
data[target] = enc.transform(data[target].to_numpy().reshape(-1, 1))

train_X, test_X, train_y, test_y = train_test_split(data[x_data], data[target], test_size=0.2, shuffle=True)

from sklearn import tree

model = tree.DecisionTreeClassifier()

model = model.fit(train_X,train_y)

print(model.score(test_X,test_y))