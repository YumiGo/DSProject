import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 경로 각자 수정하고 주석처리
#filename = '/content/2021VAERSDATA.csv'



## 각자 경로 ##
#정규
Path = '/Users/kim-jeonggyu/Python/Dataset/2021VAERS'
#유미

#형종

#종민


## Data read ## 
### The Vaccine Adverse Event Reporting System (VAERS) ###
df = pd.read_csv("".join([Path,'/2021VAERSDATA.csv']))
#print("VAERSDATA.csv :\n", data.head())

symptom = pd.read_csv("".join([Path,'/2021VAERSSYMPTOMS.csv']))
#print("VAERSSYMPTOM.csv :\n", symptom.head())

vax = pd.read_csv("".join([Path,'/2021VAERSVAX.csv']))
#print("VAERSVAX.csv :\n", vax.head())

## Merge 3 datasets into 1 dataset based on 'VAERS_ID'
df1 = pd.merge(df, symptom, on = 'VAERS_ID')
data = pd.merge(df1, vax, on = 'VAERS_ID')

# Columns of Dataset
#print(focus1.columns)

#Cheak columns we don't use because of too many null values and duplicated values
#Check null values in each column

missing_values_train = data.isnull().sum()
missing_values_train = missing_values_train.to_frame(name='num_missing')
missing_values_train['perc_missing'] = (missing_values_train['num_missing']/data.shape[0])*100
for index, row in missing_values_train.iterrows():
    if (row['num_missing'] > 0):
        print ("For \"%s\" the number of missing values are: %d (%.0f%%)" %  (index,row['num_missing'],row['perc_missing']))
print()

# Remain useful columns
# 'NUMDAYS'는 백신을 맞은 후 부작용이 일어나기 까지 시간
# 'ONSET_DATE' - 'VAX_DATE' 이다.
# 15개의 column만 사용
data = data[['VAERS_ID', 'STATE', 'AGE_YRS', 'SEX', 'RECOVD', 'NUMDAYS', 'OTHER_MEDS', 'CUR_ILL', 'ALLERGIES', 'SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5','VAX_MANU']]
#print(data)

# Find columns that have non numeric values
# To check if they are useful or not
# Print object columns names
# 0: VAERS_ID
# 3: AGE_YRS 
# 6: NUMDAYS
# 세개 제외하고 전부 non numeric value

ot = pd.DataFrame(data.dtypes == 'object').reset_index()
object_type = ot[ot[0] == True]['index']
print(object_type)
print()

# Find column that have numeric values
# To check if they are useful or not
num_type = pd.DataFrame(data.dtypes != 'object').reset_index().rename(columns={0: 'yes/no'})
num_type = num_type[num_type['yes/no'] == True]['index']






