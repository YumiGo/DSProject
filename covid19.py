import pandas as pd
# 파일 경로 각자 수정하고 주석처리
#filename = '/content/2021VAERSDATA.csv'

#정규
Path_JG = '/Users/kim-jeonggyu/Python/Dataset/2021VAERS/2021VAERSDATA.csv'
data = pd.read_csv(Path_JG)

# load data
# filename = 'C:/Users/USER/Desktop/2021VAERSDATA.csv'
#data = pd.read_csv(filename)

# drop missing values
missing_data = ['CAGE_MO', 'RPT_DATE', 'ER_VISIT', 'X_STAY', 'V_FUNDBY', 'BIRTH_DEFECT']
data.drop(missing_data, axis = 1, inplace=True)

print(data.head())
