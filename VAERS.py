import pandas as pd
import numpy as np
import scaling_encoding_cases as se
from collections import Counter
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


class VAERS:
    # function: __init__()
    # input: none
    # output: none
    # description: This function loads the VARES dataset and stores it in a DataFrame
    def __init__(self):
        import pandas as pd
        data_frame = pd.read_csv('2021VAERSDATA.csv', encoding='ISO-8859-1', low_memory=False)
        symptom = pd.read_csv('2021VAERSSYMPTOMS.csv')
        symptom.drop_duplicates(subset=['VAERS_ID'], keep='first', inplace=True, ignore_index=False)
        vax = pd.read_csv('2021VAERSVAX.csv')
        vax.drop_duplicates(subset=['VAERS_ID'], keep='first', inplace=True, ignore_index=False)
        data_frame1 = pd.merge(data_frame, symptom, on='VAERS_ID')
        self.data = pd.merge(data_frame1, vax, on='VAERS_ID')
        self.data = self.data[self.data.VAX_TYPE == 'COVID19']
        self.data = self.data[self.data.VAX_NAME.str.contains('COVID19')]
        self.data = self.data[
            ['STATE', 'AGE_YRS', 'SEX', 'RECOVD', 'NUMDAYS', 'OTHER_MEDS', 'CUR_ILL', 'ALLERGIES', 'VAX_MANU',
             'SYMPTOM1']]

    def __getitem__(self, key):
        return getattr(self, key)

    # function: preprocess
    # input: none
    # output: none
    # description: This function performs the preprocessing of the VARES dataset previously obtained from the get_data() function.
    def preprocess(self):
        # AGE_YRS
        age_outlier_index = outliers_iqr(self.data, ['AGE_YRS'])
        # drop outliers in data feature'AGE_YRS'
        self.data = self.data.drop(age_outlier_index, axis=0).reset_index(drop=True)
        median = self.data['AGE_YRS'].median()
        self.data['AGE_YRS'].fillna(median, inplace=True)
        # NUMDAYS
        num_outlier_index = outliers_iqr(self.data, ['NUMDAYS'])
        # drop doutliers in feature 'NUMDAYS'
        self.data = self.data.drop(num_outlier_index, axis=0).reset_index(drop=True)
        median = self.data['NUMDAYS'].median()
        self.data['NUMDAYS'].fillna(median, inplace=True)
        # STATE
        self.data['STATE'].fillna(method='ffill', inplace=True)
        # OTHER_MEDS, CUR_ILL, ALLERGIES
        # Change the data to all uppercase.
        self.data['OTHER_MEDS'] = self.data['OTHER_MEDS'].str.upper()
        self.data['CUR_ILL'] = self.data['CUR_ILL'].str.upper()
        self.data['ALLERGIES'] = self.data['ALLERGIES'].str.upper()
        # Replace all words that mean no data with "np.NaN".
        self.data['OTHER_MEDS'].replace(
            ['NaN', '', 'NONE', 'N/A', 'NONE.', 'NA', 'NO', 'UNKNOWN', 'NONE KNOWN', 'NKA', 'NKDA', 'NONE KNOWN',
             'NONE REPORTED'], np.NaN, inplace=True)
        self.data['CUR_ILL'].replace(
            ['NaN', 'NONE', 'N/A', 'NONE.', 'NA', 'NO', 'UNKNOWN', 'NONE KNOWN', 'NKA', 'NKDA', 'NONE KNOWN',
             'NONE REPORTED'], np.NaN, inplace=True)
        self.data['ALLERGIES'].replace(
            ['NaN', '', 'NONE', 'N/A', 'NONE.', 'NA', 'NO', 'UNKNOWN', 'NONE KNOWN', 'NKA', 'NKDA',
             'NO KNOWN ALLERGIES', 'NONE KNOWN', 'NONE REPORTED'], np.NaN, inplace=True)
        # Replace all NaN with the word "None".
        self.data['OTHER_MEDS'].fillna('None', inplace=True)
        self.data['CUR_ILL'].fillna('None', inplace=True)
        self.data['ALLERGIES'].fillna('None', inplace=True)

        # ALLERGIES
        # Replace the allergy data with the number of allergies and put it in a new column called ALL_COUNT.
        self.data['ALL_COUNT'] = 0
        for key, value in self.data['ALLERGIES'].iteritems():
            count = 0
            words = value.replace(' ', '')
            words = words.replace('AND', ',')
            split = words.split(',')
            for j in split:
                count += 1
            if value == 'None':
                self.data['ALL_COUNT'].loc[key] = 0
            else:
                self.data['ALL_COUNT'].loc[key] = count
        # Drop column ALLERGIES
        self.data.drop('ALLERGIES', axis=1, inplace=True)
        # OTHER_MEDS
        # Treat as 1 if data exists and 0 if data is not present
        for key, value in self.data['OTHER_MEDS'].iteritems():
            if value == 'None':
                self.data['OTHER_MEDS'][key] = 0
            else:
                self.data['OTHER_MEDS'][key] = 1

        # CUR_ILL
        # Treat as 1 if data exists and 0 if data is not present
        for key, value in self.data['CUR_ILL'].iteritems():
            if value == 'None':
                self.data['CUR_ILL'][key] = 0
            else:
                self.data['CUR_ILL'][key] = 1
        # SEX
        # Fill null value using method = 'ffill'
        self.data['SEX'].replace('U', np.nan, inplace=True)
        self.data['SEX'].fillna(method='ffill', inplace=True)

        # RECOVD
        # Fill null value using method = 'ffill'
        self.data['RECOVD'].replace(['U', ' '], np.nan, inplace=True)
        self.data['RECOVD'].fillna(method='ffill', inplace=True)
        # SYMPTOM1
        # Create a data frame count for the number of symptoms.
        counts = self.data['SYMPTOM1'].value_counts()
        pd.DataFrame(counts, columns=['symptom', 'case'])
        # Remove symptoms expressed in less than 100 people.
        for key, value in self.data['SYMPTOM1'].iteritems():
            if counts[value] < 100:
                self.data.drop(key, axis=0, inplace=True)
        # Delete data unrelated to adverse vaccine symptoms
        self.data = self.data[self.data.SYMPTOM1 != 'Product administered to patient of inappropriate age']
        self.data = self.data[self.data.SYMPTOM1 != 'Incorrect dose administered']
        # Categorize side effects into stages
        # Create a dictionary for mapping
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
            'Dysgeusia': 1,
            'Arthralgia': 2,
            'Diarrhoea': 2,
            'Abdominal pain': 2,
            'Rash': 2,
            'Pruritus': 2,
            'Abdominal pain upper': 2,
            'Chest pain': 2,
            'Abdominal discomfort': 2,
            'Axillary pain': 2,
            'Condition aggravated': 2,
            'Burning sensation': 2,
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
            'Cerebrovascular accident': 4,
            'SARS-CoV-2 test positive': 4
        }
        # Replace SYMPTOM1 with LEVEL Column
        self.data['LEVEL'] = self.data['SYMPTOM1'].apply(lambda x: symptom_to_levels[x])
        self.data.drop('SYMPTOM1', axis=1, inplace=True)

    def split(self, test=0.2, random=42):
        X = self.data.drop("LEVEL", axis=1)
        y = self.data["LEVEL"].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=random)

        X_train_num = X_train[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
        X_train_cat = X_train.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis=1)
        X_test_num = X_test[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
        X_test_cat = X_test.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis=1)
        self.X_train_num = X_train_num
        self.X_train_cat = X_train_cat
        self.y_train = y_train
        self.X_test_num = X_test_num
        self.X_test_cat = X_test_cat
        self.y_test = y_test

        return X_train_num, X_train_cat, y_train, X_test_num, X_test_cat, y_test

    def scaling_encoding(self, s, e):
        num_attribs = list(self.X_train_num)
        cat_attribs = list(self.X_train_cat)
        cat_len = len(self.X_train_cat.columns)
        if s == 'Standard':
            scaler = StandardScaler()
        if s == 'MinMax':
            scaler = MinMaxScaler()
        if s == 'Robust':
            scaler = RobustScaler()
        if s == 'MaxAbs':
            scaler = MaxAbsScaler()
        if e == 'Ordinal':
            encoder = OrdinalEncoder()
            # 인코딩
            X_train_cat_encoded = encoder.fit_transform(self.X_train_cat)
            X_test_cat_encoded = encoder.fit_transform(self.X_test_cat)
            # 데이터 프레임으로 변환
            X_train_cat_encoded = pd.DataFrame(X_train_cat_encoded, columns=list(self.X_train_cat))
            X_test_cat_encoded = pd.DataFrame(X_test_cat_encoded, columns=list(self.X_test_cat))
        if e == 'Label':
            encoder = LabelEncoder()
            list_of_train = []
            list_of_test = []
            for i in range(0, cat_len):
                globals()['X_train_cat_encoded{}'.format(i)] = encoder.fit_transform(self.X_train_cat[cat_attribs[i]])
                globals()['X_train_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_train_cat_encoded{}'.format(i)],
                    columns=[cat_attribs[i]])
                globals()['X_test_cat_encoded{}'.format(i)] = encoder.fit_transform(self.X_test_cat[cat_attribs[i]])
                globals()['X_test_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_test_cat_encoded{}'.format(i)],
                                                                           columns=[cat_attribs[i]])
                list_of_train.append(globals()['X_train_cat_encoded{}'.format(i)])
                list_of_test.append(globals()['X_test_cat_encoded{}'.format(i)])

            X_train_cat_encoded = pd.concat(list_of_train, ignore_index=True, axis=1)
            X_test_cat_encoded = pd.concat(list_of_test, ignore_index=True, axis=1)
        # 스케일링
        X_train_num_scaled = scaler.fit_transform(self.X_train_num)
        X_test_num_scaled = scaler.fit_transform(self.X_test_num)
        # 스케일링 - 데이터 프레임으로 변환
        X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=num_attribs)
        X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=num_attribs)
        # 스케일링, 인코딩 완료한 데이터들을 합친다
        self.X_train_prepared = pd.concat([X_train_num_scaled, X_train_cat_encoded], axis=1)
        self.X_test_prepared = pd.concat([X_test_num_scaled, X_test_cat_encoded], axis=1)
        self.bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=300, max_samples=100,
                                    bootstrap=True,
                                    max_features=9, n_jobs=-1)
        self.bag_clf.fit(self.X_train_prepared, self.y_train)


    def predict(self, num):
        some_data = self.X_test_prepared.iloc[:num]
        some_labels = self.y_test.iloc[:num]
        print('Actual Data: ', some_labels.values)
        print('Predict: ', self.bag_clf.predict(some_data))

# function: outliers_iqr
# input: Dataframe, a feature of the dataframe
# output: 1D array
# description: This function returns an array of indexes of rows where the outlier exists.
def outliers_iqr(df, feature):
    out_indexer = []
    for i in feature:
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 - q1
        alt_sinir = q1 - 1.5 * iqr
        ust_sinir = q3 + 1.5 * iqr
        out = ((df[i] < alt_sinir) | (df[i] > ust_sinir))
        out_index = df[i][out].index
        out_indexer.extend(out_index)
    out_indexer = Counter(out_indexer)
    outlier_index = [i for i, v in out_indexer.items() if v > 0]
    return outlier_index

#Example Execution Code
data = VAERS()  # Class Declaration
data.preprocess()  # Use preprocessing function - suppose three csv files are in the directory
X_train_num, X_train_cat, y_train, X_test_num, X_test_cat, y_test = data.split() # Split the data
# Print the accuracy of all possible scaling and encoding combinations
# table = se.scaling_encoding_cases(X_train_num, X_train_cat, y_train, X_test_num, X_test_cat,  y_test, 'Standard', 'MinMax', 'MaxAbs', 'Robust', 'Ordinal', 'Label')
table = se.scaling_encoding_cases(X_train_num, X_train_cat, y_train, X_test_num, X_test_cat,  y_test, '' ,'', 'MaxAbs', '', '', 'Label')
print(table) # MinMax Scaling, Label encoding is the best.

#data.scaling_encoding('MinMax', 'Label')
#data.predict(10)  # Print 10 predictions
