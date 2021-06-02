import pandas as pd
import numpy as np
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



    def preprocess(self):
        age_outlier_index = outliers_iqr(self.data, ['AGE_YRS'])
        self.data = self.data.drop(age_outlier_index, axis=0).reset_index(drop=True)  # 행 3개 드랍
        median = self.data['AGE_YRS'].median()
        self.data['AGE_YRS'].fillna(median, inplace=True)
        num_outlier_index = outliers_iqr(self.data, ['NUMDAYS'])
        self.data = self.data.drop(num_outlier_index, axis=0).reset_index(drop=True)  # 행 5884개 드랍
        median = self.data['NUMDAYS'].median()
        self.data['NUMDAYS'].fillna(median, inplace=True)
        # STATE
        self.data['STATE'].fillna(method='ffill', inplace=True)
        # OTHER_MEDS, CUR_ILL, ALLERGIES
        self.data['OTHER_MEDS'] = self.data['OTHER_MEDS'].str.upper()
        self.data['CUR_ILL'] = self.data['CUR_ILL'].str.upper()
        self.data['ALLERGIES'] = self.data['ALLERGIES'].str.upper()

        self.data['OTHER_MEDS'].replace(
            ['NaN', '', 'NONE', 'N/A', 'NONE.', 'NA', 'NO', 'UNKNOWN', 'NONE KNOWN', 'NKA', 'NKDA', 'NONE KNOWN',
             'NONE REPORTED'], np.NaN, inplace=True)
        self.data['CUR_ILL'].replace(
            ['NaN', 'NONE', 'N/A', 'NONE.', 'NA', 'NO', 'UNKNOWN', 'NONE KNOWN', 'NKA', 'NKDA', 'NONE KNOWN',
             'NONE REPORTED'], np.NaN, inplace=True)
        self.data['ALLERGIES'].replace(
            ['NaN', '', 'NONE', 'N/A', 'NONE.', 'NA', 'NO', 'UNKNOWN', 'NONE KNOWN', 'NKA', 'NKDA',
             'NO KNOWN ALLERGIES', 'NONE KNOWN', 'NONE REPORTED'], np.NaN, inplace=True)

        self.data['OTHER_MEDS'].fillna('None', inplace=True)
        self.data['CUR_ILL'].fillna('None', inplace=True)
        self.data['ALLERGIES'].fillna('None', inplace=True)

        # Allergies를 가지고 있는 개수로 변경
        # ALL_COUNT 행 만들어서 넣고
        # 이후에 ALLERGIES 삭제

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

        self.data.drop('ALLERGIES', axis=1, inplace=True)

        # OTHER_MEDS 처리
        # 있으면 1 없으면 0
        for key, value in self.data['OTHER_MEDS'].iteritems():
            if value == 'None':
                self.data['OTHER_MEDS'][key] = 0
            else:
                self.data['OTHER_MEDS'][key] = 1

        # CUR_ILL 처리
        # 있으면 1 없으면 0
        for key, value in self.data['CUR_ILL'].iteritems():
            if value == 'None':
                self.data['CUR_ILL'][key] = 0
            else:
                self.data['CUR_ILL'][key] = 1

        counts = self.data['SYMPTOM1'].value_counts()

        pd.DataFrame(counts, columns=['symptom', 'case'])

        for key, value in self.data['SYMPTOM1'].iteritems():
            if counts[value] < 100:
                self.data.drop(key, axis=0, inplace=True)
        # SEX
        # Fill null value using method = 'ffill'
        self.data['SEX'].replace('U', np.nan, inplace=True)
        self.data['SEX'].fillna(method='ffill', inplace=True)

        # RECOVD
        self.data['RECOVD'].replace(['U', ' '], np.nan, inplace=True)
        self.data['RECOVD'].fillna(method='ffill', inplace=True)

        self.data = self.data[self.data.SYMPTOM1 != 'Product administered to patient of inappropriate age']

        self.data = self.data[self.data.SYMPTOM1 != 'Incorrect dose administered']
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
        return X_train_num, X_train_cat, y_train, X_test_num, X_test_cat, y_test

    def scaling_encoding(self, X_train_num, X_train_cat, y_train,  X_test_num, X_test_cat, y_test, s1, s2,
                         s3, s4, e1, e2=None):
        s = []
        e = []
        scaler = []
        num_attribs = list(X_train_num)
        cat_attribs = list(X_train_cat)
        cat_len = len(X_train_cat.columns)

        count_scaler = 0
        count_encoder = 0

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
            # 인코딩
            X_train_cat_encoded = OrdinalEncoder().fit_transform(X_train_cat)
            X_test_cat_encoded = OrdinalEncoder().fit_transform(X_test_cat)
            # 데이터 프레임으로 변환
            X_train_cat_encoded = pd.DataFrame(X_train_cat_encoded, columns=list(X_train_cat))
            X_test_cat_encoded = pd.DataFrame(X_test_cat_encoded, columns=list(X_test_cat))
            count_encoder = count_encoder + 1
        if e1 == 'Label':
            e.append(e1)
            encoder = LabelEncoder()
            count_encoder = count_encoder + 1
            for i in range(0, cat_len):
                globals()['X_train_cat_encoded{}'.format(i)] = encoder.fit_transform(X_train_cat[cat_attribs[i]])
                globals()['X_train_cat_encoded{}'.format(i)] = pd.DataFrame(
                    globals()['X_train_cat_encoded{}'.format(i)],
                    columns=[cat_attribs[i]])
                globals()['X_test_cat_encoded{}'.format(i)] = encoder.fit_transform(X_test_cat[cat_attribs[i]])
                globals()['X_test_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_test_cat_encoded{}'.format(i)],
                                                                           columns=[cat_attribs[i]])
                X_train_cat_encoded = pd.concat([globals()['X_test_cat_encoded{}'.format(i)]], axis=1)
                X_test_cat_encoded = pd.concat([globals()['X_test_cat_encoded{}'.format(i)]], axis=1)
        if e2 == 'Ordinal':
            e.append(e2)
            # 인코딩
            X_train_cat_encoded = OrdinalEncoder().fit_transform(X_train_cat)
            X_test_cat_encoded = OrdinalEncoder().fit_transform(X_test_cat)
            # 데이터 프레임으로 변환
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
                globals()['X_train_cat_encoded{}'.format(i)] = pd.DataFrame(
                    globals()['X_train_cat_encoded{}'.format(i)],
                    columns=[cat_attribs[i]])
                globals()['X_test_cat_encoded{}'.format(i)] = encoder.fit_transform(X_test_cat[cat_attribs[i]])
                globals()['X_test_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_test_cat_encoded{}'.format(i)],
                                                                           columns=[cat_attribs[i]])
                list_of_train.append(globals()['X_train_cat_encoded{}'.format(i)])
                list_of_test.append(globals()['X_test_cat_encoded{}'.format(i)])

            X_train_cat_encoded = pd.concat(list_of_train, ignore_index=True, axis=1)
            X_test_cat_encoded = pd.concat(list_of_test, ignore_index=True, axis=1)

        count_score = 0
        scaler_encoder = []
        score = []
        for i in range(0, count_scaler):
            for j in range(0, count_encoder):
                string = s[i] + '&' + e[j]
                scaler_encoder.append(string)
                # 스케일링
                X_train_num_scaled = scaler[i].fit_transform(X_train_num)
                X_test_num_scaled = scaler[i].fit_transform(X_test_num)
                # 스케일링 - 데이터 프레임으로 변환
                X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=num_attribs)
                X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=num_attribs)
                # 스케일링, 인코딩 완료한 데이터들을 합친다
                X_train_prepared = pd.concat([X_train_num_scaled, X_train_cat_encoded], axis=1)
                X_test_prepared = pd.concat([X_test_num_scaled, X_test_cat_encoded], axis=1)
                bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=300, max_samples=100,
                                            bootstrap=True,
                                            max_features=9, n_jobs=-1)
                self.bag_clf = bag_clf
                self.X_train_prepared = X_train_prepared
                self.y_test = y_test
                bag_clf.fit(X_train_prepared, y_train)
                y_pred = bag_clf.predict(X_test_prepared)
                score.append(accuracy_score(y_test, y_pred))
                count_score = count_score + 1

        for i in range(count_score - 1):  # 리스트의 크기-1만큼 반복
            for j in range(i + 1, len(score)):  # 해당 인덱스+1부터, 리스트 크기만큼 반복
                if score[i] < score[j]:  # 인덱스의 값이 비교 인덱스보다 더 크다면
                    score[i], score[j] = score[j], score[i]  # swap 해주기
                    scaler_encoder[i], scaler_encoder[j] = scaler_encoder[j], scaler_encoder[i]
        result = []
        for i in range(0, count_score):
            a = []
            a.append(i + 1)
            a.append(scaler_encoder[i])
            a.append(score[i])
            result.append(a)

        table = pd.DataFrame(result, columns=['Rank', 'Scaler & Encoder', 'Accuracy'])
        return table


    def predict(self, num):
        some_data = self.X_train_prepared.iloc[:num]
        some_labels = self.y_test.iloc[:num]
        print('Actual Data: ', some_labels.values)
        print('Predict: ', self.bag_clf.predict(some_data))


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

#예시 실행 코드
data = VAERS()  # 클래스 선언
data.preprocess()  # 전처리 함수 사용 - 이 때 csv 파일 3개가 디렉토리에 있다고 가정함
X_train_num, X_train_cat, y_train, X_test_num, X_test_cat, y_test = data.split()
table = data.scaling_encoding(X_train_num, X_train_cat, y_train, X_test_num, X_test_cat,  y_test, 'Standard', 'MinMax',
                         'MaxAbs', 'Robust', 'Ordinal', 'Label')
print(table)
data.predict(10)  # 10개 예측 출력
