from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


class VAERS:
    def __init__(self):
        ## Data read ##
        import pandas as pd
        dataFrame = pd.read_csv('2021VAERSDATA.csv', encoding='ISO-8859-1', low_memory=False)
        symptom = pd.read_csv('2021VAERSSYMPTOMS.csv')
        symptom.drop_duplicates(subset=['VAERS_ID'], keep='first', inplace=True, ignore_index=False)
        vax = pd.read_csv('2021VAERSVAX.csv')
        vax.drop_duplicates(subset=['VAERS_ID'], keep='first', inplace=True, ignore_index=False)
        dataFrame1 = pd.merge(dataFrame, symptom, on='VAERS_ID')
        self.data = pd.merge(dataFrame1, vax, on='VAERS_ID')
        self.data = self.data[self.data.VAX_TYPE == 'COVID19']
        self.data = self.data[self.data.VAX_NAME.str.contains('COVID19')]
        self.data = self.data[
            ['STATE', 'AGE_YRS', 'SEX', 'RECOVD', 'NUMDAYS', 'OTHER_MEDS', 'CUR_ILL', 'ALLERGIES', 'VAX_MANU',
             'SYMPTOM1']]
    @staticmethod
    def outliers_iqr(df, feature):
        out_indexer = []
        for i in feature:
            Q1 = df[i].quantile(0.25)
            Q3 = df[i].quantile(0.75)

            IQR = Q3 - Q1

            alt_sinir = Q1 - 1.5 * IQR
            ust_sinir = Q3 + 1.5 * IQR

            out = ((df[i] < alt_sinir) | (df[i] > ust_sinir))

            out_index = df[i][out].index
            out_indexer.extend(out_index)

        out_indexer = Counter(out_indexer)

        outlier_index = [i for i, v in out_indexer.items() if v > 0]
        return outlier_index

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
        ########################
        ###### OTHER_MEDS ######
        ####### CUR_ILL ########
        ###### ALLERGIES #######
        ########################

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
                count += 1;
            if (value == 'None'):
                self.data['ALL_COUNT'].loc[key] = 0
            else:
                self.data['ALL_COUNT'].loc[key] = count

        self.data.drop('ALLERGIES', axis=1, inplace=True)

        # OTHER_MEDS 처리
        # 있으면 1 없으면 0
        for key, value in self.data['OTHER_MEDS'].iteritems():
            if (value == 'None'):
                self.data['OTHER_MEDS'][key] = 0
            else:
                self.data['OTHER_MEDS'][key] = 1

        # CUR_ILL 처리
        # 있으면 1 없으면 0
        for key, value in self.data['CUR_ILL'].iteritems():
            if (value == 'None'):
                self.data['CUR_ILL'][key] = 0
            else:
                self.data['CUR_ILL'][key] = 1
        # SYMPTOM1 categories
        self.data['SYMPTOM1'].astype('category').cat.categories

        self.data['SYMPTOM1']

        pd.set_option('display.max_rows', None)
        counts = self.data['SYMPTOM1'].value_counts()

        pd.DataFrame(counts, columns=['symptom', 'case'])

        for key, value in self.data['SYMPTOM1'].iteritems():
            if counts[value] < 100:
                self.data.drop(key, axis=0, inplace=True)

        ########################
        ######### SEX ##########
        ########################

        # Fill null value using method = 'ffill'
        self.data['SEX'].replace('U', np.nan, inplace=True)
        self.data['SEX'].fillna(method='ffill', inplace=True)

        ########################
        ####### RECOVD #########
        ########################

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

    def split(self):
        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=42)

    def split(self, testSize, random):
        self.train, self.test = train_test_split(self.data, test_size=testSize, random_state=random)

    def scaling(self):
        train_num = self.train[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
        train_cat = self.train.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis=1)
        test_num = self.test[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
        test_cat = self.test.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis=1)

        scaler = MinMaxScaler()
        encoder = OrdinalEncoder()
        # 인코딩
        train_cat_encoded = encoder.fit_transform(train_cat)
        test_cat_encoded = encoder.fit_transform(test_cat)
        # 데이터 프레임으로 변환
        train_cat_encoded = pd.DataFrame(train_cat_encoded, columns=list(train_cat))
        test_cat_encoded = pd.DataFrame(test_cat_encoded, columns=list(test_cat))
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
        self.train_X = train_prepared.drop('LEVEL', axis=1)
        self.train_y = train_prepared['LEVEL'].copy()
        self.test_X = test_prepared.drop('LEVEL', axis=1)
        self.test_y = test_prepared['LEVEL'].copy()

    def scaling(self, s, e):
        train_num = self.train[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
        train_cat = self.train.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis=1)
        test_num = self.test[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
        test_cat = self.test.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis=1)
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
        self.train_X = train_prepared.drop('LEVEL', axis=1)
        self.train_y = train_prepared['LEVEL'].copy()
        self.test_X = test_prepared.drop('LEVEL', axis=1)
        self.test_y = test_prepared['LEVEL'].copy()

    def fit_transform(self):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        self.bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=300, max_samples=100,
                                         bootstrap=True,
                                         max_features=9, n_jobs=-1, oob_score=True)
        self.bag_clf.fit(self.train_X, self.train_y)

    def predict(self, num):
        some_data = self.test_X.iloc[:num]
        some_labels = self.test_y.iloc[:num]
        print('실제: ', some_labels.values)
        print('예측: ', self.bag_clf.predict(some_data))

    def score(self):
        from sklearn.metrics import accuracy_score
        y_pred = self.bag_clf.predict(self.test_X)
        print('accuarcy score: ', accuracy_score(self.test_y, y_pred))


####################예시 실행 코드############################
data=VAERS() # 클래스 선언
data.preprocess() # 전처리 함수 사용 - 이 때 csv 파일 3개가 디렉토리에 있다고 가정함
data.split(0.2, 42)
data.scaling('MinMaxScaler', 'OrdinalEncoder')
data.fit_transform() # BaggingClassifier 훈련
data.predict(10) # 10개 예측 출력
data.score() # 정확도 출력