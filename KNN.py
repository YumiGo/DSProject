# data : covid19.py 에서 data cleaning 완료한 데이터
############################### Spilt ###############################
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=42)
############################### Scaling and Encoding ###############################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

train_num = train[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
train_cat = train.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis = 1)

test_num = test[['AGE_YRS', 'NUMDAYS', 'ALL_COUNT']]
test_cat = test.drop(['AGE_YRS', 'NUMDAYS', 'ALL_COUNT'], axis = 1)


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
        train_cat_encoded_6 = encoder.fit_transform(train_cat['SYMPTOM1'])
        # train_cat 데이터 프레임으로 변환
        train_cat_encoded_0 = pd.DataFrame(train_cat_encoded_0, columns=['STATE'])
        train_cat_encoded_1 = pd.DataFrame(train_cat_encoded_1, columns=['SEX'])
        train_cat_encoded_2 = pd.DataFrame(train_cat_encoded_2, columns=['RECOVD'])
        train_cat_encoded_3 = pd.DataFrame(train_cat_encoded_3, columns=['OTHER_MEDS'])
        train_cat_encoded_4 = pd.DataFrame(train_cat_encoded_4, columns=['CUR_ILL'])
        train_cat_encoded_5 = pd.DataFrame(train_cat_encoded_5, columns=['VAX_MANU'])
        train_cat_encoded_6 = pd.DataFrame(train_cat_encoded_6, columns=['SYMPTOM1'])
        # train_cat 데이터프레임 합치기
        train_cat_encoded = pd.concat([train_cat_encoded_0, train_cat_encoded_1,
                                       train_cat_encoded_2, train_cat_encoded_3,
                                       train_cat_encoded_4, train_cat_encoded_5,
                                       train_cat_encoded_6], axis=1)
        # test_cat인코딩
        test_cat_encoded_0 = encoder.fit_transform(test_cat['STATE'])
        test_cat_encoded_1 = encoder.fit_transform(test_cat['SEX'])
        test_cat_encoded_2 = encoder.fit_transform(test_cat['RECOVD'])
        test_cat_encoded_3 = encoder.fit_transform(test_cat['OTHER_MEDS'])
        test_cat_encoded_4 = encoder.fit_transform(test_cat['CUR_ILL'])
        test_cat_encoded_5 = encoder.fit_transform(test_cat['VAX_MANU'])
        test_cat_encoded_6 = encoder.fit_transform(test_cat['SYMPTOM1'])
        # test_cat 데이터 프레임으로 변환
        test_cat_encoded_0 = pd.DataFrame(test_cat_encoded_0, columns=['STATE'])
        test_cat_encoded_1 = pd.DataFrame(test_cat_encoded_1, columns=['SEX'])
        test_cat_encoded_2 = pd.DataFrame(test_cat_encoded_2, columns=['RECOVD'])
        test_cat_encoded_3 = pd.DataFrame(test_cat_encoded_3, columns=['OTHER_MEDS'])
        test_cat_encoded_4 = pd.DataFrame(test_cat_encoded_4, columns=['CUR_ILL'])
        test_cat_encoded_5 = pd.DataFrame(test_cat_encoded_5, columns=['VAX_MANU'])
        test_cat_encoded_6 = pd.DataFrame(test_cat_encoded_6, columns=['SYMPTOM1'])
        # test_cat 데이터프레임 합치기
        test_cat_encoded = pd.concat([test_cat_encoded_0, test_cat_encoded_1,
                                      test_cat_encoded_2, test_cat_encoded_3,
                                      test_cat_encoded_4, test_cat_encoded_5,
                                      test_cat_encoded_6], axis=1)
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
    train_X = train_prepared.drop('SYMPTOM1', axis=1)
    train_y = train_prepared['SYMPTOM1'].copy()
    test_X = test_prepared.drop('SYMPTOM1', axis=1)
    test_y = test_prepared['SYMPTOM1'].copy()
    # K-Nearest Neighbors Classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_X, train_y)
    # 점수 출력
    print(s + ',' + e + ':', knn.score(test_X, test_y))


scaling('StandardScaler', 'OrdinalEncoder')
scaling('MinMaxScaler', 'OrdinalEncoder')
scaling('RobustScaler', 'OrdinalEncoder')
scaling('MaxAbsScaler', 'OrdinalEncoder')
scaling('StandardScaler', 'LabelEncoder')
scaling('MinMaxScaler', 'LabelEncoder')
scaling('RobustScaler', 'LabelEncoder')
scaling('MaxAbsScaler', 'LabelEncoder')