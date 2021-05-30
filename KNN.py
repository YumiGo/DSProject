# data : covid19.py 에서 data cleaning 완료한 데이터
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