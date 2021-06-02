from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


def scaling_encoding(X_train_num, X_train_cat, y_train, X_test_cat, X_test_num, y_test, s1, s2,
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
            globals()['X_train_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_train_cat_encoded{}'.format(i)],
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
            globals()['X_train_cat_encoded{}'.format(i)] = pd.DataFrame(globals()['X_train_cat_encoded{}'.format(i)],
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


scaling_encoding(X_train_num, X_train_cat, y_train, X_test_cat, X_test_num, y_test, 'Standard', 'MinMax',
                 'Robust', 'MaxAbs', 'Ordinal', 'Label')