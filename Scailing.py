############################### Spilt ######################
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size = 0.2, random_state=42)
vaers = train_set.copy()
############################### Scaling and Encoding(한꺼번에) ######################
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
vaers_cat = vaers.drop(['AGE_YRS', 'NUMDAYS'], axis = 1) # vaers_cat: train_set의 categorical data만 있음
num_attribs = ['AGE_YRS', 'NUMDAYS'] # numerical한 feature들
cat_attribs = list(data_cat) # categorical한 feature들
# 1. Standard, One hot
pipeline1 = ColumnTransformer([
    ("num", StandardScaler(), num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
data_prepared = pipeline1.fit_transform(vaers) # 적용
#2. Standard, Label
pipeline2 = ColumnTransformer([
    ("num", StandardScaler(), num_attribs),
    ("cat", LabelEncoder(), cat_attribs),
])
from sklearn.preprocessing import MinMaxScaler
#3. MinMax, One hot
pipeline3 = ColumnTransformer([
    ("num", MinMaxScaler(), num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
#4. MinMax, Label
pipeline4 = ColumnTransformer([
    ("num", MinMaxScaler(), num_attribs),
    ("cat", LabelEncoder(), cat_attribs),
])

from sklearn.preprocessing import RobustScaler
#5. Robust, One hot
pipeline5 = ColumnTransformer([
    ("num", RobustScaler(), num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
#6. Robust, Label
pipeline6 = ColumnTransformer([
    ("num", RobustScaler(), num_attribs),
    ("cat", LabelEncoder(), cat_attribs),
])

from sklearn.preprocessing import MaxAbsScaler
#7. MaxAbs, One hot
pipeline7 = ColumnTransformer([
    ("num", MaxAbsScaler(), num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
#8. MaxAbs, Label
pipeline8 = ColumnTransformer([
    ("num", MaxAbsScaler(), num_attribs),
    ("cat", LabelEncoder(), cat_attribs),
])