import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor,AdaBoostClassifier

hp_train = pd.read_csv('data/train.csv')
hp_test = pd.read_csv("data/test.csv")

hp_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
hp_train = hp_train.drop(hp_train[hp_train['Id'] == 1299].index)
hp_train = hp_train.drop(hp_train[hp_train['Id'] == 524].index)
# 正态化SalePrice
target = np.log(hp_train['SalePrice'])
train = hp_train.drop(['SalePrice'], axis=1)
# 合并测试集和训练集
combined = train.append(hp_test)
combined.reset_index(inplace=True)
combined.drop(['index','Id'], inplace=True, axis=1)

# 显示数据
""""
print (hp_train.columns)
print (hp_train['SalePrice'].describe() )
sns.distplot(hp_train['SalePrice'])
plt.show(sns.distplot(hp_train['SalePrice']))
# 对于houseprice和GrLivArea,发现两个异常点 
var = 'GrLivArea'
data = pd.concat([hp_train['SalePrice'], hp_train[var]], axis=1)
plt.show(data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000)))
"""

# 正态化GrLivArea
#transformed histogram and normal probability plot
combined['GrLivArea'] = np.log(combined['GrLivArea'])


print ("Unique values are:", combined.BsmtHalfBath.unique())
print (train.BsmtHalfBath.value_counts(), "\n")

# 将是否有basement 分为两类
def encode(x): return 0 if x == 0 else 1
combined['enc_TotalBsmtSF'] = combined.TotalBsmtSF.apply(encode)
def encode(x): return 0 if x == 0 else 1
combined['enc_OpenPorchSF'] = combined.OpenPorchSF.apply(encode)

month_mapping = {
    1:'one',
    2:'one',
    3:'one',
    4:'two',
    5:'two',
    6: 'two',
    7: 'three',
    8: 'three',
    9:'three',
    10:'four',
    11:'four',
    12:'four',
}
combined['MoSold'] = combined.MoSold.map(month_mapping)

MoSold_dummies = pd.get_dummies(combined['MoSold'], prefix='MoSold')
combined = pd.concat([combined,MoSold_dummies], axis=1)
combined.drop('MoSold', axis=1, inplace=True)


# 将enc_condition减少为二值属性
def encode(x): return 1 if x == 'Partial' else 0
combined['enc_condition'] = combined.SaleCondition.apply(encode)

combined['GarageCars'].fillna(combined.iloc[:1458].GarageCars.median(),inplace=True)
combined['GarageArea'].fillna(combined.iloc[:1458].GarageArea.median(),inplace=True)
combined['LotFrontage'].fillna(combined.iloc[:1458].LotFrontage.median(),inplace=True)
combined['MasVnrArea'].fillna(combined.iloc[:1458].MasVnrArea.median(),inplace=True)
combined['TotalBsmtSF'].fillna(combined.iloc[:1458].TotalBsmtSF.median(),inplace=True)
# one-hot encoding
combined['enc_Street'] = pd.get_dummies(combined.Street, drop_first=True)

combined['GarageCond'].fillna('TA',inplace=True)
GarageCond_dummies = pd.get_dummies(combined['GarageCond'], prefix='GarageCond')
combined = pd.concat([combined,GarageCond_dummies], axis=1)
combined.drop('GarageCond', axis=1, inplace=True)

combined['GarageQual'].fillna('TA',inplace=True)
# one-hot 多值编码
GarageQual_dummies = pd.get_dummies(combined['GarageQual'], prefix='GarageQual')
combined = pd.concat([combined,GarageQual_dummies], axis=1)
combined.drop('GarageQual', axis=1, inplace=True)

combined['GarageYrBlt'].fillna(combined.iloc[:1458].GarageYrBlt.median(),inplace=True)

combined['GarageFinish'].fillna('Unf',inplace=True)
GarageFinish_dummies = pd.get_dummies(combined['GarageFinish'], prefix='GarageFinish')
combined = pd.concat([combined,GarageFinish_dummies], axis=1)
combined.drop('GarageFinish', axis=1, inplace=True)

combined['GarageType'].fillna('Attchd',inplace=True)
GarageType_dummies = pd.get_dummies(combined['GarageType'], prefix='GarageType')
combined = pd.concat([combined,GarageType_dummies], axis=1)
combined.drop('GarageType', axis=1, inplace=True)


combined['BsmtCond'].fillna('TA',inplace=True)
BsmtCond_dummies = pd.get_dummies(combined['BsmtCond'], prefix='BsmtCond')
combined = pd.concat([combined,BsmtCond_dummies], axis=1)
combined.drop('BsmtCond', axis=1, inplace=True)

combined['BsmtFullBath'].fillna(combined.iloc[:1458].BsmtFullBath.median(),inplace=True)

combined['BsmtUnfSF'].fillna(combined.iloc[:1458].BsmtUnfSF.median(),inplace=True)
combined['BsmtFinSF2'].fillna(combined.iloc[:1458].BsmtFinSF2.median(),inplace=True)

combined['BsmtHalfBath'].fillna(combined.iloc[:1458].BsmtHalfBath.median(),inplace=True)

#nulls = pd.DataFrame(combined.isnull().sum().sort_values(ascending=False)[:25])
#nulls.columns = ['Null Count']
#nulls.index.name = 'Feature'
#print(nulls)
# 删掉空缺值太多的5个属性
combined.drop(['PoolQC'], 1, inplace=True)
combined.drop(['MiscFeature'], 1, inplace=True)
combined.drop(['Alley'], 1, inplace=True)
combined.drop(['Fence'], 1, inplace=True)
combined.drop(['FireplaceQu'], 1, inplace=True)
"""
numeric_features = combined.select_dtypes(include=[np.number])
print(numeric_features.dtypes)
"""
train = combined.iloc[:1458]
test = combined.iloc[1458:]

predictors=['OverallQual','GrLivArea','TotRmsAbvGrd','enc_TotalBsmtSF','GarageCars','1stFlrSF','GarageArea','FullBath',
            'YearBuilt','LotFrontage','GarageCond_Ex','GarageCond_Fa','GarageCond_Gd','GarageCond_Po','GarageCond_TA',
            'GarageYrBlt','GarageQual_Ex','GarageQual_Fa','GarageQual_Gd','GarageQual_Po','GarageQual_TA','GarageFinish_Fin',
            'GarageFinish_RFn','GarageFinish_Unf','YearRemodAdd','MasVnrArea','Fireplaces','TotalBsmtSF','WoodDeckSF','enc_OpenPorchSF',
            '2ndFlrSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold_four','MoSold_one',
            'MoSold_three', 'MoSold_two','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','LowQualFinSF','YrSold','OverallCond','MSSubClass',
            'EnclosedPorch','KitchenAbvGr']
"""
,'GarageType_2Types','GarageType_Attchd','GarageType_Basment','GarageType_BuiltIn',
            'GarageType_CarPort','GarageType_Detchd','BsmtCond_Fa','BsmtCond_Gd','BsmtCond_Po','BsmtCond_TA'
"""
numeric_features = hp_train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:38], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])

x_train, x_test, y_train, y_test = train_test_split(train[predictors], target, random_state=42, test_size=.2)



"""
for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(x_train, y_train)
    preds_ridge = ridge_model.predict(x_test)

    plt.scatter(preds_ridge, y_test, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(x_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()
"""


#lr = linear_model.LinearRegression()
#lr=GradientBoostingRegressor()
lr=linear_model.RidgeCV()
algorithms=[
linear_model.LinearRegression(),
linear_model.RidgeCV(alphas=np.logspace(-3, 2, 100)),
]


full_predictions = []
for lr in algorithms:
    model = lr.fit(x_train, y_train)
    predictions=model.predict(x_test)
    full_predictions.append(predictions)
predictions = (full_predictions[0]*0.8 + full_predictions[1]*0.2)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
"""
full_predictions = []
for lr in algorithms:
    model = lr.fit(train[predictors], target)
    predictions=model.predict(test[predictors])
    full_predictions.append(predictions)
predictions = (full_predictions[0]*0.5 + full_predictions[1]*0.5)
predictions = np.exp(predictions)
submission = pd.DataFrame({
        "Id": hp_test["Id"],
        "SalePrice": predictions
    })
print(submission)
submission.to_csv('/Users/martin_yan/Desktop/submission4.csv', index=False)
"""