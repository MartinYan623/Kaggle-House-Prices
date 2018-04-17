import pandas as pd
import numpy as np

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
#combined['GrLivArea'] = np.log(combined['GrLivArea'])


print ("Unique values are:", combined.Neighborhood.unique())
print (train.Neighborhood.value_counts(), "\n")


categoricals = train.select_dtypes(exclude=[np.number])
print(categoricals.describe())

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

combined['MSZoning'].fillna('RL',inplace=True)
MSZoning_dummies = pd.get_dummies(combined['MSZoning'], prefix='MSZoning')
combined = pd.concat([combined,MSZoning_dummies], axis=1)
combined.drop('MSZoning', axis=1, inplace=True)

LotShape_dummies = pd.get_dummies(combined['LotShape'], prefix='LotShape')
combined = pd.concat([combined,LotShape_dummies], axis=1)
combined.drop('LotShape', axis=1, inplace=True)

LandContour_dummies = pd.get_dummies(combined['LandContour'], prefix='LandContour')
combined = pd.concat([combined,LandContour_dummies], axis=1)
combined.drop('LandContour', axis=1, inplace=True)

combined['Utilities'].fillna('AllPub',inplace=True)
combined['enc_Utilities'] = pd.get_dummies(combined.Utilities, drop_first=True)


LotConfig_dummies = pd.get_dummies(combined['LotConfig'], prefix='LotConfig')
combined = pd.concat([combined,LotConfig_dummies], axis=1)
combined.drop('LotConfig', axis=1, inplace=True)

LandSlope_dummies = pd.get_dummies(combined['LandSlope'], prefix='LandSlope')
combined = pd.concat([combined,LandSlope_dummies], axis=1)
combined.drop('LandSlope', axis=1, inplace=True)
print(LandSlope_dummies)

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