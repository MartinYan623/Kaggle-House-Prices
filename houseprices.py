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
import sys
sys.path.append("preprocessdata")
import preprocessdata as pre

"""
numeric_features = combined.select_dtypes(include=[np.number])
print(numeric_features.dtypes)
"""
train = pre.combined.iloc[:1458]
test = pre.combined.iloc[1458:]

predictors=['OverallQual','GrLivArea','TotRmsAbvGrd','enc_TotalBsmtSF','GarageCars','1stFlrSF','GarageArea','FullBath',
            'YearBuilt','LotFrontage','GarageCond_Ex','GarageCond_Fa','GarageCond_Gd','GarageCond_Po','GarageCond_TA',
            'GarageYrBlt','GarageQual_Ex','GarageQual_Fa','GarageQual_Gd','GarageQual_Po','GarageQual_TA','GarageFinish_Fin',
            'GarageFinish_RFn','GarageFinish_Unf','YearRemodAdd','MasVnrArea','Fireplaces','TotalBsmtSF','WoodDeckSF','enc_OpenPorchSF',
            '2ndFlrSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold_four','MoSold_one',
            'MoSold_three', 'MoSold_two','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','LowQualFinSF','YrSold','OverallCond','MSSubClass',
            'EnclosedPorch','KitchenAbvGr','MSZoning_C (all)','MSZoning_FV','MSZoning_RH','MSZoning_RL','MSZoning_RM','enc_Street',
            'LotShape_IR1','LotShape_IR2','LotShape_IR3','LotShape_Reg','LandContour_Bnk','LandContour_HLS','LandContour_Low','LandContour_Lvl',
            'enc_Utilities','LotConfig_Corner','LotConfig_CulDSac','LotConfig_FR2','LotConfig_FR3','LotConfig_Inside','LandSlope_Gtl','LandSlope_Mod','LandSlope_Sev']

#numeric_features = hp_train.select_dtypes(include=[np.number])
#corr = numeric_features.corr()
#print (corr['SalePrice'].sort_values(ascending=False)[:38], '\n')
#print (corr['SalePrice'].sort_values(ascending=False)[-5:])

x_train, x_test, y_train, y_test = train_test_split(train[predictors], pre.target, random_state=42, test_size=.2)

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

full_predictions = []
for lr in algorithms:
    model = lr.fit(train[predictors], pre.target)
    predictions=model.predict(test[predictors])
    full_predictions.append(predictions)
predictions = (full_predictions[0]*0.8 + full_predictions[1]*0.2)
predictions = np.exp(predictions)
submission = pd.DataFrame({
        "Id": pre.hp_test["Id"],
        "SalePrice": predictions
    })
#print(submission)
#submission.to_csv('/Users/martin_yan/Desktop/submission5.csv', index=False)
