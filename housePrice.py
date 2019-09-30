import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from mlxtend.regressor import StackingRegressor
from time import time

import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
##column: MSSubClass MSZoning   LotArea Street LandContour Utilities HouseStyle OverallQual OverallCond YearBuilt YearRemodAdd
# ExterQual ExterCond BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2 BsmtFinSF2 HeatingQC CentralAir KitchenQual
# GarageFinish GarageQual GarageCond

##data fill NA:BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2  GarageFinish GarageQual GarageCond
# data normalize: MSSubClass  LotArea
#
all_str = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities',
           'LotConfig', 'LandSlope', 'Neighborhood',
           'Condition1', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
           'Foundation',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
           'BsmtUnfSF',
           'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
           'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional',
           'Fireplaces', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
           'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
           'MiscVal', 'SaleType', 'SaleCondition']
filter_str = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',
              'Condition1', 'BldgType', 'OverallQual', 'OverallCond',
              'RoofStyle', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation',
              'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
              'BsmtUnfSF',
              'HeatingQC', 'CentralAir',
              'KitchenQual', 'Functional', 'GarageQual',
              'GarageType', 'GarageFinish', 'GarageCars', 'GarageCond',
              'PavedDrive',  'SaleType', 'SaleCondition']
rest_str = ['TotalArea', 'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
            'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
            'KitchenAbvGr','GarageArea',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch','MiscVal', 'ScreenPorch', 'PoolArea']


def data_fill(data):
    data.loc[data['MSZoning'].isnull(), 'MSZoning'] = 'RH'
    data.loc[data['LotFrontage'].isnull(), 'LotFrontage'] = 0

    data.loc[data['MasVnrType'].isnull(), 'MasVnrType'] = 'null'
    data.loc[data['MasVnrArea'].isnull(), 'MasVnrArea'] = 0

    data.loc[data['BsmtQual'].isnull(), 'BsmtQual'] = 0
    data.loc[data['BsmtCond'].isnull(), 'BsmtCond'] = 0
    data.loc[data['BsmtExposure'].isnull(), 'BsmtExposure'] = 0
    data.loc[data['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 0
    data.loc[data['BsmtFinType2'].isnull(), 'BsmtFinType2'] = 0
    data.loc[data['GarageFinish'].isnull(), 'GarageFinish'] = 0
    data.loc[data['GarageQual'].isnull(), 'GarageQual'] = 'NA'
    data.loc[data['GarageCond'].isnull(), 'GarageCond'] = 'NA'
    data.loc[data['GarageCars'].isnull(), 'GarageCars'] = 0
    data.loc[data['GarageArea'].isnull(), 'GarageArea'] = 0
    data.loc[data['BsmtFinSF1'].isnull(), 'BsmtFinSF1'] = 0
    data.loc[data['BsmtFinSF2'].isnull(), 'BsmtFinSF2'] = 0
    data.loc[data['TotalBsmtSF'].isnull(), 'TotalBsmtSF'] = 0
    data.loc[data['KitchenQual'].isnull(), 'KitchenQual'] = 'Gd'
    data.loc[data['GarageType'].isnull(), 'GarageType'] = 'null'
    data.loc[data['GarageFinish'].isnull(), 'GarageFinish'] = 'null'

    data.loc[data['BsmtFullBath'].isnull(), 'BsmtFullBath'] = 0
    data.loc[data['BsmtHalfBath'].isnull(), 'BsmtHalfBath'] = 0
    data.loc[data['Functional'].isnull(), 'Functional'] = 'Typ'
    data.loc[data['BsmtUnfSF'].isnull(), 'BsmtUnfSF'] = 0
    data.loc[data['SaleType'].isnull(), 'SaleType'] = 'Oth'

    data.loc[data['GarageQual'] == 'Ex', 'GarageQual'] = 'Gd'

    data['TotalArea'] = data['LotFrontage'] + data['LotArea'] + data['1stFlrSF'] + data['2ndFlrSF']

    # data.loc[data['GarageQual'] == 'Gd','GarageQual'] = 8
    # data.loc[data['GarageQual'] == 'TA','GarageQual'] = 6
    # data.loc[data['GarageQual'] == 'Fa','GarageQual'] = 4
    # data.loc[data['GarageQual'] == 'Po','GarageQual'] = 2
    # data.loc[data['GarageQual'] == 'NA','GarageQual'] = -1
    # data.loc[data['GarageQual'] == 'null','GarageQual'] = 0
    #
    # data.loc[data['ExterQual'] == 'Ex','ExterQual'] = 10
    # data.loc[data['ExterQual'] == 'Gd','ExterQual'] = 8
    # data.loc[data['ExterQual'] == 'TA','ExterQual'] = 6
    # data.loc[data['ExterQual'] == 'Fa','ExterQual'] = 4
    # data.loc[data['ExterQual'] == 'Po','ExterQual'] = 2
    #
    # data.loc[data['ExterCond'] == 'Ex', 'ExterCond'] = 10
    # data.loc[data['ExterCond'] == 'Gd', 'ExterCond'] = 8
    # data.loc[data['ExterCond'] == 'TA', 'ExterCond'] = 6
    # data.loc[data['ExterCond'] == 'Fa', 'ExterCond'] = 4
    # data.loc[data['ExterCond'] == 'Po', 'ExterCond'] = 2
    #
    # data.loc[data['BsmtQual'] == 'Ex', 'BsmtQual'] = 10
    # data.loc[data['BsmtQual'] == 'Gd', 'BsmtQual'] = 8
    # data.loc[data['BsmtQual'] == 'TA', 'BsmtQual'] = 6
    # data.loc[data['BsmtQual'] == 'Fa', 'BsmtQual'] = 4
    # data.loc[data['BsmtQual'] == 'Po', 'BsmtQual'] = 2
    # data.loc[data['BsmtQual'] == 'null', 'BsmtQual'] = 0
    # data.loc[data['BsmtQual'] == 'NA', 'BsmtQual'] = -1
    #
    # data.loc[data['BsmtCond'] == 'Ex', 'BsmtCond'] = 10
    # data.loc[data['BsmtCond'] == 'Gd', 'BsmtCond'] = 8
    # data.loc[data['BsmtCond'] == 'TA', 'BsmtCond'] = 6
    # data.loc[data['BsmtCond'] == 'Fa', 'BsmtCond'] = 4
    # data.loc[data['BsmtCond'] == 'Po', 'BsmtCond'] = 2
    # data.loc[data['BsmtQual'] == 'null', 'BsmtQual'] = 0
    # data.loc[data['BsmtCond'] == 'NA', 'BsmtCond'] = -1
    #
    # data.loc[data['BsmtExposure'] == 'Gd', 'BsmtExposure'] = 10
    # data.loc[data['BsmtExposure'] == 'Av', 'BsmtExposure'] = 8
    # data.loc[data['BsmtExposure'] == 'Mn', 'BsmtExposure'] = 6
    # data.loc[data['BsmtExposure'] == 'No', 'BsmtExposure'] = 4
    # data.loc[data['BsmtExposure'] == 'null', 'BsmtExposure'] = 0
    # data.loc[data['BsmtExposure'] == 'NA', 'BsmtExposure'] = -2
    #
    # data.loc[data['BsmtFinType1'] == 'GLQ', 'BsmtFinType1'] = 10
    # data.loc[data['BsmtFinType1'] == 'ALQ', 'BsmtFinType1'] = 8
    # data.loc[data['BsmtFinType1'] == 'BLQ', 'BsmtFinType1'] = 6
    # data.loc[data['BsmtFinType1'] == 'Rec', 'BsmtFinType1'] = 4
    # data.loc[data['BsmtFinType1'] == 'LwQ', 'BsmtFinType1'] = 2
    # data.loc[data['BsmtFinType1'] == 'Unf', 'BsmtFinType1'] = 0
    # data.loc[data['BsmtFinType1'] == 'null', 'BsmtFinType1'] = 0
    # data.loc[data['BsmtFinType1'] == 'NA', 'BsmtFinType1'] = -2
    #
    # data.loc[data['BsmtFinType2'] == 'GLQ', 'BsmtFinType2'] = 10
    # data.loc[data['BsmtFinType2'] == 'ALQ', 'BsmtFinType2'] = 8
    # data.loc[data['BsmtFinType2'] == 'BLQ', 'BsmtFinType2'] = 6
    # data.loc[data['BsmtFinType2'] == 'Rec', 'BsmtFinType2'] = 4
    # data.loc[data['BsmtFinType2'] == 'LwQ', 'BsmtFinType2'] = 2
    # data.loc[data['BsmtFinType2'] == 'Unf', 'BsmtFinType2'] = 0
    # data.loc[data['BsmtFinType2'] == 'null', 'BsmtFinType2'] = 0
    # data.loc[data['BsmtFinType2'] == 'NA', 'BsmtFinType2'] = -2
    #
    # data.loc[data['HeatingQC'] == 'Ex','HeatingQC'] = 10
    # data.loc[data['HeatingQC'] == 'Gd', 'HeatingQC'] = 8
    # data.loc[data['HeatingQC'] == 'TA', 'HeatingQC'] = 6
    # data.loc[data['HeatingQC'] == 'Fa', 'HeatingQC'] = 4
    # data.loc[data['HeatingQC'] == 'Po', 'HeatingQC'] = 2

    return data


def data_transform(data):
    return 3


def normal(data, name_str):
    # data[name_str] = (data[name_str] - data[name_str].min()) / (data[name_str].max() - data[name_str].min())
    return


def data_normalize(data, normal_list):
    for str in normal_list:
        normal(data, str)

    return data


def data_dummies(data, isTrain):
    data['MSSubClass'] = data['MSSubClass'].astype('str')
    data_dum = pd.get_dummies(data[filter_str])
    col_name = data_dum.columns.tolist()
    if 'MSSubClass_150' not in data:
        data_dum['MSSubClass_150'] = 0
        col_name.insert(col_name.index('MSSubClass_160'), 'MSSubClass_150')
    data_dum = data_dum.sort_index(axis=1)
    data_final = pd.concat([data_dum, data[rest_str]], axis=1)
    if isTrain:
        data_final = pd.concat([data[['SalePrice']], data_final], axis=1)
    return data_final


def data_split(data, size):
    split_train, split_val = train_test_split(data, test_size=size)
    return split_train, split_val


def data_process(data, isTrain=True):
    data_f = data_fill(data)
    normal_list = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'GarageQual', 'BsmtFinSF1','BsmtUnfSF','BsmtFinSF2','TotalBsmtSF','GrLivArea','1stFlrSF', '2ndFlrSF','GarageArea' ,'WoodDeckSF', 'OpenPorchSF', ]
    data_n = data_normalize(data_f, normal_list)
    data_d = data_dummies(data_n, isTrain)

    train, test = data_split(data_d, 0.1)

    return train, test


def validation_process(data):
    data_f = data_fill(data)
    normal_list = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'GarageQual', 'BsmtFinSF1','BsmtUnfSF','BsmtFinSF2','TotalBsmtSF','GrLivArea','1stFlrSF', '2ndFlrSF','GarageArea' ,'WoodDeckSF', 'OpenPorchSF', ]
    data_n = data_normalize(data_f, normal_list)
    data_d = data_dummies(data_n, False)
    return data_d
def scorer(predction,truth):
    v = ((truth - predction) ** 2).sum()
    u = ((truth - truth.mean()) ** 2).sum()
    return (1- u / v)
def train_model():
    origin = pd.read_csv("~/dataset/housePrice/originData/train.csv", engine='python')

    train, test = data_process(origin)
    knn = KNeighborsRegressor(weights='uniform')
    knn_par = {'n_neighbors':range(5,10)}

    dt = DecisionTreeRegressor(random_state=35)
    dt_par = {"max_depth":range(5,8)}

    rfr = RandomForestRegressor(random_state=35,n_jobs=-1)
    rfr_par = {'max_features':range(20,50),'min_sample_leaf':range(30,60)}

    gbr = GradientBoostingRegressor(random_state=35)
    gbr_par = {'n_estimators':range(95,105),'learning_rate':np.arange(0.1,0.9,0.05)}

    ada = AdaBoostRegressor(random_state=35)
    ada_par = {'n_estimators':range(50,60),'learning_rate':np.arange(0.1,1,0.05)}

    xgb = XGBRegressor(random_state=35)
    xgb_par = {'n_estimators':range(95,105)}

    algs = [ada,xgb]
    params= [ada_par,xgb_par]
    for i in range(len(algs)):
        fit_model(algs[i],train,params[i])
    print('finish model fitting!')
def fit_model(alg,data,parameters):
    x = data.as_matrix()[:,1:]
    y = data.as_matrix()[:,0]
    score = make_scorer(scorer)
    grid = GridSearchCV(alg,parameters,scoring='neg_mean_squared_error',cv=5)
    start = time()
    grid = grid.fit(x,y)
    end = time()
    t = round(end - start,3)
    print(grid.best_params_)
    print('searching time for {} is {} s'.format(alg.__class__.__name__,t))
    return grid
def run_model(train,test,algs):
    new_algs = []
    for alg in algs:
        alg.fit(train.as_matrix()[:, 1:], train.as_matrix()[:, 0])
        new_algs.append(alg)
        print("current {} score is : {}".format(alg.__class__.__name__,alg.score(test.as_matrix()[:,1:],test.as_matrix()[:,0])))
    return new_algs
def alg_model(train, test):
    knn = KNeighborsRegressor(weights='uniform',n_neighbors=5)
    dt = DecisionTreeRegressor(max_depth=6)
    rfr = RandomForestRegressor(n_estimators=400)
    gbr = GradientBoostingRegressor(n_estimators=104)
    ada = AdaBoostRegressor(learning_rate=0.8,n_estimators=57)
    xgb = XGBRegressor(n_estimators=104)
    algs = [knn, dt, rfr, gbr, ada, xgb]
    lr = LinearRegression()
    sclf = StackingRegressor(regressors= [gbr, xgb], meta_regressor=lr)
    algs.append(sclf)
    algs = run_model(train,test,algs)
    return algs

def run(origin, data, algs):
    cols = list(data)
    print("nan is : {} ".format(data.columns[np.where(np.isnan(data))[1]]))
    predict_list = []
    for alg in algs:
        prediction = alg.predict(data.as_matrix()[:,:])
        predict_list.append(prediction)
        print("current alg is : {}".format(alg.__class__.__name__))
        result = pd.DataFrame(
            {'Id': origin['Id'].as_matrix(), 'SalePrice': prediction.astype(np.float32)}
        )
        result.to_csv("~/dataset/housePrice/result/{}.csv".format(alg.__class__.__name__), index=False)
    value = np.zeros(len(predict_list[0]))

    for pre in predict_list:
        value += pre / len(algs)
    boost_result = pd.DataFrame(
            {'Id': origin['Id'].as_matrix(), 'SalePrice':  (value).astype(np.float32)}
        )

    boost_result.to_csv("~/dataset/housePrice/result/gradient_xgb_boost.csv", index=False)

def correspondCoeff(data):
    lasso = Lasso(alpha=0.001)
    lasso.fit(data.as_matrix()[:, 1:], data.as_matrix()[:, 0])
    fitt_lasso = pd.DataFrame({"Feature Importance": lasso.coef_}, index=data.columns[1:])
    print(fitt_lasso.sort_values('Feature Importance', ascending=False))
    fitt_lasso[fitt_lasso['Feature Importance'] != 0].sort_values("Feature Importance").plot(kind='barh',
                                                                                             figsize=(15, 25))
    plt.xticks(rotation=90)
    plt.show()


def train():
    origin = pd.read_csv("~/dataset/housePrice/originData/train.csv", engine='python')

    train, test = data_process(origin)
    # correspondCoeff(train)
    return train, alg_model(train, test)


def validation():
    data_train, algs = train()
    origin = pd.read_csv("~/dataset/housePrice/originData/test.csv", engine='python')
    validation = validation_process(origin)
    list1 = list(data_train)
    list2 = list(validation)
    for i in range(len(list1) - 1):
        if list1[i + 1] != list2[i]:
            print("i is {}; list1 is : {} ; list2 is : {}".format(i, list1[i + 1], list2[i]))
    run(origin, validation, algs)


def draw():
    origin = pd.read_csv("~/dataset/housePrice/originData/train.csv", engine='python')
    column_list = list(origin)
    col = 1
    row = int(len(column_list) / col) if (len(column_list) % col) == 0 else int(len(column_list) / col) + 1
    row = 1

    # fig,ax_list = plt.subplots(row,col,figsize=(50,50))
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # for i in range(row):
    #     for j in range(col):
    #         print("i is {};j is {}".format(i,j))
    #         if(i * col + j >= len(column_list)):
    #             break
    #         sns.barplot(x = column_list[i * col + j],y = 'SalePrice',data=origin,ax=ax_list[i][j])
    sns.barplot(x='Foundation', y='SalePrice', data=origin)
    plt.show()
    plt.savefig("all_attribute.png", dpi=500)


if __name__ == "__main__":
    validation()
