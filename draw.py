import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost.sklearn import  XGBRegressor
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
def draw():
    origin = pd.read_csv("~/dataset/housePrice/originData/train.csv", engine='python')

    sns.barplot(x='Electrical', y='SalePrice', data=origin)
    plt.show()
    plt.savefig("all_attribute.png",dpi=500)
if __name__ == "__main__":
    draw()