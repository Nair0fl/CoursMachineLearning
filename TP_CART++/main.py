from sklearn import datasets
from logistic_regression import Logistic_Regression
from SVM import SVM
from random_forest import RandomForest


def main():
    predictorRF = RandomForest()
    predicatorSVM=SVM()
    predicatorLogistic_Regression=Logistic_Regression()
    print('========Random Forest============')
    predictorRF.run()
    print('========SVM============')
    predicatorSVM.run()
    print('========Logistic Regression============')
    predicatorLogistic_Regression.run()

    

if __name__ == '__main__':
    main()