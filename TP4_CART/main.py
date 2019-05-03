import timeit
from sklearn import datasets
from randomforest import randomforest
from ovo import OvOPredictor
from ovr import OvRPredictor
from svm import SVMPredictor


def main():
    digits = datasets.load_digits()
    data = digits['data']
    target  = digits['target']
    predictorRF = randomforest(data, target)
    predictorOVO = OvOPredictor(data, target)
    predictorOVR = OvRPredictor(data, target)
    predictorSVM = SVMPredictor(data, target)

    time = timeit.timeit(predictorSVM.run, number=1)
    stats = predictorSVM.run()
    print(f' Classifier SVM \n ------------------\n')
    print(f' Temps {time:.2f} seconds\n') #En général le classifieur OVR met 17s pour traiter le data set
    print(stats) # 97% de bonne réponses
    
    time = timeit.timeit(predictorRF.run, number=1)
    stats = predictorRF.run()
    print(f' Classifier randomforest \n ------------------\n')
    print(f' Temps {time:.2f} seconds\n') #En général le classifieur OVR met 17s pour traiter le data set
    print(stats) # 97% de bonne réponses
    
    time = timeit.timeit(predictorOVR.run, number=1)
    stats = predictorOVR.run()
    print(f' Classifier ovr \n ------------------\n')
    print(f' Temps {time:.2f} seconds\n')  #En général le classifieur OVR met 5.8s pour traiter le data set
    print(stats) #sur tout les données 95% de bonne réponses
    
    time = timeit.timeit(predictorOVO.run, number=1)
    stats = predictorOVO.run()
    print(f' Classifier ovo \n ------------------\n')
    print(f' Temps {time:.2f} seconds\n')#En général le classifieur OVR met 12s pour traiter le data set
    print(stats)#98% de bonne réponses

    

if __name__ == '__main__':
    main()