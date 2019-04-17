import timeit
from sklearn import datasets
from randomforest import randomforest
from ovo import OvOPredictor
from ovr import OvRPredictor


def main():
    digits = datasets.load_digits()
    data = digits['data']
    target  = digits['target']
    predictorRF = randomforest(data, target)
    predictorOVO = OvOPredictor(data, target)
    predictorOVR = OvRPredictor(data, target)

    time = timeit.timeit(predictorRF.run, number=10)
    stats = predictorRF.run()
    print(f' Classifier randomforest \n ------------------\n')
    print(f' Temps {time:.2f} seconds\n') #En général le classifieur OVR met 17s pour traiter le data set
    print(stats) # 97% de bonne réponses
    
    time = timeit.timeit(predictorOVO.run_predict, number=10)
    stats = predictorOVO.run_predict()
    print(f' Classifier ovo \n ------------------\n')
    print(f' Temps {time:.2f} seconds\n')#En général le classifieur OVR met 12s pour traiter le data set
    print(stats)#98% de bonne réponses
    time = timeit.timeit(predictorOVR.run_predict, number=10)
    stats = predictorOVR.run_predict()
    print(f' Classifier ovr \n ------------------\n')
    print(f' Temps {time:.2f} seconds\n')  #En général le classifieur OVR met 5.8s pour traiter le data set
    print(stats) #sur tout les données 95% de bonne réponses
    

if __name__ == '__main__':
    main()