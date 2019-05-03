from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from timeit import default_timer as timer

class OvRPredictor:
    def __init__(self,data, target, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target, test_size=test_size)
        self.test_values = [(self.x_test[index],value) for index, value in enumerate(self.y_test)]
    #Creation du random classifiere
    def create_ovr(self):
        lr = LogisticRegression(solver='lbfgs', max_iter=400, multi_class='auto')
        return lr.fit(self.x_train, self.y_train)
    #Prediction avec le calcul de la precision
    def predict(self,lr):
        stats={'Bon':0,'PasBon':0}
        for elem in self.test_values:
            predicted = lr.predict([elem[0]])
            if predicted[0] == elem[1]:
                stats['Bon'] +=1                
            else:
                stats['PasBon'] +=1                
        return (stats['Bon']/len(self.test_values)*100)

    
    def run(self):
        print("OVR")
        timer_start_fit = timer()
        lr=self.create_ovr()
        timer_end_fit = timer()
        OneVsRestClassifier(lr)
        timer_start_predict = timer()
        stat=self.predict(lr)
        timer_end_predict = timer()
        print("Temps entrainement : "+str(round(timer_end_fit - timer_start_fit, 6)))
        print("Temps prediction : "+str(round(timer_end_predict - timer_start_predict, 6)))
        return stat
