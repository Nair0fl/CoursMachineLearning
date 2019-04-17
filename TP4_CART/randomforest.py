from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class randomforest:
    def __init__(self,data, target, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target, test_size=test_size)
        self.test_values = [(self.x_test[index],value) for index, value in enumerate(self.y_test)]
    #Creation du random classifiere
    def create_forest(self):
        forest = RandomForestClassifier(n_estimators=100, random_state=21)
        return forest.fit(self.x_train, self.y_train)
    #Prediction avec le calcul de la precision
    def predict(self,forest):
        stats={'Bon':0,'PasBon':0}
        for elem in self.test_values:
            predicted = forest.predict([elem[0]])
            if predicted[0] == elem[1]:
                stats['Bon'] +=1                
            else:
                stats['PasBon'] +=1                
        return (stats['Bon']/len(self.test_values)*100)

    
    def run(self):
        forest=self.create_forest()
        return self.predict(forest)