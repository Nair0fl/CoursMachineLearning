from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_20newsgroups_vectorized
import timeit

class RandomForest:
    
    def run(self):
        time_create=timeit.timeit(self.create_forest, number=10)
        self.create_forest()     
        time_prediction=timeit.timeit(self.predire, number=10)        
        a=self.predire()
        print(f' Temps {time_create:.2f} seconds pour la création de l arbre\n')
        print(f' Temps {time_prediction:.2f} seconds pour la prédiction\n')
        print("Le pourcentage de bonne réponse est de :"+ str(a) )

    def __init__(self):
      n_samples = 10000
      dataset = fetch_20newsgroups_vectorized('all')
      X = dataset.data
      y = dataset.target
      X = X[:n_samples]
      y = y[:n_samples]
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                        random_state=42,
                                                        stratify=y,
                                                        test_size=0.1)
     
      
    #Creation du random classifiere
    def create_forest(self):
        self.forest = RandomForestClassifier(n_estimators=100, random_state=21)
        self.forest.fit(self.X_train,self.y_train)
        
    #Prediction avec le calcul de la precision
    def predire(self):
        prediction = self.forest.predict(self.X_test)
        stats={'Bon':0,'PasBon':0}          
        i = 0
        for elem in self.X_test :
            if prediction[i] == self.y_test[i]:
                stats['Bon'] +=1
            else:
                stats['PasBon'] +=1
            i += 1
        return (stats['Bon']/i*100)
