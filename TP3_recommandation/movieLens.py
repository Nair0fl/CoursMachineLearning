from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt


data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.15)

# Utilisation algorithme svd
algo = SVD()
algo.fit(trainset)

# donnée pour la prédiction
uid = "154"  # id de l'user
iid = "302"  # id du movielens

# gfait une prédiction
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

# execute sur les données de test
test_pred = algo.test(testset)
x = [elem[2] -elem[3] for elem in test_pred]

plt.hist(x, 45, facecolor='g', alpha=1)

# get RMSE
print("User-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)
