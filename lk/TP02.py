import numpy as np  #--------Charger les bibliothèques python nécessaires--------#
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

file = pd.read_csv('datasets.csv') #--------Lire la base de données--------#

#file.head(5) #--------Pour lire les 5 premiers lignes--------#

#file.shape #--------Pour savoire nombres de lignes et de colonnes--------#

X = file.drop('Outcome',axis=1).values #pour supprimer les valeurs de la clasee
y = file['Outcome'].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y) #deviser partie train et partie apprentissage

from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9) 
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='La précision de partie Test :')
plt.plot(neighbors, train_accuracy, label='La précision de partie Train :')
plt.legend()
plt.xlabel('Nombre de voisins :')
plt.ylabel('La précision :')
plt.show()

