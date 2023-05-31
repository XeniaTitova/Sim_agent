from numpy.lib import format
from numpy import load
from os import listdir
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from numpy import load
from os import listdir
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

path =os.getcwd()+"/out/"#scid_1dde730d8ab3ae4a__aid_1__atype_1.npz"
list = listdir(path)
data_in = []
data_out = []
for l in list[1:] :
    data = load(path+l)
    data_in.append(data["my/past/xy"][0])
    data_out.append(data["my/future/xy"][0,0])

data_in = np.array(data_in)
data_out = np.array(data_out)

data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_in, data_out, test_size=0.1, random_state=42)

# Convertir les données en tenseurs PyTorch
X_train = torch.tensor(data_X_train, dtype=torch.float32)
y_train = torch.tensor(data_y_train, dtype=torch.float32)
X_test = torch.tensor(data_X_test, dtype=torch.float32)
y_test = torch.tensor(data_y_test, dtype=torch.float32)


# Définir le modèle
class PositionPredictor(nn.Module):
    def __init__(self):
        super(PositionPredictor, self).__init__()
        self.linear1 = nn.Linear(20, 64)  # 20 entrées (10 coordonnées * 2), 64 unités cachées
        self.linear2 = nn.Linear(64, 128)  # 64 unités cachées, 128 unités cachées
        self.linear3 = nn.Linear(128, 64)  # 128 unités cachées, 64 unités cachées
        self.linear4 = nn.Linear(64, 2)  # 64 unités cachées, 2 sorties (coordonnées xf et yf)

    def forward(self, x):
        x = x.view(-1, 20)  # Aplatir le tenseur d'entrée
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        out = self.linear4(x)
        return out

model = PositionPredictor()

# Définir la fonction de perte et l'optimiseur
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entraînement du modèle
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass et optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Affichage de la progression de l'entraînement
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Tester le modèle
with torch.no_grad():
    model.eval()
    test_input = X_test  # Utilisation du tenseur X_test pour le test
    predicted_output = model(test_input)

print("Résultats du modèle :")
for i in range(len(predicted_output)):
    predicted = predicted_output[i]
    exact = y_test[i]
    print(f"Prédiction : {predicted}\t\tValeur exacte : {exact}")

# Obtention des prédictions pour les coordonnées x et y
predicted_x = predicted_output[:, 0]
predicted_y = predicted_output[:, 1]

# Obtention des valeurs réelles pour les coordonnées x et y
exact_x = y_test[:, 0]
exact_y = y_test[:, 1]

# Création du plot
plt.figure(figsize=(10, 6))
plt.plot(predicted_x, label='Prédiction x')
plt.plot(exact_x, label='Valeur réelle x')
plt.plot(predicted_y, label='Prédiction y')
plt.plot(exact_y, label='Valeur réelle y')

# Ajout des légendes
plt.ylim(-2, 2)
plt.legend()

# Ajout des labels des axes
plt.xlabel('Échantillon')
plt.ylabel('Valeur')

# Affichage du plot
plt.show()