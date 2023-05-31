from numpy.lib import format
from numpy import load
from os import listdir
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data import TensorDataset, DataLoader
from numpy import load
from os import listdir
import os
from sklearn.model_selection import train_test_split

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

data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_in, data_out, test_size=0.01, random_state=42)

# Convertir les données en tenseurs PyTorch
X_train = torch.tensor(data_X_train, dtype=torch.float32)
y_train = torch.tensor(data_y_train, dtype=torch.float32)
X_test = torch.tensor(data_X_test, dtype=torch.float32)
y_test = torch.tensor(data_y_test, dtype=torch.float32)

# Create train and test datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Define batch size
batch_size = 64

# Create train and test dataloaders
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class PositionPredictor(nn.Module):
    def __init__(self):
        super(PositionPredictor, self).__init__()

        ### 3 layers ###
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(10, 2), stride = 1, padding = 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size =3)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32*2, kernel_size =3, stride =1, padding = 5)
        self.bn2 = nn.BatchNorm2d(32*2)
        self.maxpool2 = nn.MaxPool2d(kernel_size =3)

        self.conv3 = nn.Conv2d(in_channels=32*2, out_channels=288, kernel_size =3, stride =1, padding = 0)
        self.bn3 = nn.BatchNorm2d(288)

        self.fc1 = nn.Linear(in_features = 288, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = 2)

        

    def forward(self, x):
        x = self.conv1(x)      
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x

model = PositionPredictor()
# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entraînement du modèle
num_epochs = 1000

for epoch in range(num_epochs):
    for images, targets in trainloader:

        images = images.unsqueeze(1)

        optimizer.zero_grad()
        
        outputs = model(images)

        loss = criterion(outputs, targets)

        # Backward pass et optimisation

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