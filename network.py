from collections import Counter
from loguru import logger
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def DeepLearningClassifier(X_train, y_train, epochs=10, batch_size=64, learning_rate=0.001):
    # 1. Define neural network
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    # 2. Make model and define other variale (loss and optim)
    input_size = X_train.shape[1]
    classifier_model = SimpleClassifier(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate)

    # 3. Train!
    total_loss = 0.0 
    for epoch in range(epochs): 
        for i in range(0, len(X_train), batch_size): # iter over each batch
            optimizer.zero_grad() # zero the grad
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            y_pred = classifier_model(batch_X) # forward
            loss = criterion(y_pred, batch_y.view(-1, 1)) # calculate loss
            loss.backward() # backwards
            optimizer.step() # update weights
            total_loss += loss.item()  # total lsos
        # Logging for the epoch
        avg_loss = total_loss / (len(X_train) / batch_size)  # Calculate average loss for the epoch
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    # 4. Return trained model
    return classifier_model

   
    
    