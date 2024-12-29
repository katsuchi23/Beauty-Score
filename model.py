import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BeautyScore(nn.Module):
    def __init__(self, first_neuron):
        super(BeautyScore, self).__init__()

        self.first_out_channels = first_neuron
        
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=3, out_channels=self.first_out_channels, kernel_size=3, padding=1),  # dimension [batch_size, out_channel, 128, `128`] -> padding = 1
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels),
            nn.MaxPool2d(2), # dimension [batch_size, out_channel, 64, 64]
            
            # Second Convolutional Block
            nn.Conv2d(in_channels=self.first_out_channels, out_channels=self.first_out_channels*2, kernel_size=3, padding=1), # dimension [batch_size, out_channel, 32, 32]
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels*2),
            nn.MaxPool2d(2), # dimension [batch_size, out_channel*2, 32, 32]
            
            # Third Convolutional Block
            nn.Conv2d(in_channels=self.first_out_channels*2, out_channels=self.first_out_channels*4, kernel_size=3, padding=1), # dimension [batch_size, out_channel, 16, 16]
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels*4),
            nn.MaxPool2d(2), # dimension [batch_size, out_channel*4, 16, 16]
            
        )
        
        # Calculate size of flattened features after the convolutional layers
        self.flatten_size = self.first_out_channels * 4 * (128 // (2**3)) * (128 // (2**3)) # out_channel * (128 // 2^amount_of_max_pool) * (128 // 2^amount_of_max_pool)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.flatten_size, 256), # dimension [batch_size, 256]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), # dimension [batch_size, 128]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1), # dimension [batch_size, 1]
            nn.Sigmoid() # To get value from 0 to 1
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
    
class Trainer:
    def __init__(self, train_loader = None, val_loader = None):
        self.model = BeautyScore(first_neuron=256)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        self.num_epochs = 20

    def load_data(self):
        data_path = '/home/reynaldy/.cache/kagglehub/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores/versions/2/scut_fbp5500-cmprsd.npz'

        data = np.load(data_path)
        data['X'].shape, data['y'].shape

        features_numpy = data['X'].astype(np.float32) 
        features_numpy = np.array([cv2.resize(img, (128, 128)) for img in features_numpy]) # Resize the images to 256x256
        features = torch.tensor(features_numpy, dtype=torch.float32).to(device)
        features = features.permute(0, 3, 1, 2).to(device)

        label_numpy = data['y'].astype(np.float32)
        labels = torch.tensor(label_numpy, dtype=torch.float32).to(device)
        tensor_min = labels.min()
        tensor_max = labels.max()

        labels = (labels - tensor_min) / (tensor_max - tensor_min)
        print("Finish loading data")

        train_size = int(0.8 * len(features))
        test_size = len(features) - train_size

        train_dataset, test_dataset = random_split(TensorDataset(features, labels), [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, val_loader
        
    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        running_loss = 0.0
        train_loader, _= self.load_data()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} Loss: {loss.item()}")
        
        epoch_loss = running_loss / len(train_loader)
        if self.scheduler:
            self.scheduler.step(epoch_loss)

        print(f"Training Loss: {epoch_loss:.4f}")
        return epoch_loss
    
    def validate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        running_loss = 0.0
        _, val_loader = self.load_data()
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                
                running_loss += loss.item()
                
        epoch_loss = running_loss / len(val_loader)
        print(f"Validation Loss: {epoch_loss:.4f}")
        return epoch_loss
    
    def image_to_tensor(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image = torch.tensor(image, dtype=torch.float32).to(device)
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image
    
    def predict(self, inputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load('best_model.pth', weights_only=True))
        self.model.to(device)
        self.model.eval()
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

if __name__ == "__main__":
    trainer = Trainer()
    
    # Test the model
    image_path = '6082308423334085331.jpg'
    image_tensor = trainer.image_to_tensor(image_path)
    prediction = trainer.predict(image_tensor)
    print(f"Predicted Beauty Score: {prediction.item() * 100}")