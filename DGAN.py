###################################
#DGAN.py
#Protein-peptide interaction region residues prediction using generative sampling technique and ensemble deep learning-based models.
#shafiee.shima@razi.ac.ir
###################################
import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),  
            nn.ReLU(),
            nn.Linear(128, output_size),  
            nn.Tanh() 
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),  
            nn.ReLU(),
            nn.Linear(128, 1),  
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.model(x)


def call_GAN(input_data):
  
    input_array = np.array([int(c) for c in input_data]).reshape(1, -1)
    input_size = input_array.shape[1]

    
    generator = Generator(input_size, input_size)
    discriminator = Discriminator(input_size)

    
    criterion = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)  
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)  

   
    num_epochs = 100 
    for epoch in range(num_epochs):
       
        discriminator_optimizer.zero_grad()
        real_data = torch.from_numpy(input_array.astype(np.float32))
        real_pred = discriminator(real_data)
        real_loss = criterion(real_pred, torch.ones_like(real_pred))
        
        fake_data = torch.randn(1, input_size)
        fake_pred = discriminator(fake_data)
        fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))

        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

       
        generator_optimizer.zero_grad()
        fake_data = torch.randn(1, input_size)
        fake_pred = discriminator(fake_data)
        generator_loss = criterion(fake_pred, torch.ones_like(fake_pred))
        generator_loss.backward()
        generator_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}")

   
    generator.eval()
    with torch.no_grad():
        balanced_data_tensor = generator(torch.randn(1, input_size))
        balanced_data_array = balanced_data_tensor.detach().numpy()

    
    balanced_data_array = np.where(balanced_data_array > 0, 1, 0)
  










