# Import Libraries
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from models import model
from dataloader import Dataloader
from noise_scheduling import T
from utils import get_loss, sample_plot_image, show_images

import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 64
BATCH_SIZE = 128

data = Dataloader.load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Training Function
def train(epochs, optimizer, device):
    total_start = time.time()
    print("Start Trainig............")
    for epoch in range(epochs):
        epoch_start = time.time()
        for step, batch in enumerate(dataloader):
    #         start = time.time()
            optimizer.zero_grad()
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch[0], t)
            loss.backward()
            optimizer.step()
    #         print(f"Train time: {(time.time()-start)/60} secs")
        if epoch % 1 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            sample_plot_image()
        print(f"Epoch time: {(time.time()-epoch_start)/60} secs")
    print(f"Total tarining time: {(time.time()-total_start)/60} secs")

def main():
    show_images(data)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 100 
    train(epochs, optimizer, device)

if __name__ == '__main__':
    main()