import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 10, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
        x = torch.cat([z, label_onehot], dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)


@st.cache_resource
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


def generate_images(model, digit, count=5):
    z = torch.randn(count, 100)
    labels = torch.full((count,), digit, dtype=torch.long)
    with torch.no_grad():
        images = model(z, labels).cpu()
    return images


def show_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 2))
    for i, img in enumerate(images):
        axs[i].imshow(img.squeeze(0), cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)


st.title("🧠 Handwritten Digit Generator (0–9)")
st.write("Select a digit and generate 5 images using a trained model.")

selected_digit = st.selectbox("Choose a digit (0–9)", list(range(10)))
if st.button("Generate Images"):
    model = load_generator()
    images = generate_images(model, selected_digit)
    show_images(images)
