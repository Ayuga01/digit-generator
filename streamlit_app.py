import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cpu')

# ✅ New cDCGAN Generator architecture
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)  # Embedding for digit labels (0-9)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 10, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embed = self.label_emb(labels)
        x = torch.cat([z, label_embed], dim=1).unsqueeze(2).unsqueeze(3)
        return self.model(x)

# ✅ Streamlit cache for loading model
@st.cache_resource
def load_generator():
    model = Generator().to(device)
    model.load_state_dict(torch.load("generator.pth", map_location=device))
    model.eval()
    return model

# ✅ Generate images
def generate_images(model, digit, count=5):
    z = torch.randn(count, 100).to(device)
    labels = torch.full((count,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        images = model(z, labels).cpu()
    return images

# ✅ Display images
def show_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 2))
    for i, img in enumerate(images):
        axs[i].imshow(img.squeeze(0), cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)

# ✅ Streamlit UI
st.title("🧠 Handwritten Digit Generator (0–9) - DCGAN Version")
st.write("Select a digit and generate 5 synthetic handwritten digit images using a cDCGAN model.")

selected_digit = st.selectbox("Choose a digit (0–9)", list(range(10)))

if st.button("Generate Images"):
    model = load_generator()
    images = generate_images(model, selected_digit)
    show_images(images)
