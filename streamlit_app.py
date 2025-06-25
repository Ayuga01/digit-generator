import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cpu')

# ✅ Corrected cDCGAN Generator Class (Matches Your Colab Training)
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(10, 10)  # Must match Colab model definition!

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
        label_embed = self.label_embed(labels)
        x = torch.cat([z, label_embed], dim=1).unsqueeze(2).unsqueeze(3)
        return self.model(x)

# ✅ Streamlit Cached Model Loader
@st.cache_resource
def load_generator():
    model = Generator().to(device)
    model.load_state_dict(torch.load("generator.pth", map_location=device))
    model.eval()
    return model

# ✅ Generate Images
def generate_images(model, digit, count=5):
    z = torch.randn(count, 100).to(device)
    labels = torch.full((count,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        images = model(z, labels).cpu()
    return images

# ✅ Display Images
def show_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 2))
    for i, img in enumerate(images):
        img_np = img.squeeze().numpy()
        axs[i].imshow(img_np, cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)

# ✅ Streamlit UI
st.title("🧠 Handwritten Digit Generator (0–9) - Improved DCGAN")
st.write("Select a digit and generate 5 clean synthetic handwritten digit images using a cDCGAN model.")

selected_digit = st.selectbox("Choose a digit (0–9)", list(range(10)))

if st.button("Generate Images"):
    model = load_generator()
    images = generate_images(model, selected_digit)
    show_images(images)
