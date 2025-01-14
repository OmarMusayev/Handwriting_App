import torch
import numpy as np
import sys
import os
# import matplotlib.pyplot as plt

sys.path.append("../")
from utils import plot_stroke
from utils.constants import Global
from utils.dataset import HandwritingDataset
from utils.data_utils import (
    data_denormalization,
    data_normalization,
    valid_offset_normalization,
)
from models.models import HandWritingSynthesisNet
from generate import generate_conditional_sequence
from app.model_singleton import get_preloaded_model

def generate_handwriting(
    char_seq="hello world",
    real_text="",
    style_path="../app/static/mobile/style.npy",
    save_path="",
    app_path="",
    n_samples=1,
    bias=10.0,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(app_path, "../data/")
    
    # Instead of using a model_path, we get the preloaded model from our singleton:
    model = get_preloaded_model()
    
    # Initialize the dataset:
    train_dataset = HandwritingDataset(data_path, split="train", text_req=True)

    prime = True
    is_map = False

    # Load style file (adjust the path if necessary)
    style = np.load(style_path, allow_pickle=True, encoding="bytes").astype(np.float32)
    
    print("Priming text: ", real_text)
    mean, std, style = data_normalization(style)
    style = torch.from_numpy(style).unsqueeze(0).to(device)
    print("Priming sequence size: ", style.shape)
    ytext = real_text + " " + char_seq + "  "

    for i in range(n_samples):
        # Now pass the preloaded model instead of a model path
        gen_seq, phi = generate_conditional_sequence(
            model,            # Preloaded model instance from the singleton
            char_seq,
            device,
            train_dataset.char_to_id,
            train_dataset.idx_to_char,
            bias,
            prime,
            style,
            real_text,
            is_map,
        )
        
        # Denormalize the generated offsets using training set stats
        print("data denormalization...")
        end = style.shape[1]
        gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)
        print("Generated sequence shape:", gen_seq.shape)
        
        # Plot the generated stroke sequence and save the image
        plot_stroke(
            gen_seq[0], os.path.join(save_path, "gen_stroke_" + str(i) + ".png")
        )
        print("Image saved to:", save_path)
