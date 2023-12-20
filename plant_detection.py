import torch
from torchvision import datasets, transforms, models  # datsets  , transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from nongbu_model import CNN

targets_size = 39
model = CNN(targets_size)
model.load_state_dict(torch.load("C:/Users/ganks/projects/nongbu_friend/plant_disease_model_1_latest.pt"))
model.eval()


data = pd.read_csv("C:/Users/ganks/projects/nongbu_friend/disease_info.csv", encoding="cp1252")



def single_prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    print("Original : ", image_path[12:-4])
    pred_csv = data["disease_name"][index]
    print(pred_csv)
    return pred_csv

# single_prediction("C:/Users/ganks/projects/nongbu_friend/grape_black_rot.JPG")