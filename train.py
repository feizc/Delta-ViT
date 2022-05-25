from PIL import Image 
import torch 
from model import transform, ViTModel, ViTConfig, ViTForImageClassification

image = Image.open('./data/test.jpg') 

preprocess = transform(224)
inputs = preprocess(image)

model = ViTModel.from_pretrained('./ckpt/vit') 

