import json
import os
import redis
import torch
import torch.nn.functional as F
import zlib

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

from PIL import Image
from io import BytesIO

from fastapi import FastAPI, Form, File
from fastapi.responses import  JSONResponse

app = FastAPI()
REDIS_HOST=os.environ.get('REDIS_HOST')
REDIS_PORT=os.environ.get('REDIS_PORT')

@app.on_event('startup')
async def initialize():
    weights = ResNet50_Weights.DEFAULT
    global MODEL 
    global LABELS
    with open("imagenet_class_index.json", 'r') as f:   
        class_idx = json.load(f)
        LABELS = [class_idx[str(k)][1] for k in range(len(class_idx))]
    MODEL = resnet50(weights=weights)
    MODEL.eval()
    pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0)
    global REDIS
    REDIS = redis.Redis(connection_pool=pool)

def _preprocess_image(img):
    img_pil = Image.open(BytesIO(img)).convert('RGB')
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    t = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=imagenet_mean, std=imagenet_std)]
            )
    img_tensor = t(img_pil)
    return torch.unsqueeze(img_tensor, 0)

def _classify_image(preprocessed_image):
    global MODEL
    out = MODEL(preprocessed_image)
    _, index = torch.max(out, 1)
    pct = F.softmax(out, dim=1)[0] * 100
    return (LABELS[index[0]], pct[index[0]].item())

@app.post('/classify')
async def image(data: bytes = File()):
    preprocessed_image = _preprocess_image(data)
    result = _classify_image(preprocessed_image)
    await write_to_cache(data, result)
    return JSONResponse({'content': result[0], 'confidence': str(result[1])})

async def write_to_cache(file, result):
    hash = zlib.adler32(file)
    REDIS.set(hash, json.dumps({'content': result[0], 'confidence': str(result[1])}))
