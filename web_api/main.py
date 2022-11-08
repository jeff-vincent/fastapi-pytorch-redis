import os
import aiohttp
import json
import redis
import zlib
from fastapi import FastAPI, File
from fastapi.responses import  HTMLResponse

app = FastAPI()
REDIS_HOST=os.environ.get('REDIS_HOST')
REDIS_PORT=os.environ.get('REDIS_PORT')
PYTORCH_HOST=os.environ.get('PYTORCH_HOST')
PYTORCH_PORT=os.environ.get('PYTORCH_PORT')

@app.on_event('startup')
async def initialize():
    pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
    global REDIS
    REDIS = redis.Redis(connection_pool=pool)

@app.get('/')
async def index_view():
    return HTMLResponse("""
        <div style="background-color: #707bb2; margin: 15px; border-radius: 5px; padding: 15px; width: 300px">
        <b>Upload an image: </b>
        <form action="/classify" method="post" enctype="multipart/form-data">
            <p><input type=file name=file value="Pick an image">
            <p><input type=submit value="Upload">
        </form>
        </div>""")

@app.post('/classify')
async def classify_image(file: bytes = File()):
    cached_data = await check_for_cached(file)
    if cached_data == None:
        form = aiohttp.FormData()
        form.add_field('data', file)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f'http://{PYTORCH_HOST}:{PYTORCH_PORT}/classify', data=form) as response:
                    r = await response.text()
                    data = json.loads(r)
                    return data
        except Exception as e:
            return HTMLResponse(f'<h3>Error:{str(e)}</h3>')
    return cached_data

async def check_for_cached(file):
    hash = zlib.adler32(file)
    data = REDIS.get(hash)
    if data:
        return json.loads(data)
    return None
    