import pandas as pd
import json
import db
import os
from bson import ObjectId
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
import shutil
import uuid
from Indexer import ProductSearchIndexer
from QueryParser import QueryRewriter
from ImageMaster import BLIPCaptionGenerator
from AudioMaster import AudioSearchPipeline
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods
    allow_headers=["*"],       # Allow all headers
)
indexer = ProductSearchIndexer()
QueryRewriter = QueryRewriter()
image_master= BLIPCaptionGenerator(device="cuda")
audio_master = AudioSearchPipeline()
def load_data(path):
    data = pd.read_csv(path)
    return data

def format_json(data):
    return json.loads(data.to_json(orient='records'))

# Helper to convert MongoDB _id to string
def convert_doc(doc):
    doc['id'] = str(doc['_id'])
    del doc['_id']
    return doc

# class Product(BaseModel):
#     text: str|None

# Insert into MongoDB
@app.post("/items")
def create_item(product):
    result = db.collection.insert_one(product.model_dump())
    return {"product": result}

# Read all from MongoDB
@app.get("/items")
def read_items():
    items = db.collection.find()
    return [convert_doc(item) for item in items]
    
# @app.get("/items/{ids}")
from bson import ObjectId

def get_items(ids):
    try:
        # Convert string IDs to ObjectId instances
        object_ids = [ObjectId(id.strip()) for id in ids]

        # Query using ObjectId
        cursor = db.collection.find({"_id": {"$in": object_ids}})
        items = []

        for item in cursor:
            item["id"] = str(item["_id"])
            del item["_id"]
            items.append(item)

        return items

    except Exception as e:
        return {"error": str(e)}
@app.post("/search")
async def handle_input(
    text: Optional[str] = Form(None),
    image:  UploadFile | str | None = File(None),
    voice: UploadFile | str | None = File(None)
):
    if not text and not image and not voice:
        return JSONResponse(status_code=400, content={"error": "No input provided"})
    result = {}
    q = ""
    # Handle text input
    if text:
        q = q + text

    # === Handle Image ===
    if image:
        image_path = f"temp_{uuid.uuid4()}_{image.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # You can add preprocessing logic here (e.g., image classification)
        # result["image_stored_as"] = image_path
        q_image= image_master.generate_caption(image_path)
        print(q_image)
        q = q + " " + q_image
        # Delete after processing
        os.remove(image_path)

    # === Handle Voice ===
    if voice:
        voice_path = f"temp_{uuid.uuid4()}_{voice.filename}"
        with open(voice_path, "wb") as buffer:
            shutil.copyfileobj(voice.file, buffer)
        
        # You can add voice processing logic here (e.g., speech-to-text)
        # result["voice_stored_as"] = voice_path
        q_audio=audio_master.process(voice_path)
        q = q + " " + q_audio
        print(q_audio)

        # Delete after processing
        os.remove(voice_path)
    # === Search Logic ===
    structured_query = QueryRewriter.rewrite(q)
    super_query = " ".join([
    structured_query.get("query") or "",
    structured_query.get("category") or "",
    structured_query.get("intent") or ""
]).strip()
    print(super_query)
    search_ids = indexer.search(super_query)
    print(search_ids)
    products = get_items(search_ids)
    result["products"] = products
    result["structured_query"] = structured_query
    return result

@app.get("/")
def read_root():
    data = load_data('./amazon_products.csv')
    data = format_json(data.head())
    return {}