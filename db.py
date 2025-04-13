import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables
load_dotenv()

# Load values from .env
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION")

# MongoDB client
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Optional: Test connection
try:
    client.admin.command("ping")
    print("✅ Connected to MongoDB")
except Exception as e:
    print("❌ MongoDB connection error:", e)