"""
db.py — MongoDB connection (singleton)
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

_client = None
_db = None


def get_db():
    global _client, _db
    if _db is None:
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise RuntimeError("MONGO_URI not set in environment")
        _client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        _db = _client["zatch"]
    return _db