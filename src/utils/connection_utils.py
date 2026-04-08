from pymongo import MongoClient
import time, os

MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get('MONGODB_DB_NAME')

start = time.perf_counter()
try:
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    print(f"Connected to MongoDB database '{DB_NAME}' successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    db = None

print(f"MongoDB connection completed in {time.perf_counter() - start:.2f}s.")
