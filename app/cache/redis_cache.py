import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")

try:
    redis_client = redis.StrictRedis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    redis_client = None

def get_cached_prediction(key: str):
    if redis_client is None:
        return None
    try:
        value = redis_client.get(key)
        return float(value) if value else None
    except Exception:
        return None

def set_cache_prediction(key: str, value):
    if redis_client is None:
        return
    try:
        redis_client.set(key, str(float(value)))
    except Exception:
        pass

