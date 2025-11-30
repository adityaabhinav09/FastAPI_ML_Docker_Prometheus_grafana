import joblib
import json
import pandas as pd
from app.core.config import settings
from app.cache.redis_cache import set_cache_prediction, get_cached_prediction


model = joblib.load(settings.MODEL_PATH)

def predict_car_price(data: dict):
    # Create a deterministic cache key using sorted JSON
    cache_key = json.dumps(data, sort_keys=True)
    cached = get_cached_prediction(cache_key)
    if cached is not None:
        return cached
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]
    set_cache_prediction(cache_key, prediction)
    return prediction

