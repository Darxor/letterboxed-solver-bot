from functools import wraps
from datetime import datetime
import logging

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        logging.info(f"{args[0].__class__.__name__}.{func.__name__} ran in: {end - start} for puzzle: {args[0].input_string}")
        return result
    return wrapper