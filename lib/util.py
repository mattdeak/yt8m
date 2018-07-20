import time
import random

def timed(func):
    """Decorator that makes function print out its total runtime."""
    def _func(*args):
        print(f"Starting {func.__name__}...")
        start_time = time.time()
        out = func(*args)
        print(f"{func.__name__} took {time.time() - start_time} seconds")
        return out
    return _func


def dict_sample(d, n):
    keys = random.sample(list(d), n)
    return {k: d[k] for k in keys}


class imputer_dict(dict):
    """A dictionary replacement that causes KeyErrors to default to a given key."""
    def __init__(self, default_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_key = default_key

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            return super().__getitem__(self.default_key)
