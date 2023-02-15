"""Generic utilities.
Decorators for logging, timing and more."""
import logging
import functools
import time
import tqdm
from pathlib import Path
from typing import Union


def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


def log(func):
    """Print when entering and exiting function"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        print(f"Entering {func.__name__}...")
        value = func(*args, **kwargs)
        print(f"Exited {func.__name__}!")
        return value
    return wrapper_debug

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        print(f"Timing {func.__name__}...")
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs!")
        return value
    return wrapper_timer

def append_suffix(path: Union[str, Path], suffix="", separator=""):
    """Append suffix to the name portion of a path or path-like string"""
    old_path = Path(path)
    new_name = f"{old_path.name}{separator}{suffix}"
    new_path = old_path.with_name(new_name)
    if isinstance(path, str):
        return str(new_path)
    elif isinstance(path, Path):
        return new_path