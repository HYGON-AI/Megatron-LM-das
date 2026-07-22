from functools import wraps


_CONFIG = None

def core_transformer_config_from_args_wrapper(func):
    @wraps(func)
    def wrapper(args, config_class=None):
        global _CONFIG
        if _CONFIG is not None:
            return _CONFIG

        _CONFIG = func(args, config_class=config_class)
        return _CONFIG
    
    return wrapper
