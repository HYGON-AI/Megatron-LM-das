def offloading_checker(tensor):
    return hasattr(tensor, "offloading_activation") and tensor.offloading_activation
