import numpy as np

def add_gaussian_noise(image_bgr: np.ndarray, mean: float = 0.0, std: float = 10.0):
    noisy = image_bgr.astype(np.float32)
    noise = np.random.normal(mean, std, image_bgr.shape).astype(np.float32)
    noisy = noisy + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_speckle_noise(image_bgr: np.ndarray, var: float = 0.05):
    noisy = image_bgr.astype(np.float32)
    noise = np.random.normal(0, np.sqrt(var), image_bgr.shape).astype(np.float32)
    noisy = noisy + noisy * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
