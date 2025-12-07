import numpy as np
from PIL import Image

def uniform_quantize_image(image_array, bit_depth):
    """Uniform quantization for color images - uses your existing code"""
    if len(image_array.shape) == 2:  # Grayscale
        # Add channel dimension for compatibility
        image_array = image_array[:, :, np.newaxis]
    
    # Build quantization table
    q_table = build_quantization_table_fast(bit_depth)
    
    # Quantize each channel
    quantized_channels = []
    for i in range(image_array.shape[2]):
        channel = image_array[:, :, i]
        quantized = quantize_channel_fast(channel, q_table)
        quantized_channels.append(quantized)
    
    # Reconstruct
    reconstructed_img, reconstructed_array = reconstruct_color_image_fast(
        tuple(quantized_channels), q_table, image_array.shape
    )
    
    return quantized_channels, reconstructed_img, reconstructed_array

def calculate_image_metrics(original_array, compressed_array):
    """Calculate image quality metrics"""
    # MSE and PSNR
    mse = np.mean((original_array - compressed_array) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse) if mse > 0 else float('inf')
    
    return {
        'mse': mse,
        'psnr': psnr
    }

# Import your existing uniform functions
# In uniform_quantizer.py, line ~70:
from image_quantization import (
    build_quantization_table_fast,
    quantize_channel_fast,
    reconstruct_color_image_fast
)