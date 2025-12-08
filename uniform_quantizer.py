import numpy as np
from PIL import Image
import time

def build_quantization_table(bit_depth):
    """Build quantization table using vectorized operations"""
    levels = 2 ** bit_depth
    step_size = 256 / levels
    
    # Create ranges using vectorized operations
    range_starts = np.arange(levels) * step_size
    range_ends = np.minimum(range_starts + step_size, 256)
    
    # Q values
    q_values = np.arange(levels, dtype=np.int32)
    
    # Q^-1 values (centers of ranges)
    q_inverse = (range_starts + np.minimum(range_ends - 1, 255)) / 2
    
    # Create lookup tables for fast quantization
    pixel_to_q = np.zeros(256, dtype=np.int32)
    pixel_to_q_inverse = np.zeros(256, dtype=np.float32)
    
    for i in range(levels):
        start = int(range_starts[i])
        end = int(np.ceil(range_ends[i]))
        pixel_to_q[start:end] = q_values[i]
        pixel_to_q_inverse[start:end] = q_inverse[i]
    
    return {
        'levels': levels,
        'step_size': step_size,
        'range_starts': range_starts,
        'range_ends': range_ends,
        'q_values': q_values,
        'q_inverse': q_inverse,
        'pixel_to_q': pixel_to_q,
        'pixel_to_q_inverse': pixel_to_q_inverse
    }

def quantize_channel(channel_array, q_table):
    """Quantize a single channel using vectorized operations"""
    # Convert to integers for lookup (0-255)
    channel_int = channel_array.astype(np.uint8)
    
    # Vectorized lookup
    quantized_flat = q_table['pixel_to_q'][channel_int.flatten()]
    
    # Reshape back to original shape
    quantized = quantized_flat.reshape(channel_array.shape)
    
    return quantized

def quantize_color_image(img_array, bit_depth):
    """Quantize RGB color image using vectorized operations"""
    # Build quantization table once
    q_table = build_quantization_table(bit_depth)
    
    # Separate and quantize each channel
    r_quantized = quantize_channel(img_array[:, :, 0], q_table)
    g_quantized = quantize_channel(img_array[:, :, 1], q_table)
    b_quantized = quantize_channel(img_array[:, :, 2], q_table)
    
    return (r_quantized, g_quantized, b_quantized), q_table

def reconstruct_color_image(quantized_channels, q_table, original_shape):
    """Reconstruct color image using vectorized operations"""
    r_quantized, g_quantized, b_quantized = quantized_channels
    
    # Pre-calculate dequantized values for all possible Q values
    max_q = q_table['levels'] - 1
    q_to_pixel = np.zeros(max_q + 1, dtype=np.float32)
    
    for q in range(max_q + 1):
        q_to_pixel[q] = q_table['q_inverse'][q]
    
    # Vectorized dequantization for each channel
    channels = []
    for q_channel in [r_quantized, g_quantized, b_quantized]:
        dequantized_flat = q_to_pixel[q_channel.flatten()]
        dequantized = dequantized_flat.reshape(original_shape[:2])
        dequantized = np.clip(dequantized, 0, 255).astype(np.uint8)
        channels.append(dequantized)
    
    # Combine channels
    reconstructed_array = np.stack(channels, axis=-1)
    reconstructed_img = Image.fromarray(reconstructed_array, mode='RGB')
    
    return reconstructed_img, reconstructed_array