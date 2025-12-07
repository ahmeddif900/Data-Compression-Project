import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==================== METRICS CALCULATION ====================

def calculate_color_metrics(original_array, reconstructed_array, bit_depth):
    """Calculate performance metrics for image quantization"""
    # Vectorized MSE calculation
    diff = original_array - reconstructed_array.astype(np.float32)
    mse_per_channel = np.mean(diff ** 2, axis=(0, 1))
    mse_overall = np.mean(mse_per_channel)
    
    # PSNR
    psnr = 10 * np.log10((255 ** 2) / mse_overall) if mse_overall > 0 else float('inf')
    
    # Compression info
    bits_before_total = 24  # 8 bits × 3 channels
    bits_after_total = bit_depth * 3
    
    compression_ratio_bits = bits_before_total / bits_after_total
    bits_saved = bits_before_total - bits_after_total
    percent_saved = (1 - bits_after_total / bits_before_total) * 100
    
    # File size estimation
    height, width = original_array.shape[:2]
    original_bits = height * width * bits_before_total
    compressed_bits = height * width * bits_after_total
    size_reduction = ((original_bits - compressed_bits) / original_bits) * 100
    
    return {
        'mse_per_channel': mse_per_channel,
        'mse_overall': mse_overall,
        'psnr': psnr,
        'bits_before_total': bits_before_total,
        'bits_after_total': bits_after_total,
        'compression_ratio_bits': compression_ratio_bits,
        'compression_ratio_percent': compression_ratio_bits * 100,
        'bits_saved': bits_saved,
        'percent_saved': percent_saved,
        'estimated_size_reduction': size_reduction,
        'original_size_bits': original_bits,
        'compressed_size_bits': compressed_bits,
        'bit_depth': bit_depth,
        'color_levels': 2 ** bit_depth,
        'total_colors': (2 ** bit_depth) ** 3
    }

# ==================== VISUALIZATION ====================

def create_comparison_plot(original_img, reconstructed_img, bit_depth, metrics):
    """Create comparison plot for Streamlit"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert to arrays
    original_array = np.array(original_img)
    reconstructed_array = np.array(reconstructed_img)
    
    # Original image
    axes[0, 0].imshow(original_array)
    axes[0, 0].set_title('Original Image', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Quantized image
    axes[0, 1].imshow(reconstructed_array)
    axes[0, 1].set_title(f'Quantized ({bit_depth}-bit/channel)', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference map
    diff = np.mean(np.abs(original_array.astype(float) - reconstructed_array.astype(float)), axis=2)
    im = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=64)
    axes[1, 0].set_title('Difference Map', fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Metrics table
    axes[1, 1].axis('off')
    metrics_text = f"""
    Quantization Results:
    
    Bit-depth: {bit_depth}-bit/channel
    Levels per channel: {2**bit_depth}
    Total colors: {(2**bit_depth)**3:,}
    
    Quality Metrics:
    MSE: {metrics['mse_overall']:.2f}
    PSNR: {metrics['psnr']:.2f} dB
    
    Compression:
    Original: {metrics['bits_before_total']} bpp
    Quantized: {metrics['bits_after_total']} bpp
    Savings: {metrics['bits_saved']} bpp
    Size reduction: {metrics['estimated_size_reduction']:.1f}%
    Ratio: {metrics['compression_ratio_bits']:.2f}:1
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_histogram_comparison(original_array, reconstructed_array):
    """Create histogram comparison of color distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['Red', 'Green', 'Blue']
    
    for i in range(3):
        axes[i].hist(original_array[:, :, i].flatten(), bins=50, alpha=0.5, 
                    color=colors[i].lower(), density=True, label='Original')
        axes[i].hist(reconstructed_array[:, :, i].flatten(), bins=50, alpha=0.5,
                    color='gray', density=True, label='Quantized')
        axes[i].set_title(f'{colors[i]} Channel Distribution')
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig