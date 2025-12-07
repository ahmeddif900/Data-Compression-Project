import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os

# ==================== IMAGE UTILITIES ====================
def create_histogram_comparison(original_array, compressed_array):
    """Create histogram comparison plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original histogram
    axes[0].hist(original_array.flatten(), bins=50, alpha=0.7, color='blue', label='Original')
    axes[0].set_title('Original Image Histogram')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Compressed histogram
    axes[1].hist(compressed_array.flatten(), bins=50, alpha=0.7, color='green', label='Compressed')
    axes[1].set_title('Compressed Image Histogram')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_quantization_table_plot(quantization_table):
    """Create visualization of quantization table"""
    q_values = [row['Q'] for row in quantization_table]
    q_inverses = [row['Q_inverse'] for row in quantization_table]
    ranges = [row['range'] for row in quantization_table]
    
    fig = go.Figure()
    
    # Add Q values
    fig.add_trace(go.Scatter(
        x=q_values, y=q_inverses,
        mode='markers+lines',
        name='Q vs Q^-1',
        marker=dict(size=10, color='blue')
    ))
    
    # Add range bars
    for i, (start, end) in enumerate(ranges):
        fig.add_shape(
            type="line",
            x0=q_values[i], y0=start,
            x1=q_values[i], y1=end,
            line=dict(color="red", width=2),
            name=f"Range {i}"
        )
    
    fig.update_layout(
        title="Quantization Table Visualization",
        xaxis_title="Quantization Level (Q)",
        yaxis_title="Pixel Value",
        showlegend=True,
        height=500
    )
    
    return fig

def create_error_distribution_plot(original_array, compressed_array):
    """Create error distribution plot"""
    errors = original_array - compressed_array
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors.flatten(),
        nbinsx=50,
        name='Error Distribution',
        marker_color='orange',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Error Distribution (Original - Compressed)",
        xaxis_title="Error Value",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig
def load_image_for_quantization(image_path):
    """Load and prepare image for quantization"""
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        return img_array, img.copy(), os.path.basename(image_path)
        
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")

def save_quantized_image(reconstructed_img, original_filename, bit_depth, output_dir="output"):
    """Save quantized image with proper naming"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    name, ext = os.path.splitext(original_filename)
    output_filename = f"{name}_quantized_{bit_depth}bit{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save with optimization
    reconstructed_img.save(output_path, optimize=True, quality=95)
    
    return output_path, output_filename

def get_image_info(image_array, image_name):
    """Get basic image information"""
    height, width, channels = image_array.shape
    total_pixels = height * width
    original_size_bits = total_pixels * 24  # 24 bits per pixel for RGB
    
    return {
        'name': image_name,
        'dimensions': f"{width} × {height}",
        'pixels': total_pixels,
        'channels': channels,
        'original_size_bits': original_size_bits,
        'original_size_mb': (original_size_bits / 8) / (1024 * 1024)
    }