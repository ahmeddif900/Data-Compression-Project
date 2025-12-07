import numpy as np
from PIL import Image
import time

def conversion_function(image):
    """Convert image to grayscale and flatten"""
    try:
        # Handle different input types
        if isinstance(image, str):
            # Path string
            img = Image.open(image)
        elif hasattr(image, 'read'):  # Uploaded file object
            img = Image.open(image)
        else:
            # Assume it's already a PIL Image
            img = image
        
        # Convert RGB to grayscale
        grayscale_img = img.convert('L')
        
        # Convert grayscale image to numpy array
        img_array = np.array(grayscale_img, dtype=np.float32)  # Keep as float
        
        # Save the shape before flattening
        original_shape = img_array.shape
        
        # Flatten 2D array to 1D array
        flattened_array = img_array.flatten()
        
        return flattened_array, original_shape, grayscale_img
    
    except Exception as e:
        raise Exception(f"Error in conversion_function: {str(e)}")

def non_uniform_scalar_decompression(pixel_array, max_iterations=10):
    """Find Q^-1 values through clustering (non-uniform quantization)"""
    # Start with all pixels in one cluster
    clusters = {-1: np.arange(len(pixel_array))}  # Initially all pixels in one cluster
    
    iteration = 0
    previous_centers = set()
    iteration_details = []
    
    while iteration < max_iterations:
        iteration += 1
        
        new_clusters = {}
        current_centers = set()
        
        for center, indices in clusters.items():
            if len(indices) == 0:
                continue
            
            # Get pixel values for this cluster
            cluster_values = pixel_array[indices]
            
            # Calculate average
            avg = np.mean(cluster_values)
            
            # Determine floor and ceil
            if iteration == 1 and avg == int(avg):
                # Special case: first iteration with integer average
                floor_val = int(avg) - 1
                ceil_val = int(avg) + 1
            else:
                floor_val = int(np.floor(avg))
                ceil_val = int(np.ceil(avg))
            
            # Store iteration details for display
            iteration_details.append({
                'iteration': iteration,
                'cluster_size': len(indices),
                'average': avg,
                'floor': floor_val,
                'ceil': ceil_val
            })
            
            # If floor == ceil (integer average after first iteration), no split
            if floor_val == ceil_val and iteration > 1:
                new_clusters[floor_val] = indices
                current_centers.add(floor_val)
            else:
                # Split based on distance to floor and ceil
                midpoint = (floor_val + ceil_val) / 2
                
                # Vectorized classification
                mask_floor = cluster_values < midpoint
                mask_ceil = ~mask_floor
                
                indices_floor = indices[mask_floor]
                indices_ceil = indices[mask_ceil]
                
                if len(indices_floor) > 0:
                    new_clusters[floor_val] = indices_floor
                    current_centers.add(floor_val)
                
                if len(indices_ceil) > 0:
                    new_clusters[ceil_val] = indices_ceil
                    current_centers.add(ceil_val)
        
        # Check convergence: if centers haven't changed, stop
        if current_centers == previous_centers:
            break
        
        previous_centers = current_centers
        clusters = new_clusters
    
    # Extract Q^-1 values (sorted cluster centers)
    q_inverse_values = np.array(sorted(current_centers))
    
    return q_inverse_values, iteration_details, iteration

def non_uniform_scalar_compression(pixel_array, q_inverse_values, full_scale=255, bit_depth=8):
    """Build quantization table and quantize pixels"""
    # Calculate levels number
    levels_number = 2 ** bit_depth
    
    # Verify that number of Q^-1 values matches levels_number
    if len(q_inverse_values) != levels_number:
        # If we have fewer Q^-1 values, interpolate to get required number
        if len(q_inverse_values) < levels_number:
            original_indices = np.linspace(0, len(q_inverse_values) - 1, len(q_inverse_values))
            new_indices = np.linspace(0, len(q_inverse_values) - 1, levels_number)
            q_inverse_values = np.interp(new_indices, original_indices, q_inverse_values).astype(int)
        elif len(q_inverse_values) > levels_number:
            # If we have more, keep only evenly spaced values
            indices = np.linspace(0, len(q_inverse_values) - 1, levels_number, dtype=int)
            q_inverse_values = q_inverse_values[indices]
    
    # Build quantization table
    quantization_table = []
    
    for i in range(levels_number):
        # Calculate range boundaries
        if i == 0:
            range_start = 0
        else:
            # Round to nearest integer: average of previous and current Q^-1
            range_start = int(round((q_inverse_values[i-1] + q_inverse_values[i]) / 2)) + 1
        
        if i == levels_number - 1:
            range_end = full_scale
        else:
            # Round to nearest integer: average of current and next Q^-1
            range_end = int(round((q_inverse_values[i] + q_inverse_values[i+1]) / 2))
        
        quantization_table.append({
            'range': (range_start, range_end),
            'Q': i,
            'Q_inverse': q_inverse_values[i],
            'range_size': range_end - range_start + 1
        })
    
    # Quantize the pixel array (vectorized for speed)
    quantized_array = np.zeros(len(pixel_array), dtype=np.int32)
    
    # Create a lookup array for faster quantization
    pixel_to_q = np.zeros(full_scale + 1, dtype=np.int32)
    
    for row in quantization_table:
        range_start = int(row['range'][0])
        range_end = int(row['range'][1])
        pixel_to_q[range_start:range_end+1] = row['Q']
    
    # **FIX: Convert clipped_pixels to integers before indexing**
    clipped_pixels = np.clip(pixel_array, 0, full_scale)
    
    # Ensure the pixel values are integers
    if clipped_pixels.dtype != np.int32 and clipped_pixels.dtype != np.int64:
        clipped_pixels = clipped_pixels.astype(np.int32)
    
    # Vectorized quantization using the lookup table
    quantized_array = pixel_to_q[clipped_pixels]
    
    return quantized_array, quantization_table, full_scale, bit_depth, q_inverse_values

def uniform_scalar_decompression(quantized_array, quantization_table, original_shape):
    """Decompress/reconstruct image from quantized data"""
    # Vectorized dequantization using lookup
    q_inverse_lookup = np.array([row['Q_inverse'] for row in quantization_table])
    dequantized_array = q_inverse_lookup[quantized_array]
    
    # Convert to uint8 for image representation
    dequantized_array = dequantized_array.astype(np.uint8)
    
    # Reshape back to original 2D shape
    reshaped_array = dequantized_array.reshape(original_shape)
    
    # Convert array back to PIL Image (mode 'L' for grayscale)
    decompressed_image = Image.fromarray(reshaped_array, mode='L')
    
    return decompressed_image, dequantized_array

def calculate_MSE(original_array, quantized_array, quantization_table):
    """Calculate Mean Squared Error"""
    # Vectorized lookup of Q^-1 values
    q_inverse_lookup = np.array([row['Q_inverse'] for row in quantization_table])
    dequantized_values = q_inverse_lookup[quantized_array]
    
    # Calculate error = (pixel_value - Q^-1)
    errors = original_array - dequantized_values
    
    # Calculate MSE = (Error^2) / n
    mse = np.sum(errors ** 2) / len(original_array)
    
    return mse

def calculate_compression_ratio(quantization_table, original_bits=8):
    """Calculate Compression Ratio (CR)"""
    # Get maximum value in Q column (last row Q value)
    max_q_value = quantization_table[-1]['Q']
    
    # Number of bits before compression
    bits_before = original_bits
    
    # Number of bits after compression
    bits_after = int(np.ceil(np.log2(max_q_value + 1)))
    
    # Compression ratio
    cr_percentage = (bits_before / bits_after) * 100
    cr_ratio = bits_before / bits_after
    
    return {
        'bits_before': bits_before,
        'bits_after': bits_after,
        'percentage': cr_percentage,
        'ratio': cr_ratio,
        'max_q_value': max_q_value
    }

def calculate_psnr(mse, max_pixel=255):
    """Calculate Peak Signal-to-Noise Ratio"""
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)

def analyze_quantization_table(quantization_table):
    """Analyze quantization table statistics"""
    ranges = [row['range'] for row in quantization_table]
    q_inverses = [row['Q_inverse'] for row in quantization_table]
    range_sizes = [row['range_size'] for row in quantization_table]
    
    return {
        'num_levels': len(quantization_table),
        'q_inverse_min': min(q_inverses),
        'q_inverse_max': max(q_inverses),
        'q_inverse_mean': np.mean(q_inverses),
        'range_size_min': min(range_sizes),
        'range_size_max': max(range_sizes),
        'range_size_mean': np.mean(range_sizes),
        'ranges': ranges,
        'q_inverses': q_inverses
    }