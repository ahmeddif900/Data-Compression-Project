import streamlit as st
import time
import base64
import numpy as np
from PIL import Image
import math
import struct
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from io import BytesIO

# Import from separate files
from rle import run_length_encoding, run_length_decoding
from huffman import huffman_encoding, huffman_decoding
from golomb import golomb_encoding
from rle_golomb import rle_golomb_encoding, rle_golomb_decoding
from lzw import LZW_encoding, LZW_decoding

# Import image compression methods
from image_quantization import (
    quantize_color_image_fast, 
    reconstruct_color_image_fast,
)
from non_uniform_quantizer import (
        conversion_function,
        non_uniform_scalar_decompression,
        non_uniform_scalar_compression,
        uniform_scalar_decompression,
        calculate_MSE,
        calculate_compression_ratio,
        calculate_psnr,
        analyze_quantization_table
    )
from image_metrics import (
    calculate_color_metrics, 
    create_comparison_plot,
    create_histogram_comparison
)
from image_utils import (
    create_histogram_comparison,
    create_quantization_table_plot,
    create_error_distribution_plot,
    load_image_for_quantization,
    save_quantized_image,
    get_image_info
)

# Page config
st.set_page_config(
    page_title="Data Compression Analyzer",
    layout="wide"
)

# Title
st.title("Data Compression Analyzer")
st.markdown("---")

# Initialize session state
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'input_bytes' not in st.session_state:
    st.session_state.input_bytes = None
if 'original_size' not in st.session_state:
    st.session_state.original_size = 0
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'quantized_image' not in st.session_state:
    st.session_state.quantized_image = None
if 'compression_metrics' not in st.session_state:
    st.session_state.compression_metrics = None

# Initialize session state
for key in [ 'grayscale_image', 'quantized_image', 'quantization_table', 'pixel_array',
            'original_shape', 'q_inverse_values', 'iteration_details','algorithm_type']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    input_type = st.radio(
        "Input Type:",
        ["Upload File", "Enter Text"]
    )
    
    compression_type = st.radio(
        "Compression Type:",
        ["Lossless Compression", "Lossy Compression"]
    )
    
    if compression_type == "Lossless Compression":
        algorithm = st.selectbox(
            "Algorithm:",
            ["Run-Length Encoding (RLE)", "Huffman Coding", "Golomb Coding","RLE + Golomb (Two-Stage)","LZW Coding", "Compare All"]
        )
    
        if algorithm == "Golomb Coding":
            m_value = st.number_input("Parameter m:", min_value=2, max_value=256, value=4)
    else:  # Lossy Compression
        # Add algorithm selection for lossy compression
        lossy_algorithm = st.selectbox(
            "Select Quantization Algorithm:",
            ["Uniform Scalar Quantization","Non-Uniform Scalar Quantization"]
        )

# Helper functions
def calculate_entropy(text):
    """Calculate Shannon entropy in bits per symbol"""
    if not text:
        return 0
    
    freq = Counter(text)
    entropy = 0
    total = len(text)
    
    for count in freq.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy

def get_compressed_size(compressed_data):
    """Get compressed size from different data types"""
    if isinstance(compressed_data, bytes):
        return len(compressed_data)
    elif isinstance(compressed_data, tuple):
        return len(compressed_data[0])
    elif isinstance(compressed_data, list):
        return len(compressed_data) * 2  # Approximate: 2 bytes per code
    else:
        return 0
def apply_non_uniform_quantization(image, bit_depth, full_scale, max_iterations):
    """Apply non-uniform quantization to image"""
    # Convert to grayscale and flatten
    flattened_pixels, original_shape, grayscale_img = conversion_function(image)
    
    # Find Q^-1 values (clustering)
    q_inverse_values, iteration_details, num_iterations = non_uniform_scalar_decompression(
        flattened_pixels, max_iterations
    )
    
    # Build quantization table and compress
    quantized_pixels, quant_table, fs, bd, q_inv = non_uniform_scalar_compression(
        flattened_pixels, q_inverse_values, full_scale, bit_depth
    )
    
    # Decompress/reconstruct image
    decompressed_img, decompressed_array = uniform_scalar_decompression(
        quantized_pixels, quant_table, original_shape
    )
    
    # Calculate metrics
    mse = calculate_MSE(flattened_pixels, quantized_pixels, quant_table)
    psnr = calculate_psnr(mse)
    cr_info = calculate_compression_ratio(quant_table)
    table_stats = analyze_quantization_table(quant_table)
    
    metrics = {
        'mse': mse,
        'psnr': psnr,
        'compression_ratio': cr_info,
        'table_stats': table_stats,
        'q_inverse_values': q_inv,
        'bit_depth': bd,
        'levels': 2 ** bd,
        'full_scale': fs,
        'num_iterations': num_iterations,
        'iteration_details': iteration_details,
        'quantization_table': quant_table,
        'q_inverse_values_raw': q_inverse_values,
        'original_pixels': flattened_pixels,
        'quantized_pixels': quantized_pixels
    }
    
    return decompressed_img, decompressed_array, quantized_pixels, metrics, grayscale_img
# Main area
st.header("Input Data")
if compression_type == "Lossless Compression":
    if input_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=['txt', 'csv', 'json', 'xml', 'py', 'html', 'cpp', 'java']
        )
        
        if uploaded_file:
            file_bytes = uploaded_file.read()
            file_name = uploaded_file.name
            text_data = file_bytes.decode('utf-8', errors='ignore')
            original_size = len(file_bytes)
            
            # Store in session state
            st.session_state.input_data = text_data
            st.session_state.input_bytes = file_bytes
            st.session_state.original_size = original_size
            st.session_state.file_name = file_name
            
            # Display file info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("File Name", file_name)
            with col2:
                st.metric("File Size", f"{original_size:,} bytes")
            with col3:
                st.metric("Characters", f"{len(text_data):,}")
            with col4:
                entropy = calculate_entropy(text_data)
                st.metric("Entropy", f"{entropy:.3f} bits/char")
            
            with st.expander("File Preview (First 500 characters)"):
                st.text(text_data[:500])
            
    else:  # Enter Text
        input_text = st.text_area(
            "Enter text to compress:",
            height=200,
            value="AAAAAAAAAABBBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEEEEEEEEE"
        )
        
        if input_text:
            text_data = input_text
            file_bytes = input_text.encode('utf-8')
            original_size = len(file_bytes)
            
            # Store in session state
            st.session_state.input_data = text_data
            st.session_state.input_bytes = file_bytes
            st.session_state.original_size = original_size
            st.session_state.file_name = "text_input.txt"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", f"{len(text_data):,}")
            with col2:
                st.metric("Size", f"{original_size:,} bytes")
            with col3:
                entropy = calculate_entropy(text_data)
                st.metric("Entropy", f"{entropy:.3f} bits/char")
        else:
            st.info("Enter some text to compress")
            # Clear session state
            st.session_state.input_data = None
            st.session_state.input_bytes = None
            st.session_state.original_size = 0

    # Check if we have data to process
    if st.session_state.input_data and st.session_state.input_bytes:
        input_data = st.session_state.input_data
        input_bytes = st.session_state.input_bytes
        original_size = st.session_state.original_size
        
        st.markdown("---")
        
        if algorithm != "Compare All":
            # Single algorithm mode
            st.header(f"Algorithm: {algorithm}")
            
            # Algorithm info
            with st.expander(" Algorithm Information"):
                if algorithm == "Run-Length Encoding (RLE)":
                    st.markdown("""
                    **How RLE Works:**
                    - Scans data for consecutive repeated characters
                    - Replaces runs with [count, character] pairs
                    - Example: "AAAAABBB" becomes [5, 'A', 3, 'B']
                    
                    **Best for:** Files with long repeated sequences
                    **Time Complexity:** O(n)
                    """)
                    
                    # Show RLE example
                    if len(input_data) > 10:
                        sample = input_data[:10]
                        runs = []
                        i = 0
                        while i < len(sample):
                            count = 1
                            while i + count < len(sample) and sample[i] == sample[i + count]:
                                count += 1
                            runs.append(f"'{sample[i]}' × {count}")
                            i += count
                        st.write(f"**Example (first 10 chars):** {' → '.join(runs)}")
                        
                elif algorithm == "Huffman Coding":
                    st.markdown("""
                    **How Huffman Coding Works:**
                    - Creates optimal prefix codes based on character frequencies
                    - Frequent characters get shorter codes
                    - Builds a binary tree structure
                    
                    **Best for:** Text files with varied character frequencies
                    **Time Complexity:** O(n log n)
                    """)
                    
                elif algorithm == "Golomb Coding":
                    st.markdown(f"""
                    **How Golomb Coding Works:**
                    - Parameterized coding with parameter m = {m_value}
                    - Encodes numbers using quotient (unary) and remainder (binary)
                    - Optimized for geometric distributions
                    
                    **Best for:** Numerical data, run-length encoded data
                    **Parameter m:** Controls coding efficiency
                    """)
                    
                    # Show Golomb example
                    if len(input_bytes) > 0:
                        byte_val = input_bytes[0]
                        q = byte_val // m_value
                        r = byte_val % m_value
                        b = math.ceil(math.log2(m_value))
                        
                        unary_code = "1" * q + "0"
                        if r < 2**b - m_value:
                            binary_code = format(r, f'0{b-1}b')
                        else:
                            binary_code = format(r + 2**b - m_value, f'0{b}b')
                        
                        st.write(f"**Example (first byte {byte_val}):**")
                        st.write(f"- Quotient (q): {q} → Unary: {unary_code}")
                        st.write(f"- Remainder (r): {r} → Binary: {binary_code}")
                        st.write(f"- Total code: {unary_code + binary_code}")
                        
                elif algorithm == "LZW Coding":
                    st.markdown("""
                    **How LZW Works:**
                    - Dictionary-based compression
                    - Builds dictionary of strings dynamically
                    - Replaces repeated phrases with dictionary indices
                    
                    **Best for:** Text with repeated phrases
                    **Used in:** GIF, TIFF, Unix compress
                    **Time Complexity:** O(n)
                    """)
            
            if st.button(f"Run {algorithm}", type="primary"):
                start_time = time.time()
                
                try:
                    compressed_data = None
                    decompressed_bytes = None
                    extra_info = {}
                    
                    if algorithm == "Run-Length Encoding (RLE)":
                        compressed_data = run_length_encoding(input_data)
                        decompressed_data = run_length_decoding(compressed_data)
                        decompressed_bytes = decompressed_data.encode('utf-8')
                        extra_info['type'] = 'rle'
                        
                    elif algorithm == "Huffman Coding":
                        compressed_bytes, codes, tree, encoded_bits, padding = huffman_encoding(input_bytes)
                        compressed_data = (compressed_bytes, codes, tree, encoded_bits, padding)
                        decompressed_bytes = huffman_decoding(compressed_bytes, codes, padding)
                        extra_info['type'] = 'huffman'
                        extra_info['codes'] = codes
                        extra_info['padding'] = padding
                        
                    elif algorithm == "Golomb Coding":
                        compressed_data = golomb_encoding(input_bytes, m_value)
                        decompressed_bytes = None
                        extra_info['type'] = 'golomb'
                    elif algorithm == "RLE + Golomb (Two-Stage)":
                        try:
                            # Let user choose m parameter
                            col1, col2 = st.columns(2)
                            with col1:
                                m_value = st.number_input(
                                    "Golomb Parameter m:", 
                                    min_value=2, 
                                    max_value=128, 
                                    value=4,
                                    key="rle_golomb_m"
                                )
                            with col2:
                                show_details = st.checkbox("Show compression stages", value=True)
                            
                            # Perform two-stage compression
                            start_time = time.time()
                            compressed_data = rle_golomb_encoding(input_data, m_value)
                            compression_time = time.time() - start_time
                            
                            # Test decompression
                            decompressed_text = rle_golomb_decoding(compressed_data)
                            decompressed_bytes = decompressed_text.encode('utf-8')
                            
                            extra_info['type'] = 'rle_golomb'
                            extra_info['m_value'] = m_value
                            
                            # Show compression stages if requested
                            if show_details:
                                with st.expander("Two-Stage Compression Details"):
                                    # Stage 1: RLE analysis
                                    runs = []
                                    count = 1
                                    for i in range(1, len(input_data)):
                                        if input_data[i] == input_data[i-1]:
                                            count += 1
                                        else:
                                            runs.append((count, input_data[i-1]))
                                            count = 1
                                    runs.append((count, input_data[-1]))
                                    
                                    st.write(f"**Stage 1: RLE Compression**")
                                    st.write(f"- Original characters: {len(input_data)}")
                                    st.write(f"- After RLE: {len(runs)} runs")
                                    st.write(f"- RLE compression ratio: {len(input_data)/len(runs):.1f}:1")
                                    
                                    # Show run length distribution
                                    run_lengths = [count for count, _ in runs]
                                    if run_lengths:
                                        avg_run = sum(run_lengths) / len(run_lengths)
                                        st.write(f"- Average run length: {avg_run:.1f}")
                                        st.write(f"- Run lengths: {run_lengths[:20]}{'...' if len(run_lengths) > 20 else ''}")
                                    
                                    # Stage 2: Golomb analysis
                                    st.write(f"\n**Stage 2: Golomb on Run Lengths**")
                                    st.write(f"- Parameter m: {m_value}")
                                    st.write(f"- Run lengths to encode: {len(run_lengths)}")
                                    
                                    # Calculate expected Golomb bits
                                    b = math.ceil(math.log2(m_value))
                                    threshold = 2**b - m_value
                                    total_bits = 0
                                    for rl in run_lengths[:100]:  # Sample
                                        q = rl // m_value
                                        r = rl % m_value
                                        total_bits += q + 1  # Unary part
                                        total_bits += b - 1 if r < threshold else b
                                    
                                    avg_bits_per_run = total_bits / min(len(run_lengths), 100)
                                    st.write(f"- Estimated bits per run: {avg_bits_per_run:.1f}")
                                    st.write(f"- Estimated total bits: {avg_bits_per_run * len(run_lengths):.0f}")
                                    
                        except Exception as e:
                            st.error(f"RLE+Golomb error: {str(e)}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())
                            compressed_data = b''
                            decompressed_bytes = None
                        
                    elif algorithm == "LZW Coding":
                        # Your LZW_encoding returns only codes list
                        compressed_codes = LZW_encoding(input_data)
                        
                        # Your LZW_decoding returns (text, dictionary)
                        decompressed_text, lzw_dict = LZW_decoding(compressed_codes)
                        decompressed_bytes = decompressed_text.encode('utf-8')
                        
                        # Store just the codes list
                        compressed_data = compressed_codes
                        extra_info['type'] = 'lzw'
                        extra_info['codes'] = compressed_codes
                        extra_info['dict'] = lzw_dict
                    
                    compression_time = time.time() - start_time
                    compressed_size = get_compressed_size(compressed_data)
                    ratio = calculate_compression_ratio(original_size, compressed_size)
                    
                    # Display results
                    st.subheader("Compression Results")
                    
                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Original Size", f"{original_size:,} B")
                    with col2:
                        st.metric("Compressed Size", f"{compressed_size:,} B")
                    with col3:
                        savings = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0
                        st.metric("Space Saved", f"{savings:.1f}%")
                    with col4:
                        st.metric("Compression Time", f"{compression_time:.3f} s")
                    
                    # Additional metrics
                    col5, col6 = st.columns(2)
                    with col5:
                        st.metric("Compression Ratio", f"{ratio:.2f}:1")
                    with col6:
                        entropy = calculate_entropy(input_data)
                        st.metric("Entropy", f"{entropy:.3f} bits/char")
                    
                    # Visualization
                    st.subheader("Visualization")
                    
                    # Size comparison chart
                    fig1 = go.Figure(data=[
                        go.Bar(name='Original', x=['Size'], y=[original_size], marker_color='blue'),
                        go.Bar(name='Compressed', x=['Size'], y=[compressed_size], marker_color='green')
                    ])
                    fig1.update_layout(
                        title="Size Comparison",
                        yaxis_title="Size (bytes)",
                        height=300
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Savings gauge
                    fig2 = go.Figure(data=[
                        go.Indicator(
                            mode="gauge+number",
                            value=savings,
                            title="Space Saved",
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "green" if savings > 50 else "orange" if savings > 20 else "red"},
                                'steps': [
                                    {'range': [0, 20], 'color': "lightcoral"},
                                    {'range': [20, 50], 'color': "lightyellow"},
                                    {'range': [50, 100], 'color': "lightgreen"}]
                            }
                        )
                    ])
                    fig2.update_layout(height=300)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Check decompression
                    st.subheader(" Decompression Test")
                    if decompressed_bytes is not None:
                        if decompressed_bytes == input_bytes:
                            st.success("**Decompression Successful!** Original and decompressed files match exactly.")
                        else:
                            # Check length first
                            if len(decompressed_bytes) != len(input_bytes):
                                st.error(f" **Length mismatch:** Original: {len(input_bytes)} bytes, Decompressed: {len(decompressed_bytes)} bytes")
                            else:
                                # Check character by character
                                mismatch_count = 0
                                for i in range(min(len(input_bytes), len(decompressed_bytes))):
                                    if input_bytes[i] != decompressed_bytes[i]:
                                        mismatch_count += 1
                                        if mismatch_count <= 3:
                                            st.write(f"Mismatch at position {i}: Original '{chr(input_bytes[i])}' ({input_bytes[i]}) vs Decompressed '{chr(decompressed_bytes[i])}' ({decompressed_bytes[i]})")
                                
                                if mismatch_count > 0:
                                    st.error(f"**Decompression Failed!** {mismatch_count} mismatches found")
                    else:
                        st.warning("Decompression not available for this algorithm")
                    
                    # Download section
                    st.subheader("Download")
                    
                    # Convert to bytes for download
                    if isinstance(compressed_data, bytes):
                        bytes_data = compressed_data
                    elif isinstance(compressed_data, tuple):
                        bytes_data = compressed_data[0]
                    elif isinstance(compressed_data, list):
                        # For LZW codes list
                        try:
                            # Convert list of integers to bytes
                            bytes_data = struct.pack(f'{len(compressed_data)}H', *compressed_data)
                        except:
                            # Fallback: convert to JSON string
                            import json
                            bytes_data = json.dumps(compressed_data).encode('utf-8')
                    else:
                        bytes_data = b''
                    
                    if bytes_data:
                        b64 = base64.b64encode(bytes_data).decode()
                        filename = f"compressed_{algorithm.replace(' ', '_')}_{st.session_state.file_name}.bin"
                        download_link = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download Compressed File</a>'
                        st.markdown(download_link, unsafe_allow_html=True)
                    
                    # Algorithm-specific details
                    if algorithm == "Huffman Coding" and 'codes' in extra_info:
                        with st.expander("Huffman Codes (Top 10)"):
                            codes = extra_info['codes']
                            freq = Counter(input_data)
                            top_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
                            
                            table_data = []
                            for char, count in top_chars:
                                code = codes.get(char, "N/A")
                                table_data.append({
                                    "Character": repr(char)[1:-1],
                                    "Frequency": count,
                                    "Code": code,
                                    "Length": len(code)
                                })
                            
                            df = pd.DataFrame(table_data)
                            st.table(df)
                    
                    elif algorithm == "LZW Coding" and 'dict' in extra_info:
                        with st.expander("🔍 LZW Details"):
                            codes = extra_info.get('codes', [])
                            lzw_dict = extra_info.get('dict', {})
                            
                            st.write(f"**Number of codes:** {len(codes)}")
                            st.write(f"**Dictionary size:** {len(lzw_dict)} entries")
                            
                            if codes and len(codes) > 0:
                                # Show first 10 codes
                                st.write("**First 10 codes:**")
                                for i, code in enumerate(codes[:10]):
                                    st.write(f"  Code {i}: {code}")
                                
                                # Show some dictionary entries
                                st.write("**Dictionary entries (sample):**")
                                dict_items = list(lzw_dict.items())[:10]
                                table_data = []
                                for code, string in dict_items:
                                    table_data.append({
                                        "Code": code,
                                        "String": repr(string)[1:-1][:30]
                                    })
                                
                                if table_data:
                                    df = pd.DataFrame(table_data)
                                    st.table(df)
                    
                except Exception as e:
                    st.error(f"Error during compression: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        else:
            # Compare All mode
            st.header("Algorithm Comparison")
            
            if st.button("Compare All Algorithms", type="primary"):
                algorithms_to_compare = [
                    ("RLE", "Run-Length Encoding"),
                    ("Huffman", "Huffman Coding"),
                    ("Golomb", "Golomb Coding"),
                    ("LZW", "LZW Coding"),
                    ("RLE_Golomb", "RLE + Golomb")
                ]
                
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (algo_code, algo_name) in enumerate(algorithms_to_compare):
                    status_text.text(f"Testing {algo_name}...")
                    progress_bar.progress((idx + 1) / len(algorithms_to_compare))
                    
                    try:
                        start_time = time.time()
                        
                        if algo_code == "RLE":
                            compressed = run_length_encoding(input_data)
                        elif algo_code == "Huffman":
                            result = huffman_encoding(input_bytes)
                            compressed = result[0]
                        elif algo_code == "Golomb":
                            compressed = golomb_encoding(input_bytes, m_value)
                        elif algo_code == "RLE_Golomb":
                            compressed = rle_golomb_encoding(input_data, m_value)
                        elif algo_code == "LZW":
                            compressed = LZW_encoding(input_data)  # Just codes list
                        
                        comp_time = time.time() - start_time
                        comp_size = get_compressed_size(compressed)
                        ratio = calculate_compression_ratio(original_size, comp_size)
                        
                        results.append({
                            "Algorithm": algo_name,
                            "Time (s)": round(comp_time, 3),
                            "Size (bytes)": comp_size,
                            "Ratio": round(ratio, 2)
                        })
                        
                    except Exception as e:
                        st.warning(f"{algo_name} failed: {str(e)}")
                
                progress_bar.empty()
                status_text.text("Comparison complete!")
                
                if results:
                    df = pd.DataFrame(results)
                    
                    # Display table
                    st.subheader("Comparison Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Size comparison chart
                    st.subheader("Size Comparison")
                    fig1 = go.Figure(data=[
                        go.Bar(
                            x=df["Algorithm"],
                            y=df["Size (bytes)"],
                            text=df["Size (bytes)"],
                            textposition='auto',
                            marker_color=['blue', 'green', 'orange', 'red']
                        )
                    ])
                    fig1.update_layout(
                        title="Compressed Size Comparison (lower is better)",
                        xaxis_title="Algorithm",
                        yaxis_title="Size (bytes)",
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Time comparison chart
                    st.subheader("Time Comparison")
                    fig2 = go.Figure(data=[
                        go.Bar(
                            x=df["Algorithm"],
                            y=df["Time (s)"],
                            text=df["Time (s)"],
                            textposition='auto',
                            marker_color=['blue', 'green', 'orange', 'red']
                        )
                    ])
                    fig2.update_layout(
                        title="Compression Time Comparison (lower is better)",
                        xaxis_title="Algorithm",
                        yaxis_title="Time (seconds)",
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Find best algorithms
                    best_size = df.loc[df["Size (bytes)"].idxmin()]
                    best_time = df.loc[df["Time (s)"].idxmin()]
                    best_ratio = df.loc[df["Ratio"].idxmax()]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f" Best Size:** {best_size['Algorithm']}\n{best_size['Size (bytes)']:,} bytes")
                    with col2:
                        st.info(f" Best Time:** {best_time['Algorithm']}\n{best_time['Time (s)']} seconds")
                    with col3:
                        st.info(f" Best Ratio:** {best_ratio['Algorithm']}\n{best_ratio['Ratio']}:1")
                    
                    # Recommendation
                    st.subheader(" Recommendation")
                    if len(input_data) < 100:
                        st.success("**For small text:** RLE is usually fastest and simple")
                    elif input_data.count(input_data[0]) > len(input_data) * 0.3:
                        st.success("**For repetitive text:** RLE gives best compression")
                    else:
                        st.success("**For general text:** Huffman or LZW are good choices")
    else:
        st.info(" Please upload a file or enter text to begin compression")

    # Information section
    st.markdown("---")
    with st.expander(" About Compression Algorithms"):
        st.markdown("""
        **Algorithm Comparison Guide:**
        
        | Algorithm | Best For | Speed | Compression | Complexity |
        |-----------|----------|-------|-------------|------------|
        | **RLE** | Repetitive data (AAAAABBB) | Very Fast | Good for repeats | O(n) |
        | **Huffman** | General text files | Moderate | Optimal prefix codes | O(n log n) |
        | **Golomb** | Geometric distributions | Fast | Parameter-dependent | O(n) |
        | **LZW** | Repeated phrases | Moderate | Dictionary-based | O(n) |
        
        **Key Metrics:**
        - **Compression Ratio:** Original size / Compressed size (higher is better)
        - **Space Saved:** Percentage reduction in size
        - **Entropy:** Theoretical minimum bits per character
        - **Time:** Processing time in seconds
        
        **Tips:**
        - For small files, speed matters more than compression ratio
        - For large files, compression ratio is more important
        - Test different algorithms to find the best for your data
        """)
else:  # Lossy Compression - Image Quantization
    st.header(" Lossy Compression - Image Quantization")
    
    # Algorithm selection
    if lossy_algorithm == "Uniform Scalar Quantization":
        st.session_state.algorithm_type = "Uniform Scalar Quantization"
        algorithm_desc = "Uniform quantization uses equal-sized intervals"
    else:
        st.session_state.algorithm_type = "Non-Uniform Scalar Quantization"
        algorithm_desc = "Non-uniform quantization adapts to image content"
    
    st.write(f"**Algorithm:** {st.session_state.algorithm_type}")
    st.write(f"**Description:** {algorithm_desc}")
    
    # Upload image section
    st.subheader("1. Upload Image File")
    
    uploaded_file = st.file_uploader(
        "Choose an image file:",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'],
        key="image_upload"
    )
    
    if uploaded_file:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            file_name = uploaded_file.name
            
            st.subheader("Original Image")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption=f"Original: {file_name}", use_column_width=True)
            
            with col2:
                st.write("**Image Information:**")
                st.write(f"**Filename:** {file_name}")
                st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]}")
                st.write(f"**Total Pixels:** {image.size[0] * image.size[1]:,}")
                st.write(f"**File Size:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")
                st.write(f"**Color Mode:** {image.mode}")
            
            # Parameters section
            st.subheader("2. Select Compression Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                bit_depth = st.slider(
                    "Bits per color channel:",
                    min_value=1,
                    max_value=8,
                    value=4 if st.session_state.algorithm_type == "Non-Uniform Scalar Quantization" else 6,
                    help="Lower values = more compression but lower quality"
                )
                
                if st.session_state.algorithm_type == "Non-Uniform Scalar Quantization":
                    max_iterations = st.slider(
                        "Maximum iterations for clustering:",
                        min_value=1,
                        max_value=20,
                        value=10
                    )
            
            with col2:
                # Compression prediction
                original_bpp = 24  # 8 bits × 3 channels
                compressed_bpp = bit_depth * 3
                compression_ratio = original_bpp / compressed_bpp
                size_reduction = ((original_bpp - compressed_bpp) / original_bpp) * 100
                
                st.metric("Predicted Compression", f"{compression_ratio:.1f}:1")
                st.metric("Predicted Size Reduction", f"{size_reduction:.1f}%")
                st.metric("New Bit Depth", f"{compressed_bpp} bpp")
            
            # Apply compression button
            st.subheader("3.  Apply Compression")
            
            if st.button(" Compress Image", type="primary", use_container_width=True):
                with st.spinner(f"Applying {bit_depth}-bit quantization..."):
                    start_time = time.time()
                    
                    if st.session_state.algorithm_type == "Uniform Scalar Quantization":
                        # Convert image to array
                        img_array = np.array(image.convert('RGB'), dtype=np.float32)
                        
                        # Apply uniform quantization
                        quantized_channels, q_table = quantize_color_image_fast(img_array, bit_depth)
                        recon_img, recon_array = reconstruct_color_image_fast(
                            quantized_channels, q_table, img_array.shape
                        )
                        
                        # Calculate metrics
                        metrics = calculate_color_metrics(img_array, recon_array, bit_depth)
                        
                    else:  # Non-Uniform Quantization
                        # For non-uniform, we work with grayscale
                        img_array = np.array(image.convert('L'), dtype=np.float32)
                        
                        # Apply non-uniform quantization
                        flattened_pixels = img_array.flatten()
                        original_shape = img_array.shape
                        
                        # Find Q^-1 values
                        q_inverse_values, iteration_details, num_iterations = non_uniform_scalar_decompression(
                            flattened_pixels, max_iterations
                        )
                        
                        # **FIX: Ensure pixel values are integers for indexing**
                        flattened_pixels_int = flattened_pixels.astype(np.int32)
                        
                        # Compress
                        quantized_pixels, quant_table, fs, bd, q_inv = non_uniform_scalar_compression(
                            flattened_pixels_int, q_inverse_values, 255, bit_depth
                        )
                        
                        # Decompress
                        recon_img, decompressed_array = uniform_scalar_decompression(
                            quantized_pixels, quant_table, original_shape
                        )
                        
                        # Calculate metrics
                        mse = calculate_MSE(flattened_pixels, quantized_pixels, quant_table)
                        psnr = calculate_psnr(mse)
                        cr_info = calculate_compression_ratio(quant_table, original_bits=8) 

                        metrics = {
                            'mse_overall': mse,
                            'psnr': psnr,
                            'compression_ratio_bits': cr_info['ratio'],
                            'estimated_size_reduction': ((8 - bit_depth) / 8) * 100,
                            'bits_before_total': 8,
                            'bits_after_total': bit_depth,
                            'original_size_bits': img_array.size * 8,
                            'compressed_size_bits': img_array.size * bit_depth
                        }
                    
                    compression_time = time.time() - start_time
                    
                    # Store results
                    st.session_state.original_image = image
                    st.session_state.quantized_image = recon_img
                    st.session_state.compression_metrics = metrics
                    st.session_state.compression_time = compression_time
                    st.session_state.bit_depth = bit_depth
                    
                    st.success(f"Compression completed in {compression_time:.2f} seconds!")
            
            # Display results if available
            if st.session_state.quantized_image is not None:
                st.markdown("---")
                st.subheader("4.  Compression Results")
                
                # Display comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state.original_image, caption="Original", use_column_width=True)
                with col2:
                    st.image(st.session_state.quantized_image, 
                            caption=f"Compressed ({st.session_state.bit_depth}-bit)", 
                            use_column_width=True)
                
                # Display metrics
                metrics = st.session_state.compression_metrics
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                with col2:
                    st.metric("MSE", f"{metrics['mse_overall']:.2f}")
                with col3:
                    st.metric("Compression Ratio", f"{metrics['compression_ratio_bits']:.2f}:1")
                with col4:
                    st.metric("Size Reduction", f"{metrics['estimated_size_reduction']:.1f}%")
                # Visualizations
                st.write("**Compression Visualization:**")
                
                # Create a compression comparison chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Original',
                    x=['Size'],
                    y=[metrics['original_size_bits']],
                    marker_color='blue',
                    text=[f"{metrics['original_size_bits']/1e6:.1f}M bits"],
                    textposition='auto'
                ))
                
                fig.add_trace(go.Bar(
                    name='Compressed',
                    x=['Size'],
                    y=[metrics['compressed_size_bits']],
                    marker_color='green',
                    text=[f"{metrics['compressed_size_bits']/1e6:.1f}M bits"],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Size Comparison (bits)",
                    yaxis_title="Bits",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                # Download section
                st.markdown("---")
                st.subheader("Download Compressed Image")
                
                buffered = BytesIO()
                st.session_state.quantized_image.save(buffered, format="PNG", optimize=True)
                b64 = base64.b64encode(buffered.getvalue()).decode()
                download_filename = f"compressed_{st.session_state.algorithm_type.replace(' ', '_')}_{bit_depth}bit_{file_name}"
                download_link = f'<a href="data:image/png;base64,{b64}" download="{download_filename}">⬇️ Download Compressed Image</a>'
                st.markdown(download_link, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    else:
        st.info(" Please upload an image file to begin lossy compression")
# Footer
st.markdown("---")
st.caption("Data Compression Project | All algorithms implemented from scratch | Streamlit Application")