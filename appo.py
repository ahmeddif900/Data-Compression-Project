import streamlit as st
import time
import base64
import math
import heapq
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import struct

# Page config
st.set_page_config(
    page_title="Data Compression Analyzer",
    layout="wide"
)

# Title
st.title("Data Compression Application")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    compression_type = st.radio(
        "Compression Type:",
        ["Lossless Compression"]
    )
    
    algorithm = st.selectbox(
        "Algorithm:",
        ["Run-Length Encoding (RLE)", "Huffman Coding", "Golomb Coding", "LZW Coding", "Compare All"]
    )
    
    if algorithm == "Golomb Coding":
        m_value = st.number_input("Parameter m:", min_value=2, max_value=256, value=4)
    
    if algorithm == "Compare All":
        comparison_type = st.radio(
            "Compare by:",
            ["Compression Ratio", "Processing Speed", "Memory Usage"]
        )

# Algorithm Implementations
def run_length_encoding(text: str) -> bytes:
    if not text:
        return b""

    output = bytearray()
    count = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1] and count < 255:
            count += 1
        else:
            ch_bytes = text[i-1].encode("utf-8")
            output.append(count)
            output.append(len(ch_bytes))
            output.extend(ch_bytes)
            count = 1

    ch_bytes = text[-1].encode("utf-8")
    output.append(count)
    output.append(len(ch_bytes))
    output.extend(ch_bytes)

    return bytes(output)

def run_length_decoding(data: bytes) -> str:
    if not data:
        return ""

    output = []
    i = 0

    while i < len(data):
        count = data[i]
        length = data[i + 1]
        i += 2

        ch_bytes = data[i:i + length]
        i += length

        ch = ch_bytes.decode("utf-8")
        output.append(ch * count)

    return "".join(output)

def build_huffman_tree(text):
    freq = Counter(text)
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    return heap[0]

def get_codes(tree):
    codes = {}
    for item in tree[1:]:
        codes[item[0]] = item[1]
    return codes

def huffman_encoding(data):
    text = data.decode('utf-8', errors='ignore')
    tree = build_huffman_tree(text)
    codes = get_codes(tree)
    
    encoded_bits = ''.join(codes[char] for char in text)
    
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = encoded_bits[i:i+8]
        encoded_bytes.append(int(byte.ljust(8, '0'), 2))
    
    return bytes(encoded_bytes), codes, tree

def huffman_decoding(encoded_data, codes):
    bit_string = ''.join(f'{byte:08b}' for byte in encoded_data)
    reversed_codes = {v: k for k, v in codes.items()}
    
    decoded = ''
    buffer = ''
    for bit in bit_string:
        buffer += bit
        if buffer in reversed_codes:
            decoded += reversed_codes[buffer]
            buffer = ''
    
    return decoded.encode('utf-8')

def golomb_encoding(data, m):
    encoded_bits = ""
    
    for byte in data:
        n = byte
        q = n // m
        r = n % m
        
        encoded_bits += "1" * q + "0"
        
        b = math.ceil(math.log2(m))
        if r < 2**b - m:
            encoded_bits += format(r, f'0{b-1}b')
        else:
            encoded_bits += format(r + 2**b - m, f'0{b}b')
    
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = encoded_bits[i:i+8]
        encoded_bytes.append(int(byte.ljust(8, '0'), 2))
    
    return bytes(encoded_bytes)

def LZW_encoding(text):
    dictionary = {chr(i): i for i in range(256)}
    dict_entries = []
    current_c = ""
    next_code = 256
    result = []

    for next_c in text:
        combine = current_c + next_c
        if combine in dictionary:
            current_c = combine
        else:
            result.append(dictionary[current_c])
            dictionary[combine] = next_code
            dict_entries.append((next_code, combine))
            next_code += 1
            current_c = next_c

    if current_c:
        result.append(dictionary[current_c])

    return result, dict_entries

def LZW_decoding(codeword):
    dictionary = {}
    dict_entries = []
    for i in range(256):
        dictionary[i] = chr(i)
        dict_entries.append((i, chr(i)))
    
    text = dictionary[codeword[0]]
    next_code = 256
    
    for i in range(1, len(codeword)):
        current_code = codeword[i]
        
        if current_code in dictionary:
            current_string = dictionary[current_code]
        else:
            current_string = dictionary[codeword[i-1]] + dictionary[codeword[i-1]][0]
        
        text += current_string
        dictionary[next_code] = dictionary[codeword[i-1]] + current_string[0]
        dict_entries.append((next_code, dictionary[next_code]))
        next_code += 1
    
    return text, dict_entries

# Calculate entropy
def calculate_entropy(data):
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    entropy = 0
    total = len(data)
    for count in freq.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy

# Main area
st.header("File Upload")
uploaded_file = st.file_uploader(
    "Upload a text file to compress:",
    type=['txt', 'csv', 'json', 'xml', 'py', 'html']
)

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name
    
    # Display file info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", file_name)
    with col2:
        st.metric("File Size", f"{len(file_bytes):,} bytes")
    with col3:
        entropy = calculate_entropy(file_bytes)
        st.metric("Entropy", f"{entropy:.3f} bits/byte")
    
    # File preview
    with st.expander("File Preview (First 500 characters)"):
        text_data = file_bytes.decode('utf-8', errors='ignore')
        st.text_area("", text_data[:500], height=150, disabled=True)
    
    st.markdown("---")
    
    if algorithm != "Compare All":
        # Single algorithm mode
        st.header(f"Algorithm: {algorithm}")
        
        # Algorithm description
        with st.expander("How This Algorithm Works"):
            if algorithm == "Run-Length Encoding (RLE)":
                st.markdown("""
                **RLE Algorithm:**
                - Scans the data for consecutive repeated characters
                - Replaces runs of identical characters with [count, character] pairs
                - Best for data with many consecutive repeats (e.g., simple graphics)
                - Simple and fast but poor for varied data
                """)
            elif algorithm == "Huffman Coding":
                st.markdown("""
                **Huffman Coding:**
                - Creates optimal prefix codes based on character frequencies
                - Builds a binary tree with frequent characters near the root
                - Uses variable-length codes (shorter for frequent characters)
                - Optimal for stationary sources
                """)
            elif algorithm == "Golomb Coding":
                st.markdown("""
                **Golomb Coding:**
                - Parameterized coding for geometric distributions
                - Uses parameter 'm' to optimize for specific data patterns
                - Encodes numbers using quotient (unary) and remainder (binary)
                - Excellent for run-length encoded data or geometric distributions
                """)
            elif algorithm == "LZW Coding":
                st.markdown("""
                **LZW Algorithm:**
                - Dictionary-based compression
                - Builds dictionary of strings dynamically during encoding
                - Replaces repeated phrases with dictionary indices
                - Widely used in GIF, TIFF, and Unix compress
                """)
        
        # Compression button
        if st.button(f"Run {algorithm} Compression", type="primary"):
            start_time = time.time()
            
            try:
                if algorithm == "Run-Length Encoding (RLE)":
                    compressed_data = run_length_encoding(text_data)
                    
                    # Show RLE example
                    with st.expander("RLE Compression Example"):
                        st.write("**First few runs:**")
                        sample_text = text_data[:50]
                        runs = []
                        i = 0
                        while i < len(sample_text) and len(runs) < 5:
                            count = 1
                            while i + count < len(sample_text) and sample_text[i] == sample_text[i + count]:
                                count += 1
                            runs.append(f"'{sample_text[i]}' × {count}")
                            i += count
                        st.write(" → ".join(runs))
                    
                elif algorithm == "Huffman Coding":
                    compressed_data, codes, tree = huffman_encoding(file_bytes)
                    
                    # Show Huffman codes
                    with st.expander("Huffman Codes (Top 10 characters)"):
                        # Get top 10 frequent characters
                        char_freq = Counter(text_data)
                        top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        table_data = []
                        for char, freq in top_chars:
                            code = codes.get(char, "N/A")
                            table_data.append({
                                "Character": repr(char)[1:-1],
                                "Frequency": freq,
                                "Code": code,
                                "Code Length": len(code)
                            })
                        
                        df = pd.DataFrame(table_data)
                        st.table(df)
                    
                elif algorithm == "Golomb Coding":
                    compressed_data = golomb_encoding(file_bytes, m_value)
                    
                    # Show Golomb encoding example
                    with st.expander("Golomb Encoding Example (First 5 bytes)"):
                        example_bytes = file_bytes[:5]
                        table_data = []
                        for byte in example_bytes:
                            n = byte
                            q = n // m_value
                            r = n % m_value
                            b = math.ceil(math.log2(m_value))
                            
                            unary_code = "1" * q + "0"
                            if r < 2**b - m_value:
                                binary_code = format(r, f'0{b-1}b')
                            else:
                                binary_code = format(r + 2**b - m_value, f'0{b}b')
                            
                            table_data.append({
                                "Byte": n,
                                "Binary": format(n, '08b'),
                                "Quotient (q)": q,
                                "Remainder (r)": r,
                                "Unary Code": unary_code,
                                "Binary Code": binary_code,
                                "Total Code": unary_code + binary_code
                            })
                        
                        df = pd.DataFrame(table_data)
                        st.table(df)
                    
                elif algorithm == "LZW Coding":
                    compressed_data, dict_entries = LZW_encoding(text_data)
                    
                    # Show LZW dictionary growth
                    with st.expander("LZW Dictionary (First 10 new entries)"):
                        table_data = []
                        for i, (code, string) in enumerate(dict_entries[:10]):
                            table_data.append({
                                "Code": code,
                                "String": repr(string)[1:-1]
                            })
                        
                        df = pd.DataFrame(table_data)
                        st.table(df)
                
                compression_time = time.time() - start_time
                
                # Calculate sizes
                original_size = len(file_bytes)
                if isinstance(compressed_data, list):
                    compressed_size = len(compressed_data) * 4
                elif isinstance(compressed_data, str):
                    compressed_size = len(compressed_data.encode('utf-8'))
                else:
                    compressed_size = len(compressed_data)
                
                # Display results
                st.markdown("### Compression Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original", f"{original_size:,} B")
                with col2:
                    st.metric("Compressed", f"{compressed_size:,} B")
                with col3:
                    ratio = original_size / compressed_size if compressed_size > 0 else 0
                    st.metric("Ratio", f"{ratio:.2f}:1")
                with col4:
                    st.metric("Time", f"{compression_time:.3f} s")
                
                # Visualizations
                st.markdown("### Visualizations")
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    # Size comparison
                    fig = go.Figure(data=[
                        go.Bar(name='Original', x=['Size'], y=[original_size], marker_color='blue'),
                        go.Bar(name='Compressed', x=['Size'], y=[compressed_size], marker_color='green')
                    ])
                    fig.update_layout(title='Size Comparison', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_viz2:
                    # Savings chart
                    savings = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0
                    fig = go.Figure(data=[
                        go.Indicator(
                            mode="gauge+number",
                            value=savings,
                            title="Space Saved",
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={'axis': {'range': [0, 100]},
                                  'bar': {'color': "green"},
                                  'steps': [
                                      {'range': [0, 50], 'color': "lightgray"},
                                      {'range': [50, 75], 'color': "gray"},
                                      {'range': [75, 100], 'color': "darkgray"}]}
                        )
                    ])
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download section
                st.markdown("### Download")
                
                if isinstance(compressed_data, list):
                    bytes_data = struct.pack(f'{len(compressed_data)}I', *compressed_data)
                elif isinstance(compressed_data, str):
                    bytes_data = compressed_data.encode('utf-8')
                else:
                    bytes_data = compressed_data
                
                b64 = base64.b64encode(bytes_data).decode()
                download_filename = f"{algorithm.replace(' ', '_')}_{file_name}.bin"
                download_link = f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}">Download Compressed File</a>'
                st.markdown(download_link, unsafe_allow_html=True)
                
                # Algorithm Recommendation
                st.markdown("### Algorithm Assessment")
                
                if ratio > 3:
                    assessment = "Excellent compression for this file type"
                elif ratio > 1.5:
                    assessment = "Good compression"
                elif ratio > 1:
                    assessment = "Mild compression"
                else:
                    assessment = "Poor compression - file expanded"
                
                st.info(f"**{algorithm}**: {assessment}")
                
                # Show what files work best
                if algorithm == "RLE":
                    st.caption("Best for: Simple text, images with large uniform areas, monochrome images")
                elif algorithm == "Huffman":
                    st.caption("Best for: General text files, documents with varied character frequencies")
                elif algorithm == "Golomb":
                    st.caption("Best for: Numerical data, geometric distributions, run-length encoded data")
                elif algorithm == "LZW":
                    st.caption("Best for: Text with repeated phrases, source code, natural language")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:
        # Compare All mode
        st.header("Algorithm Comparison")
        
        if st.button("Run All Algorithms", type="primary"):
            results = []
            
            algorithms_to_test = ["RLE", "Huffman", "Golomb", "LZW"]
            
            progress_bar = st.progress(0)
            
            for idx, algo in enumerate(algorithms_to_test):
                progress_bar.progress((idx + 1) / len(algorithms_to_test))
                
                start_time = time.time()
                
                try:
                    if algo == "RLE":
                        compressed = run_length_encoding(text_data)
                    elif algo == "Huffman":
                        compressed, _, _ = huffman_encoding(file_bytes)
                    elif algo == "Golomb":
                        compressed = golomb_encoding(file_bytes, m_value)
                    elif algo == "LZW":
                        compressed, _ = LZW_encoding(text_data)
                    
                    comp_time = time.time() - start_time
                    
                    # Calculate size
                    if isinstance(compressed, list):
                        comp_size = len(compressed) * 4
                    elif isinstance(compressed, str):
                        comp_size = len(compressed.encode('utf-8'))
                    else:
                        comp_size = len(compressed)
                    
                    ratio = len(file_bytes) / comp_size if comp_size > 0 else 0
                    
                    results.append({
                        "Algorithm": algo,
                        "Time (s)": round(comp_time, 3),
                        "Size (B)": comp_size,
                        "Ratio": round(ratio, 2)
                    })
                    
                except Exception as e:
                    st.warning(f"{algo} failed: {str(e)}")
            
            # Display comparison results
            if results:
                df = pd.DataFrame(results)
                
                st.markdown("### Comparison Results")
                st.table(df)
                
                # Comparison chart
                fig = go.Figure()
                
                if comparison_type == "Compression Ratio":
                    fig.add_trace(go.Bar(x=df["Algorithm"], y=df["Ratio"], name="Ratio"))
                    fig.update_layout(title="Compression Ratio Comparison", yaxis_title="Ratio")
                elif comparison_type == "Processing Speed":
                    fig.add_trace(go.Bar(x=df["Algorithm"], y=df["Time (s)"], name="Time"))
                    fig.update_layout(title="Processing Speed Comparison", yaxis_title="Time (seconds)")
                else:
                    fig.add_trace(go.Bar(x=df["Algorithm"], y=df["Size (B)"], name="Size"))
                    fig.update_layout(title="Compressed Size Comparison", yaxis_title="Size (bytes)")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Find best algorithm
                if comparison_type == "Compression Ratio":
                    best = max(results, key=lambda x: x["Ratio"])
                    st.success(f"**Best for compression ratio:** {best['Algorithm']} ({best['Ratio']}:1 ratio)")
                elif comparison_type == "Processing Speed":
                    best = min(results, key=lambda x: x["Time (s)"])
                    st.success(f"**Fastest algorithm:** {best['Algorithm']} ({best['Time (s)']} seconds)")
                else:
                    best = min(results, key=lambda x: x["Size (B)"])
                    st.success(f"**Smallest output:** {best['Algorithm']} ({best['Size (B)']:,} bytes)")
            else:
                st.error("All algorithms failed to compress the file")

# Information section
st.markdown("---")
with st.expander("About This Application"):
    st.markdown("""
    **Data Compression Algorithms Implemented:**
    
    1. **Run-Length Encoding (RLE)** - Simple, fast, best for repetitive data
    2. **Huffman Coding** - Optimal prefix codes, good for text compression
    3. **Golomb Coding** - Parameterized, excellent for specific distributions
    4. **LZW (Lempel-Ziv-Welch)** - Dictionary-based, handles repeated phrases well
    
    **Features:**
    - Compare algorithm performance
    - Visualize compression results
    - See algorithm internals
    - Download compressed files
    - Get algorithm recommendations
    """)

# Footer
st.markdown("---")
st.caption("Data Compression Project | All algorithms implemented from scratch")