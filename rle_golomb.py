import math

def rle_golomb_encoding(text, m=4):
    """
    Two-stage compression:
    1. RLE to get run lengths
    2. Golomb to compress run lengths
    """
    if not text:
        return b''
    
    # Stage 1: Run-Length Encoding
    runs = []
    count = 1
    
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            runs.append((count, text[i-1]))
            count = 1
    
    # Add last run
    runs.append((count, text[-1]))
    
    # Store characters separately
    characters = [char for _, char in runs]
    run_lengths = [count for count, _ in runs]
    
    # Stage 2: Golomb encode run lengths
    encoded_bits = ""
    
    for run_length in run_lengths:
        n = run_length
        q = n // m
        r = n % m
        
        # Unary code for quotient
        encoded_bits += "1" * q + "0"
        
        # Binary code for remainder
        b = math.ceil(math.log2(m))
        if r < 2**b - m:
            encoded_bits += format(r, f'0{b-1}b')
        else:
            encoded_bits += format(r + 2**b - m, f'0{b}b')
    
    # Convert bit string to bytes for run lengths
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte_bits = encoded_bits[i:i+8]
        if len(byte_bits) < 8:
            byte_bits = byte_bits.ljust(8, '0')
        encoded_bytes.append(int(byte_bits, 2))
    
    # Combine: [m_value, num_runs, characters, encoded_run_lengths]
    result = bytearray()
    
    # 1. Store m value (1 byte)
    result.append(m)
    
    # 2. Store number of runs (4 bytes)
    result.extend(len(runs).to_bytes(4, 'big'))
    
    # 3. Store characters (as UTF-8)
    char_bytes = ''.join(characters).encode('utf-8')
    result.extend(len(char_bytes).to_bytes(4, 'big'))
    result.extend(char_bytes)
    
    # 4. Store Golomb-encoded run lengths
    result.extend(len(encoded_bytes).to_bytes(4, 'big'))
    result.extend(encoded_bytes)
    
    return bytes(result)

def rle_golomb_decoding(data):
    """Decode RLE+Golomb compressed data"""
    if not data:
        return ""
    
    data = bytearray(data)
    index = 0
    
    # 1. Read m value
    m = data[index]
    index += 1
    
    # 2. Read number of runs
    num_runs = int.from_bytes(data[index:index+4], 'big')
    index += 4
    
    # 3. Read characters
    char_len = int.from_bytes(data[index:index+4], 'big')
    index += 4
    
    characters = data[index:index+char_len].decode('utf-8')
    index += char_len
    
    # 4. Read Golomb-encoded run lengths
    encoded_len = int.from_bytes(data[index:index+4], 'big')
    index += 4
    
    encoded_bytes = data[index:index+encoded_len]
    
    # Decode Golomb to get run lengths
    encoded_bits = ''.join(f'{byte:08b}' for byte in encoded_bytes)
    
    run_lengths = []
    bit_index = 0
    b = math.ceil(math.log2(m))
    threshold = 2**b - m
    
    for _ in range(num_runs):
        # Decode unary part (count ones until zero)
        q = 0
        while bit_index < len(encoded_bits) and encoded_bits[bit_index] == '1':
            q += 1
            bit_index += 1
        
        # Skip the zero
        if bit_index >= len(encoded_bits):
            break
        bit_index += 1
        
        # Determine bits needed based on threshold comparison
        # We need to decode some bits first to know if r < threshold
        # For the first b-1 bits:
        if bit_index + (b-1) > len(encoded_bits):
            break
        
        # Read b-1 bits first
        temp_bits = encoded_bits[bit_index:bit_index + (b-1)]
        temp_val = int(temp_bits, 2)
        
        if temp_val < threshold:
            # r < threshold: we've read all needed bits
            r = temp_val
            bits_needed = b - 1
        else:
            # r >= threshold: need one more bit
            if bit_index + b > len(encoded_bits):
                break
            full_bits = encoded_bits[bit_index:bit_index + b]
            full_val = int(full_bits, 2)
            r = full_val - threshold
            bits_needed = b
        
        bit_index += bits_needed
        run_length = q * m + r
        run_lengths.append(run_length)
    
    # Reconstruct original text
    reconstructed = ""
    # Make sure we have enough run lengths
    min_len = min(len(run_lengths), len(characters))
    for i in range(min_len):
        reconstructed += characters[i] * run_lengths[i]
    
    return reconstructed