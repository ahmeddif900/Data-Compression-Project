import heapq
from collections import Counter

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
    """Returns: (compressed_bytes, codes, tree, encoded_bits, padding)"""
    text = data.decode('utf-8', errors='ignore')
    tree = build_huffman_tree(text)
    codes = get_codes(tree)
    
    encoded_bits = ''.join(codes[char] for char in text)
    
    # Calculate padding needed
    padding = 8 - (len(encoded_bits) % 8)
    if padding == 8:
        padding = 0
    
    # Add padding bits
    padded_bits = encoded_bits + ('0' * padding)
    
    encoded_bytes = bytearray()
    for i in range(0, len(padded_bits), 8):
        byte = padded_bits[i:i+8]
        encoded_bytes.append(int(byte, 2))
    
    return bytes(encoded_bytes), codes, tree, encoded_bits, padding

def huffman_decoding(encoded_data, codes, padding=0):
    """Decode Huffman encoded data"""
    # Convert bytes to bit string
    bit_string = ''.join(f'{byte:08b}' for byte in encoded_data)
    
    # Remove padding bits
    if padding > 0:
        bit_string = bit_string[:-padding]
    
    # Reverse codes for decoding
    reversed_codes = {v: k for k, v in codes.items()}
    
    decoded = ''
    buffer = ''
    
    for bit in bit_string:
        buffer += bit
        if buffer in reversed_codes:
            decoded += reversed_codes[buffer]
            buffer = ''
    
    return decoded.encode('utf-8')