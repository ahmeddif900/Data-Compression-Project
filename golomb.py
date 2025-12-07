# import math
# def golomb_encode_number(n, m):
#     q = n // m
#     r = n % m
#     unary = "1" * q + "0"
#     b = math.log2(m)
#     if b.is_integer():
#         r_bin = format(r, f"0{int(b)}b")
#     else:
#         b = math.ceil(b)
#         x = 2**b - m
#         if r < x:
#             r_bin = format(r, f"0{b-1}b")
#         else:
#             r_bin = format(r + x, f"0{b}b")
#     return unary + r_bin
# def golomb_encoding(data, m):
#     """Encode bytes using Golomb coding"""
#     if not data:
#         return b''
    
#     encoded_bits = ""
    
#     # Encode each byte
#     for byte in data:
#         encoded_bits += golomb_encode_number(byte, m)
    
#     # Convert bit string to bytes
#     encoded_bytes = bytearray()
#     for i in range(0, len(encoded_bits), 8):
#         byte_bits = encoded_bits[i:i+8]
#         if len(byte_bits) < 8:
#             byte_bits = byte_bits.ljust(8, '0')
#         encoded_bytes.append(int(byte_bits, 2))
    
#     return bytes(encoded_bytes)
import math
import struct

def golomb_encode_fast(data, m):
    """Fast Golomb encoding using integer operations"""
    if not data:
        return b''
    
    # Pre-calculate values
    b = math.ceil(math.log2(m))
    threshold = 2**b - m
    
    encoded_bits = []
    current_byte = 0
    bit_position = 7  # Start from most significant bit
    
    for byte in data:
        n = byte
        q = n // m
        r = n % m
        
        # Encode quotient in unary (q ones followed by zero)
        for _ in range(q):
            current_byte |= (1 << bit_position)
            bit_position -= 1
            if bit_position < 0:
                encoded_bits.append(current_byte)
                current_byte = 0
                bit_position = 7
        
        # Add the zero after ones
        current_byte &= ~(1 << bit_position)  # Ensure bit is 0
        bit_position -= 1
        if bit_position < 0:
            encoded_bits.append(current_byte)
            current_byte = 0
            bit_position = 7
        
        # Encode remainder
        if r < threshold:
            bits_needed = b - 1
            remainder = r
        else:
            bits_needed = b
            remainder = r + threshold
        
        # Encode remainder bits
        for i in range(bits_needed - 1, -1, -1):
            if remainder & (1 << i):
                current_byte |= (1 << bit_position)
            bit_position -= 1
            if bit_position < 0:
                encoded_bits.append(current_byte)
                current_byte = 0
                bit_position = 7
    
    # Add final byte if there are remaining bits
    if bit_position != 7:
        encoded_bits.append(current_byte)
    
    return bytes(encoded_bits)

# Keep the original for reference
def golomb_encoding_slow(data, m):
    """Original slow version for comparison"""
    if not data:
        return b''
    
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
        byte_bits = encoded_bits[i:i+8]
        if len(byte_bits) < 8:
            byte_bits = byte_bits.ljust(8, '0')
        encoded_bytes.append(int(byte_bits, 2))
    
    return bytes(encoded_bytes)

# Use fast version by default
golomb_encoding = golomb_encode_fast