def run_length_encoding(text: str) -> bytes:
    if not text:
        return b""

    output = bytearray()
    count = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1] and count < 255:
            count += 1
        else:
            # encode character to UTF-8 bytes
            ch_bytes = text[i-1].encode("utf-8")

            output.append(count)              # run length
            output.append(len(ch_bytes))      # number of UTF-8 bytes
            output.extend(ch_bytes)           # raw bytes of character

            count = 1

    # last run
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
        count = data[i]        # repetition count
        length = data[i + 1]   # number of UTF-8 bytes
        i += 2

        # read exactly 'length' bytes for the character
        ch_bytes = data[i:i + length]
        i += length

        ch = ch_bytes.decode("utf-8")
        output.append(ch * count)

    return "".join(output)