def LZW_encoding(text):
    # Initialize dictionary with single-character ASCII
    dictionary = {chr(i): i for i in range(256)}
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
            next_code += 1
            current_c = next_c

    if current_c:
        result.append(dictionary[current_c])

    return result
def LZW_decoding(codeword):
    # Initialize dictionary with ASCII characters (code -> string)
    dictionary = {}
    for i in range(256):
        dictionary[i] = chr(i)
    
    # Start with first code
    text = dictionary[codeword[0]]
    next_code = 256
    
    # Process remaining codes
    for i in range(1, len(codeword)):
        current_code = codeword[i]
        
        if current_code in dictionary:
            # Code exists in dictionary
            current_string = dictionary[current_code]
        else:
            # Special case: code not in dictionary (can happen in LZW)
            current_string =  dictionary[codeword[i-1]] +  dictionary[codeword[i-1]][0]
        
        # Add to output
        text += current_string
        
        # Add new entry to dictionary: previous string + first char of current string
        dictionary[next_code] = dictionary[codeword[i-1]] + current_string[0]
        next_code += 1
    
    return text,dictionary
