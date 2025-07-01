import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encode(text):
    """
    Huffman encoding for data compression
    """
    # Build frequency table
    frequency = Counter(text)
    heap = []
    
    # Create nodes for each character
    for char, freq in frequency.items():
        node = HuffmanNode(char, freq)
        heapq.heappush(heap, node)
    
    # Handle special case with only one unique character
    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = HuffmanNode(None, node.freq)
        root.left = node
        return root
    
    # Build the tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        internal = HuffmanNode(None, left.freq + right.freq)
        internal.left = left
        internal.right = right
        heapq.heappush(heap, internal)
    
    # Generate codes from the tree
    root = heap[0]
    codes = {}
    
    def traverse(node, code):
        if node:
            if node.char:
                codes[node.char] = code
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')
    
    traverse(root, '')
    
    # Encode the text
    encoded_text = ''.join(codes[char] for char in text)
    
    print(f"Original text length: {len(text) * 8} bits")
    print(f"Compressed length: {len(encoded_text)} bits")
    print(f"Compression ratio: {len(encoded_text) / (len(text) * 8) * 100:.2f}%")
    
    return encoded_text, codes

def huffman_decode(encoded_text, codes):
    """
    Huffman decoding
    """
    reverse_codes = {code: char for char, code in codes.items()}
    decoded_text = []
    current_code = ""
    
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text.append(reverse_codes[current_code])
            current_code = ""
            
    return ''.join(decoded_text) 