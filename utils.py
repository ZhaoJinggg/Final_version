import zlib
import numpy as np
from PIL import Image

def calculate_adler32(data):
    """
    Calculate Adler-32 checksum
    """
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        # Add padding to ensure data length is a multiple of 8
        padding_length = (8 - len(data) % 8) % 8
        padded_data = data + '0' * padding_length
        
        byte_blocks = []
        for i in range(0, len(padded_data), 8):
            byte_blocks.append(int(padded_data[i:i+8], 2).to_bytes(1, 'big'))
        data_bytes = b''.join(byte_blocks)
    
    checksum = zlib.adler32(data_bytes) & 0xffffffff
    binary_checksum = format(checksum, '032b')
    print(f"Data length: {len(data)}, Checksum: {binary_checksum[:8]}...{binary_checksum[-8:]}")
    return binary_checksum

def append_checksum(bitstream):
    """
    Append checksum to bitstream
    """
    checksum = calculate_adler32(bitstream)
    return bitstream + checksum

def verify_checksum(bitstream):
    """
    Verify checksum
    """
    if len(bitstream) <= 32:
        print(f"Error: Bitstream too short ({len(bitstream)} bits) for checksum verification")
        return False, ""
    
    data = bitstream[:-32]
    received_checksum = bitstream[-32:]
    calculated_checksum = calculate_adler32(data)
    
    is_valid = (received_checksum == calculated_checksum)
    
    if not is_valid:
        print(f"Checksum verification failed")
        print(f"Received checksum: {received_checksum[:8]}...{received_checksum[-8:]}")
        print(f"Calculated checksum: {calculated_checksum[:8]}...{calculated_checksum[-8:]}")
    else:
        print(f"Checksum verification successful")
        
    return is_valid, data

def image_to_matrix(image_path):
    """
    Convert image to RGB digital matrix
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        rgb_matrix = np.array(img)
        height, width, channels = rgb_matrix.shape
        print(f"Image size: {width}x{height}")
        print(f"Channels: {channels}")
        print(f"Matrix shape: {rgb_matrix.shape}")
        return rgb_matrix
    except Exception as e:
        print(f"Error converting image: {e}")
        return None

def save_image(image_matrix, output_path):
    """
    Save image matrix to file
    """
    try:
        stego_pil = Image.fromarray(image_matrix.astype(np.uint8))
        stego_pil.save(output_path)
        print(f"Image saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False 