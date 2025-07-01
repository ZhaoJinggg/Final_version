import numpy as np
import random
from huffman import huffman_encode, huffman_decode
from utils import append_checksum, verify_checksum, image_to_matrix, save_image
from evaluation import evaluate_steganography, calculate_lsb_max_capacity

class LSBSteganographySystem:
    def __init__(self, bits_per_channel=1, lfsr_seed=0x1234):
        self.bits_per_channel = bits_per_channel
        self.lfsr_seed = lfsr_seed
        
        # Set fixed random seeds for consistent results
        np.random.seed(self.lfsr_seed)
        random.seed(self.lfsr_seed)
    
    def calculate_lsb_capacity(self, rgb_matrix, bits_per_channel=1):
        """
        Calculate embedding capacity of an image using LSB method
        """
        height, width, channels = rgb_matrix.shape
        return height * width * channels * bits_per_channel
    
    def embed_lsb(self, rgb_matrix, bitstream, bits_per_channel=1):
        """
        Embed data using standard LSB technique
        """
        print(f"\n======= LSB Data Embedding Process =======")
        stego_img = np.copy(rgb_matrix)
        height, width, channels = stego_img.shape
        
        # Calculate maximum capacity
        max_capacity = self.calculate_lsb_capacity(rgb_matrix, bits_per_channel)
        print(f"Maximum embedding capacity: {max_capacity} bits")
        print(f"Data size: {len(bitstream)} bits")
        
        if len(bitstream) > max_capacity:
            print(f"Warning: Data size exceeds image capacity. Truncating data.")
            bitstream = bitstream[:max_capacity]
        
        # Create a flat view of the image for easier traversal
        flat_image = stego_img.reshape(-1)
        
        # Create mask for clearing the LSBs before embedding
        # For example, if bits_per_channel=1, mask = 11111110 (254)
        # If bits_per_channel=2, mask = 11111100 (252)
        mask = 256 - (2 ** bits_per_channel)
        
        # Embed data
        bit_index = 0
        for pixel_index in range(len(flat_image)):
            if bit_index >= len(bitstream):
                break
                
            # Get next group of bits to embed
            end_index = min(bit_index + bits_per_channel, len(bitstream))
            bits_to_embed = bitstream[bit_index:end_index]
            
            # Pad with zeros if needed
            while len(bits_to_embed) < bits_per_channel:
                bits_to_embed += '0'
                
            # Convert bits to integer
            embed_value = int(bits_to_embed, 2)
            
            # Clear LSBs and embed data
            flat_image[pixel_index] = (flat_image[pixel_index] & mask) | embed_value
            
            bit_index += bits_per_channel
        
        print(f"Successfully embedded {bit_index} bits into the image")
        
        # Restore original shape
        stego_img = flat_image.reshape(height, width, channels)
        
        return stego_img, bit_index
    
    def extract_lsb(self, stego_img, data_length, bits_per_channel=1):
        """
        Extract data from LSB
        """
        print(f"\n======= LSB Data Extraction Process =======")
        height, width, channels = stego_img.shape
        
        # Check if extraction is possible
        max_capacity = self.calculate_lsb_capacity(stego_img, bits_per_channel)
        if data_length > max_capacity:
            print(f"Error: Requested data length exceeds image capacity")
            return None
        
        # Create a flat view of the image
        flat_image = stego_img.reshape(-1)
        
        # Create mask for extracting LSBs
        # For bits_per_channel=1: mask = 00000001 (1)
        # For bits_per_channel=2: mask = 00000011 (3)
        mask = (2 ** bits_per_channel) - 1
        
        # Extract data
        extracted_bits = []
        bits_extracted = 0
        
        for pixel_index in range(len(flat_image)):
            if bits_extracted >= data_length:
                break
                
            # Extract LSBs using mask
            pixel_value = flat_image[pixel_index]
            extracted_value = pixel_value & mask
            
            # Convert extracted value to binary and remove '0b' prefix
            bits = bin(extracted_value)[2:].zfill(bits_per_channel)
            
            # Add extracted bits, but don't exceed data_length
            remaining_bits = data_length - bits_extracted
            if remaining_bits >= bits_per_channel:
                extracted_bits.append(bits)
                bits_extracted += bits_per_channel
            else:
                extracted_bits.append(bits[:remaining_bits])
                bits_extracted += remaining_bits
        
        print(f"Successfully extracted {bits_extracted} bits from the image")
        
        # Join all bits into a single string
        extracted_bitstream = ''.join(extracted_bits)
        return extracted_bitstream
    
    def embed(self, image_path, data, output_path=None):
        """
        Embed data using LSB steganography method
        """
        print("\n======= LSB Data Embedding Process =======")
        rgb_matrix = image_to_matrix(image_path)
        if rgb_matrix is None:
            return None
            
        # 1. Data preprocessing
        encoded_data, codes = huffman_encode(data)
        bitstream_with_checksum = append_checksum(encoded_data)
        
        # Record original data length
        self.original_data_length = len(encoded_data)
        self.original_with_checksum_length = len(bitstream_with_checksum)
        print(f"Original data length: {self.original_data_length} bits")
        print(f"With checksum: {self.original_with_checksum_length} bits")
        
        # 2. Embed data using LSB method
        stego_img, bits_embedded = self.embed_lsb(rgb_matrix, bitstream_with_checksum, self.bits_per_channel)
        
        # Save stego image if path is provided
        if output_path:
            save_image(stego_img, output_path)
        
        # Store information for extraction and evaluation
        self.original_matrix = rgb_matrix
        self.huffman_codes = codes
        self.original_data = data
        self.encoded_data = encoded_data
        
        return stego_img
        
    def extract(self, stego_image_path=None, stego_img=None):
        """
        Extract data from an LSB stego image
        """
        print("\n======= LSB Data Extraction Process =======")
        if stego_img is None and stego_image_path is None:
            print("Error: Must provide either stego image path or stego image matrix")
            return None
            
        # Load stego image if path is provided
        if stego_img is None:
            stego_img = image_to_matrix(stego_image_path)
            if stego_img is None:
                return None
        
        # Check if we know the original data length
        if not hasattr(self, 'original_with_checksum_length'):
            print("Warning: Original data length unknown. Will extract maximum possible data.")
            extraction_length = self.calculate_lsb_capacity(stego_img, self.bits_per_channel)
        else:
            extraction_length = self.original_with_checksum_length
            
        # Extract data using LSB method
        extracted_bitstream = self.extract_lsb(stego_img, extraction_length, self.bits_per_channel)
        
        # Verify checksum
        is_valid, data_bitstream = verify_checksum(extracted_bitstream)
        if not is_valid:
            print("Error: Data integrity verification failed. The extracted checksum does not match.")
            return None
            
        # Use Huffman codes to decode data
        if hasattr(self, 'huffman_codes'):
            try:
                decoded_data = huffman_decode(data_bitstream, self.huffman_codes)
                print(f"Successfully extracted and decoded data: {len(decoded_data)} characters")
                return decoded_data
            except Exception as e:
                print(f"Error decoding data: {e}")
                return None
        else:
            print("Warning: No Huffman code table available, cannot decode data. Returning raw bitstream.")
            return data_bitstream
            
    def evaluate(self, stego_img=None):
        """
        Evaluate LSB steganography performance
        """
        if not hasattr(self, 'original_matrix') or not hasattr(self, 'encoded_data'):
            print("Error: Must perform embedding process before evaluation")
            return None
            
        if stego_img is None:
            print("Warning: No stego image provided, cannot perform evaluation")
            return None
            
        # Calculate maximum capacity
        max_capacity = calculate_lsb_max_capacity(self.original_matrix, self.bits_per_channel)
        
        # Extract data from stego image
        extraction_length = self.original_with_checksum_length
        extracted_bitstream = self.extract_lsb(stego_img, extraction_length, self.bits_per_channel)
        
        # Verify checksum and evaluate
        is_valid, data_bitstream = verify_checksum(extracted_bitstream)
        return evaluate_steganography(self.original_matrix, stego_img, self.encoded_data, data_bitstream, max_capacity) 