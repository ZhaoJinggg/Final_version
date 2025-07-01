import numpy as np
import random
from huffman import huffman_encode, huffman_decode
from utils import append_checksum, verify_checksum, image_to_matrix, save_image
from evaluation import evaluate_steganography, calculate_custom_max_capacity

class CustomSteganographySystem:
    def __init__(self, block_size=8, sub_block_size=1, segment_length=2, lfsr_seed=0x1234, variance_threshold=2000):
        self.block_size = block_size
        self.sub_block_size = sub_block_size
        self.segment_length = segment_length
        self.lfsr_seed = lfsr_seed
        self.variance_threshold = variance_threshold
        self.perturbation_mapping = self.create_perturbation_mapping()
        
        # Set fixed random seeds for consistent results
        np.random.seed(self.lfsr_seed)
        random.seed(self.lfsr_seed)
    
    def create_perturbation_mapping(self):
        """
        Creates mapping for 2-bit segments to pixel value adjustments
        """
        perturbations = {
            '00': 0,    # No change
            '01': 1,    # +1 to channel  
            '10': -1,   # -1 to channel
            '11': 2     # +1 to channel
        }
        return perturbations
    
    # LFSR implementation from original code
    class LFSR:
        def __init__(self, seed, taps):
            self.state = seed & 0xFFFF
            self.taps = taps
        def next(self):
            result = self.state
            feedback = 0
            for tap in self.taps:
                feedback ^= (self.state >> tap) & 1
            self.state = ((self.state >> 1) | (feedback << 15)) & 0xFFFF
            return result
        def generate_sequence(self, length):
            return [self.next() for _ in range(length)]
    
    def divide_into_blocks(self, rgb_matrix, block_size=8):
        """
        Divide the image into blocks of specified size
        """
        height, width, _ = rgb_matrix.shape
        blocks = []
        block_positions = {}
        blocks_h = height // block_size
        blocks_w = width // block_size
        block_index = 0
        for i in range(blocks_h):
            for j in range(blocks_w):
                h_start = i * block_size
                h_end = (i + 1) * block_size
                w_start = j * block_size
                w_end = (j + 1) * block_size
                block = rgb_matrix[h_start:h_end, w_start:w_end, :]
                blocks.append(block)
                block_positions[block_index] = (i, j)
                block_index += 1
        print(f"Image divided into {len(blocks)} blocks of size {block_size}x{block_size}")
        print(f"Block distribution: {blocks_h} rows x {blocks_w} columns")
        return blocks, block_positions, blocks_h, blocks_w
    
    def analyze_blocks(self, blocks):
        """
        Analyze blocks to calculate statistical properties
        """
        block_stats = {
            "variances": [],
            "means": [],
            "std_devs": [],
        }
        for block in blocks:
            variance = np.var(block)
            mean = np.mean(block)
            std_dev = np.std(block)
            block_stats["variances"].append(variance)
            block_stats["means"].append(mean)
            block_stats["std_devs"].append(std_dev)
        block_stats["avg_variance"] = np.mean(block_stats["variances"])
        block_stats["min_variance"] = np.min(block_stats["variances"])
        block_stats["max_variance"] = np.max(block_stats["variances"])
        print(f"Block analysis completed:")
        print(f"  Average variance: {block_stats['avg_variance']:.2f}")
        print(f"  Minimum variance: {block_stats['min_variance']:.2f}")
        print(f"  Maximum variance: {block_stats['max_variance']:.2f}")
        return block_stats
    
    def find_smooth_blocks(self, rgb_matrix, variance_threshold, block_size=8):
        """
        Find smooth 8x8 blocks in the image using an absolute variance threshold
        """
        blocks, block_positions, blocks_h, blocks_w = self.divide_into_blocks(rgb_matrix, block_size)
        block_stats = self.analyze_blocks(blocks)
        
        smooth_indices = []
        for i, variance in enumerate(block_stats["variances"]):
            if variance < variance_threshold:
                smooth_indices.append(i)
        
        print(f"Found {len(smooth_indices)} smooth blocks (variance < {variance_threshold:.2f})")
        print(f"Smooth blocks are {len(smooth_indices) / len(block_stats['variances']) * 100:.2f}% of total blocks")
        return smooth_indices, blocks, block_positions, blocks_h, blocks_w, block_stats
    
    def select_candidate_blocks(self, smooth_blocks, seed, num_candidates=10):
        """
        Deterministically select candidate blocks using LFSR
        """
        if len(smooth_blocks) < num_candidates:
            print(f"Warning: Not enough smooth blocks. Required: {num_candidates}, Available: {len(smooth_blocks)}")
            num_candidates = len(smooth_blocks)
        
        # Set fixed seed for consistency
        lfsr = self.LFSR(seed, [0, 2, 3, 5])
        
        # Generate enough random numbers to select candidate blocks
        random_sequence = lfsr.generate_sequence(num_candidates * 2)
        
        candidate_blocks = []
        used_indices = set()
        
        for rand_val in random_sequence:
            idx = rand_val % len(smooth_blocks)
            block_idx = smooth_blocks[idx]
            
            if block_idx not in candidate_blocks and idx not in used_indices:
                candidate_blocks.append(block_idx)
                used_indices.add(idx)
                
            if len(candidate_blocks) == num_candidates:
                break
                
        print(f"Selected {len(candidate_blocks)} candidate blocks")
        return candidate_blocks
    
    def create_sub_blocks_mapping(self, block_idx, rgb_matrix, block_positions, block_size=8, sub_block_size=1):
        """
        Create 8×8 grid of 1×1 pixels within an 8×8 smooth block
        """
        # Get coordinates of the 8×8 block
        block_row, block_col = block_positions[block_idx]
        h_start = block_row * block_size
        w_start = block_col * block_size
        
        # Create mapping for individual pixels (8×8 grid, each pixel is 1×1)
        sub_blocks = {}
        sub_idx = 0
        
        for i in range(8):  # rows (0-7)
            for j in range(8):  # columns (0-7)
                pixel_h = h_start + i
                pixel_w = w_start + j
                
                # Save pixel coordinates
                sub_blocks[sub_idx] = {
                    'position': (i, j),  # relative position (0-7, 0-7)
                    'coordinates': (pixel_h, pixel_w),  # actual coordinates for single pixel
                    'global_position': (block_row, block_col, i, j)  # global position
                }
                sub_idx += 1
        
        return sub_blocks
    
    def get_adjacent_pixel_index(self, mother_idx, total_pixels=64):
        """
        Get adjacent pixel index for the child pixel.
        Priority: right neighbor, then bottom neighbor
        """
        # Convert 1D index to 2D position (8x8 grid)
        mother_row = mother_idx // 8
        mother_col = mother_idx % 8
        
        # Try right neighbor first
        if mother_col + 1 < 8:
            child_row = mother_row
            child_col = mother_col + 1
        # If no right neighbor, try bottom neighbor
        elif mother_row + 1 < 8:
            child_row = mother_row + 1
            child_col = mother_col
        else:
            # Edge case: use left neighbor or top neighbor
            if mother_col > 0:
                child_row = mother_row
                child_col = mother_col - 1
            else:
                child_row = mother_row - 1
                child_col = mother_col
        
        # Convert back to 1D index
        child_idx = child_row * 8 + child_col
        return child_idx
    
    def select_mother_child_pairs(self, candidate_blocks, rgb_matrix, block_positions, block_size=8, sub_block_size=1, lfsr_seed=0x1234):
        """
        Select mother-child pairs for each candidate smooth block
        Mother pixel: 1×1 pixel selected using LFSR
        Child pixel: Adjacent 1×1 pixel next to the mother pixel
        """
        np.random.seed(lfsr_seed)  # Set fixed seed
        lfsr = self.LFSR(lfsr_seed, [0, 2, 3, 5])
        
        all_pairs = []
        
        for block_idx in candidate_blocks:
            # Create pixel mapping for current smooth block
            sub_blocks = self.create_sub_blocks_mapping(block_idx, rgb_matrix, block_positions, block_size, sub_block_size)
            
            # Use LFSR to select mother pixel index
            mother_pixel_idx = lfsr.next() % 64  # 64 pixels (8×8)
            
            # Get adjacent pixel index for child
            child_pixel_idx = self.get_adjacent_pixel_index(mother_pixel_idx)
            
            # Record mother and child pixel information
            mother_info = sub_blocks[mother_pixel_idx]
            child_info = sub_blocks[child_pixel_idx]
            
            pair_info = {
                'block_idx': block_idx,  # smooth block index
                'mother_pixel_idx': mother_pixel_idx,  # mother pixel index in smooth block
                'child_pixel_idx': child_pixel_idx,    # child pixel index in smooth block
                'mother_position': mother_info['position'],  # mother pixel position (i,j)
                'child_position': child_info['position'],    # child pixel position (i,j)
                'mother_coordinates': mother_info['coordinates'],  # mother pixel coordinates
                'child_coordinates': child_info['coordinates'],    # child pixel coordinates
                'mother_global': mother_info['global_position'],   # mother pixel global position
                'child_global': child_info['global_position']      # child pixel global position
            }
            
            all_pairs.append(pair_info)
        
        print(f"Created {len(all_pairs)} mother-child pixel pairs within smooth blocks")
        return all_pairs
    
    def segment_bitstream(self, bitstream, segment_length=2):
        """
        Segment the bitstream into fixed-length chunks
        """
        segments = []
        
        # Ensure bitstream length is a multiple of segment_length
        padding_length = (segment_length - len(bitstream) % segment_length) % segment_length
        padded_bitstream = bitstream + '0' * padding_length
        
        # Segment the bitstream
        for i in range(0, len(padded_bitstream), segment_length):
            segment = padded_bitstream[i:i+segment_length]
            segments.append(segment)
        
        print(f"Segmented bitstream into {len(segments)} segments of {segment_length} bits each")
        print(f"Padded {padding_length} bits to ensure proper segmentation")
        
        return segments
    
    def apply_perturbation(self, rgb_matrix, mother_child_pairs, segments, perturbation_mapping):
        """
        Apply perturbations to embed data using mother-child pixel pairs
        """
        stego_img = np.copy(rgb_matrix)
        
        # Each pair can encode 3 segments (one per RGB channel)
        segments_per_pair = 3
        total_capacity = len(mother_child_pairs) * segments_per_pair
        
        print(f"Total embedding capacity: {total_capacity * 2} bits ({total_capacity} segments)")
        print(f"Actual data to embed: {len(segments)} segments ({len(segments) * 2} bits)")
        
        # Handle data size vs. capacity
        if len(segments) > total_capacity:
            segments = segments[:total_capacity]
            print(f"Warning: Truncated data to fit capacity ({total_capacity} segments)")
        elif len(segments) < total_capacity:
            padding_segments = ['00'] * (total_capacity - len(segments))
            segments = segments + padding_segments
            print(f"Added {len(padding_segments)} padding segments")
        
        # Embed segments into mother-child pairs
        embedded_segments = {}
        segment_index = 0
        
        for i, pair_info in enumerate(mother_child_pairs):
            # Get mother and child pixel coordinates
            m_h, m_w = pair_info['mother_coordinates']
            c_h, c_w = pair_info['child_coordinates']
            
            # Debug first few pixel pairs
            if i < 3:
                print(f"Embedding - Pixel pair {i}")
                print(f"Smooth block: {pair_info['block_idx']}")
                print(f"Mother pixel: {pair_info['mother_pixel_idx']} at position {pair_info['mother_position']}")
                print(f"Child pixel: {pair_info['child_pixel_idx']} at position {pair_info['child_position']}")
                print(f"Mother coordinates: ({m_h}, {m_w})")
                print(f"Child coordinates: ({c_h}, {c_w})")
            
            # Copy mother pixel to child pixel
            mother_pixel = rgb_matrix[m_h, m_w, :].copy()
            stego_img[c_h, c_w, :] = mother_pixel
            
            # Get next 3 segments (one for each channel)
            current_segments = []
            for j in range(segments_per_pair):
                if segment_index < len(segments):
                    current_segments.append(segments[segment_index])
                    segment_index += 1
                else:
                    current_segments.append('00')  # Default segment
            
            if i < 3:
                print(f"Segments to embed: {current_segments}")
            
            # Apply perturbations to child pixel (each channel separately)
            for c in range(3):
                segment = current_segments[c]
                perturbation = perturbation_mapping.get(segment, 0)
                
                # Perturb pixel value and keep within 0-255
                pixel_value = int(stego_img[c_h, c_w, c])
                perturbed_value = pixel_value + perturbation
                perturbed_value = max(0, min(255, perturbed_value))
                stego_img[c_h, c_w, c] = perturbed_value
            
            # Record embedded segments
            embedded_segments[i] = current_segments
        
        print(f"Embedded {segment_index} 2-bit segments into the image across {len(mother_child_pairs)} pixel pairs")
        
        return stego_img, embedded_segments, segment_index
    
    def extract_data(self, stego_img, mother_child_pairs, perturbation_mapping):
        """
        Extract data from stego image using mother-child pixel pairs
        """
        extracted_segments = []
        debug_info = []
        
        for i, pair_info in enumerate(mother_child_pairs):
            # Get mother and child pixel coordinates
            m_h, m_w = pair_info['mother_coordinates']
            c_h, c_w = pair_info['child_coordinates']
            
            # Get mother and child pixels
            mother_pixel = stego_img[m_h, m_w, :]
            child_pixel = stego_img[c_h, c_w, :]
            
            # Debug first few pixel pairs
            if i < 3:
                debug_info.append(f"Pixel pair {i}, Mother pos: {pair_info['mother_position']}, Child pos: {pair_info['child_position']}")
                debug_info.append(f"Mother pixel: {mother_pixel}")
                debug_info.append(f"Child pixel: {child_pixel}")
            
            # Extract segments from each channel
            channel_segments = []
            for c in range(3):  # RGB channels
                # Calculate pixel difference for this channel
                diff = int(child_pixel[c]) - int(mother_pixel[c])
                
                # Find segment closest to original perturbation value
                possible_diffs = [0, 1, -1, 2]  # Possible perturbation values
                possible_segments = ['00', '01', '10', '11']  # Corresponding segments
                
                closest_idx = np.argmin([abs(diff - d) for d in possible_diffs])
                segment = possible_segments[closest_idx]
                
                channel_segments.append(segment)
                
                if i < 5:
                    debug_info.append(f"Pixel pair {i}, Channel {c}: diff={diff}, segment={segment}")
            
            extracted_segments.extend(channel_segments)
        
        # Print debug info
        for info in debug_info:
            print(info)
            
        extracted_bitstream = ''.join(extracted_segments)
        print(f"Extracted {len(extracted_segments)} segments, total {len(extracted_bitstream)} bits")
        return extracted_bitstream, extracted_segments

    def embed(self, image_path, data, output_path=None):
        """
        Embed data using custom steganography method
        """
        print("\n======= Custom Steganography: Data Embedding Process =======")
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
        
        # 2. Find smooth blocks with fixed variance threshold
        smooth_blocks, blocks, block_positions, blocks_h, blocks_w, block_stats = self.find_smooth_blocks(
            rgb_matrix, self.variance_threshold, self.block_size
        )
        
        # 3. Select candidate blocks using LFSR
        num_candidates = min(len(smooth_blocks) // 2, 
                           self.original_with_checksum_length // (self.segment_length * 3) + 10)  # Add redundancy
        candidate_blocks = self.select_candidate_blocks(smooth_blocks, self.lfsr_seed, num_candidates)
        
        # 4. Create mother-child pairs
        self.mother_child_pairs = self.select_mother_child_pairs(
            candidate_blocks, 
            rgb_matrix, 
            block_positions, 
            self.block_size, 
            1,  # sub_block_size = 1 for 1x1 pixels
            self.lfsr_seed
        )
        
        # 5. Segment bitstream
        segments = self.segment_bitstream(bitstream_with_checksum, self.segment_length)
        
        # 6. Apply perturbations to embed data
        stego_img, self.embedded_segments, self.segments_used = self.apply_perturbation(
            rgb_matrix, 
            self.mother_child_pairs,
            segments, 
            self.perturbation_mapping
        )
        
        # Save stego image if path provided
        if output_path:
            save_image(stego_img, output_path)
        
        # Store information for extraction and evaluation
        self.original_matrix = rgb_matrix
        self.blocks = blocks
        self.block_positions = block_positions
        self.blocks_h = blocks_h
        self.blocks_w = blocks_w
        self.block_stats = block_stats
        self.smooth_blocks = smooth_blocks
        self.candidate_blocks = candidate_blocks
        self.huffman_codes = codes
        self.original_data = data
        self.encoded_data = encoded_data
        
        return stego_img
    
    def extract(self, stego_image_path=None, stego_img=None):
        """
        Extract data using custom steganography method
        """
        print("\n======= Custom Steganography: Data Extraction Process =======")
        if stego_img is None and stego_image_path is None:
            print("Error: Must provide either stego image path or stego image matrix")
            return None
            
        # Load stego image if path provided
        if stego_img is None:
            stego_img = image_to_matrix(stego_image_path)
            if stego_img is None:
                return None
        
        # Regenerate mother-child pairs if not already available
        if not hasattr(self, 'mother_child_pairs') or stego_image_path is not None:
            print("Regenerating mother-child pairs for extraction...")
            
            # Find smooth blocks with same variance threshold
            smooth_blocks, blocks, block_positions, blocks_h, blocks_w, _ = self.find_smooth_blocks(
                stego_img, self.variance_threshold, self.block_size
            )
            
            # Select same candidate blocks using same LFSR seed
            num_candidates = min(len(smooth_blocks) // 2, 30)  # Use a reasonably large value
            candidate_blocks = self.select_candidate_blocks(smooth_blocks, self.lfsr_seed, num_candidates)
            
            # Generate same mother-child pairs
            self.mother_child_pairs = self.select_mother_child_pairs(
                candidate_blocks,
                stego_img, 
                block_positions,
                self.block_size,
                1,  # sub_block_size = 1 for 1x1 pixels
                self.lfsr_seed
            )
        
        # Check if we have valid mother-child pairs
        if not self.mother_child_pairs:
            print("Error: No valid mother-child pairs found for extraction")
            return None
            
        # Extract data using mother-child pairs
        extracted_bitstream, extracted_segments = self.extract_data(
            stego_img, 
            self.mother_child_pairs, 
            self.perturbation_mapping
        )
        
        # Trim extracted data to match original length if known
        if hasattr(self, 'original_with_checksum_length'):
            print(f"Trimming extracted bitstream from {len(extracted_bitstream)} to {self.original_with_checksum_length} bits")
            extracted_bitstream = extracted_bitstream[:self.original_with_checksum_length]
        
        # Verify checksum
        is_valid, data_bitstream = verify_checksum(extracted_bitstream)
        if not is_valid:
            print("Error: Data integrity verification failed. The extracted checksum does not match.")
            return None
            
        # Decode data using Huffman codes
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
        Evaluate custom steganography performance
        """
        if not hasattr(self, 'original_matrix') or not hasattr(self, 'encoded_data'):
            print("Error: Must perform embedding process before evaluation")
            return None
            
        if stego_img is None:
            print("Warning: No stego image provided, cannot perform evaluation")
            return None
            
        # Calculate maximum capacity
        max_capacity = calculate_custom_max_capacity(self.original_matrix, self.variance_threshold, self.block_size)
        
        # Extract data from stego image
        extracted_bitstream, _ = self.extract_data(
            stego_img, 
            self.mother_child_pairs, 
            self.perturbation_mapping
        )
        
        # Trim to original length
        if hasattr(self, 'original_with_checksum_length'):
            extracted_bitstream = extracted_bitstream[:self.original_with_checksum_length]
        
        # Verify checksum and evaluate
        is_valid, data_bitstream = verify_checksum(extracted_bitstream)
        return evaluate_steganography(self.original_matrix, stego_img, self.encoded_data, data_bitstream, max_capacity) 