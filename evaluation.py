import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_msssim import ms_ssim
import sewar

def calculate_ms_ssim(img1, img2):
    """
    Calculate MS-SSIM value between two images
    """
    tensor1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    ms_ssim_val = ms_ssim(tensor1, tensor2, data_range=1.0)
    return ms_ssim_val.item()

def calculate_fsim(img1, img2):
    """
    Calculate FSIM value between two images using sewar module
    """
    try:
        if hasattr(sewar, 'fsim'):
            fsim_val = sewar.fsim(img1, img2)
            return fsim_val
        elif hasattr(sewar, 'fsim_color'):
            fsim_val = sewar.fsim_color(img1, img2)
            return fsim_val
        else:
            print("Warning: FSIM calculation not available in sewar module")
            return 0.0
    except Exception as e:
        print(f"Error calculating FSIM: {e}")
        return 0.0

def calculate_custom_max_capacity(rgb_matrix, variance_threshold, block_size=8):
    """
    Calculate maximum embedding capacity using custom method
    """
    # Find smooth blocks
    height, width, _ = rgb_matrix.shape
    blocks_h = height // block_size
    blocks_w = width // block_size
    total_blocks = blocks_h * blocks_w
    
    smooth_block_count = 0
    variances = []
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            h_start = i * block_size
            h_end = (i + 1) * block_size
            w_start = j * block_size
            w_end = (j + 1) * block_size
            block = rgb_matrix[h_start:h_end, w_start:w_end, :]
            variance = np.var(block)
            variances.append(variance)
            if variance < variance_threshold:
                smooth_block_count += 1
    
    # Debug information
    print(f"Custom capacity calculation:")
    print(f"  Image size: {height}x{width}")
    print(f"  Block size: {block_size}x{block_size}")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Variance threshold: {variance_threshold}")
    print(f"  Smooth blocks found: {smooth_block_count} ({smooth_block_count/total_blocks*100:.1f}%)")
    print(f"  Variance range: {np.min(variances):.2f} - {np.max(variances):.2f}")
    print(f"  Average variance: {np.mean(variances):.2f}")
    
    # CORRECTED: Maximum candidate blocks that can be selected
    max_candidate_blocks = smooth_block_count // 2  # This matches the actual selection logic
    # Each candidate block provides 1 mother-child pair, each pair stores 3 segments * 2 bits = 6 bits
    max_capacity = max_candidate_blocks * 3 * 2
    
    print(f"  Maximum candidate blocks: {max_candidate_blocks}")
    print(f"  Max capacity: {max_capacity} bits")
    
    return max_capacity

def evaluate_steganography(original_img, stego_img, original_data, extracted_data, max_capacity=None, checksum_failed=False, extraction_failed=False):
    """
    Evaluate steganography system performance: capacity, visual quality, and data integrity
    """
    capacity = len(original_data)
    psnr_value = psnr(original_img, stego_img)
    
    try:
        ssim_value = ssim(original_img, stego_img, multichannel=True, win_size=3)
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        ssim_value = 0.0
    
    try:
        ms_ssim_value = calculate_ms_ssim(original_img, stego_img)
    except Exception as e:
        print(f"Warning: MS-SSIM calculation failed: {e}")
        ms_ssim_value = 0.0
    
    fsim_value = calculate_fsim(original_img, stego_img)
    
    # Handle data integrity evaluation based on extraction status
    if extraction_failed:
        bit_accuracy = 0.0
        error_positions = []
        error_count = len(original_data)  # All bits are considered errors
        integrity_status = "Extraction Failed"
    elif checksum_failed:
        # Calculate real bit accuracy even when checksum fails
        min_len = min(len(original_data), len(extracted_data))
        correct_bits = sum(1 for a, b in zip(original_data[:min_len], extracted_data[:min_len]) if a == b)
        bit_accuracy = correct_bits / min_len * 100 if min_len > 0 else 0
        error_positions = [i for i in range(min_len) if original_data[i] != extracted_data[i]]
        error_count = len(error_positions)
        integrity_status = "Checksum Failed"
    else:
        # Normal case: compare original and extracted data
        min_len = min(len(original_data), len(extracted_data))
        correct_bits = sum(1 for a, b in zip(original_data[:min_len], extracted_data[:min_len]) if a == b)
        bit_accuracy = correct_bits / min_len * 100 if min_len > 0 else 0
        error_positions = [i for i in range(min_len) if original_data[i] != extracted_data[i]]
        error_count = len(error_positions)
        integrity_status = "Success"
    
    evaluation = {
        "capacity": capacity,
        "max_capacity": max_capacity if max_capacity else capacity,
        "capacity_bpp": capacity / (original_img.shape[0] * original_img.shape[1]),
        "max_capacity_bpp": (max_capacity if max_capacity else capacity) / (original_img.shape[0] * original_img.shape[1]),
        "psnr": psnr_value,
        "ssim": ssim_value,
        "ms_ssim": ms_ssim_value,
        "fsim": fsim_value,
        "bit_accuracy": bit_accuracy,
        "error_count": error_count,
        "error_positions": error_positions[:10] if error_positions else [],  # First 10 error positions
        "integrity_status": integrity_status,
        "checksum_failed": checksum_failed,
        "extraction_failed": extraction_failed
    }
    
    print("\nCustom Steganography System Evaluation:")
    print(f"  Used capacity: {capacity} bits ({evaluation['capacity_bpp']:.4f} bpp)")
    print(f"  Maximum capacity: {evaluation['max_capacity']} bits ({evaluation['max_capacity_bpp']:.4f} bpp)")
    print(f"  Capacity utilization: {(capacity / evaluation['max_capacity'] * 100):.2f}%")
    print(f"  PSNR: {psnr_value:.2f} dB")
    print(f"  SSIM: {ssim_value:.4f}")
    print(f"  MS-SSIM: {ms_ssim_value:.4f}")
    print(f"  FSIM: {fsim_value:.4f}")
    print(f"  Data integrity status: {integrity_status}")
    if not extraction_failed:
        print(f"  Bit accuracy: {bit_accuracy:.2f}%")
        print(f"  Error count: {error_count} of {min(len(original_data), len(extracted_data))} bits")
        if error_positions:
            print(f"  First few error positions: {evaluation['error_positions']}")
    else:
        print(f"  Bit accuracy: N/A ({integrity_status})")
    
    return evaluation 