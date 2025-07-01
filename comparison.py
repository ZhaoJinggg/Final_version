import os
from lsb_steganography import LSBSteganographySystem
from custom_steganography import CustomSteganographySystem
from evaluation import print_comparison_table

def compare_steganography_methods(image_path, secret_data=None, output_folder=None):
    """
    Run a complete comparison between LSB and custom steganography methods
    """
    # Create output folder if needed
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Default secret message if none provided
    if secret_data is None:
        secret_data = "This is a secret message for testing our steganography systems. It includes some Unicode characters: 你好, こんにちは, مرحبا"
    
    print("\n=== Starting Steganography Methods Comparison ===")
    print(f"Image: {image_path}")
    print(f"Secret data ({len(secret_data)} characters): {secret_data[:50]}...")
    
    # Initialize systems
    lsb_system = LSBSteganographySystem(bits_per_channel=1)
    custom_system = CustomSteganographySystem(
        block_size=8, 
        sub_block_size=1,
        segment_length=2, 
        lfsr_seed=0x1234
    )
    
    # Output paths
    lsb_output_path = os.path.join(output_folder, "lsb_stego.png") if output_folder else None
    custom_output_path = os.path.join(output_folder, "custom_stego.png") if output_folder else None
    
    # Run LSB steganography
    print("\n--- LSB Steganography ---")
    lsb_stego_img = lsb_system.embed(image_path, secret_data, lsb_output_path)
    if lsb_stego_img is None:
        print("LSB steganography failed, cannot compare methods")
        return
    
    lsb_extracted_data = lsb_system.extract(stego_img=lsb_stego_img)
    if lsb_extracted_data is None:
        print("LSB extraction failed")
    else:
        print(f"LSB extracted data ({len(lsb_extracted_data)} characters): {lsb_extracted_data[:50]}...")
        print(f"LSB extraction success: {lsb_extracted_data == secret_data}")
    
    lsb_evaluation = lsb_system.evaluate(lsb_stego_img)
    
    # Run custom steganography
    print("\n--- Custom Steganography ---")
    custom_stego_img = custom_system.embed(image_path, secret_data, custom_output_path)
    if custom_stego_img is None:
        print("Custom steganography failed, cannot compare methods")
        return
    
    custom_extracted_data = custom_system.extract(stego_img=custom_stego_img)
    if custom_extracted_data is None:
        print("Custom extraction failed")
    else:
        print(f"Custom extracted data ({len(custom_extracted_data)} characters): {custom_extracted_data[:50]}...")
        print(f"Custom extraction success: {custom_extracted_data == secret_data}")
    
    custom_evaluation = custom_system.evaluate(custom_stego_img)
    
    # Print comparison table
    if lsb_evaluation and custom_evaluation:
        print_comparison_table(lsb_evaluation, custom_evaluation)
    
    return lsb_stego_img, custom_stego_img, lsb_evaluation, custom_evaluation 