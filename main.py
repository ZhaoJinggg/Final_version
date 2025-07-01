from custom_steganography import CustomSteganographySystem
import os
import time

def print_results_table(results):
    """
    Print a formatted table showing results for all images
    """
    print(" "*45 + "CUSTOM STEGANOGRAPHY RESULTS TABLE")  
      
    # Header
    header = f"{'Image':<15} {'PSNR':<8} {'SSIM':<8} {'MS-SSIM':<9} {'FSIM':<8} {'Accuracy':<10} {'Capacity':<9} {'Max Cap':<8} {'Util%':<7} {'Status':<10}"
    print(header)
    print("-" * 120)
    
    # Data rows
    for result in results:
        image_name = result['image_name']
        if 'evaluation' in result and result['evaluation']:
            # Visual quality metrics are available
            eval_data = result['evaluation']
            
            # Use custom status if available, otherwise determine from success
            if 'status' in result:
                status = result['status']
            else:
                status = "✓ Success" if result['success'] else "✗ Failed"
            
            # Show bit accuracy only if extraction was successful
            if eval_data.get('extraction_failed', False):
                bit_accuracy_str = "N/A"
            else:
                bit_accuracy_str = f"{eval_data['bit_accuracy']:.2f}"
            
            row = f"{image_name:<15} {eval_data['psnr']:<8.2f} {eval_data['ssim']:<8.4f} {eval_data['ms_ssim']:<9.4f} {eval_data['fsim']:<8.4f} {bit_accuracy_str:<10} {eval_data['capacity']:<9d} {eval_data['max_capacity']:<8d} {(eval_data['capacity']/eval_data['max_capacity']*100):<7.2f} {status:<10}"
        else:
            # No evaluation data available
            status = result.get('error', 'Failed')
            row = f"{image_name:<15} {'N/A':<8} {'N/A':<8} {'N/A':<9} {'N/A':<8} {'N/A':<10} {'N/A':<9} {'N/A':<8} {'N/A':<7} {status:<10}"
        print(row)
    
    print("-" * 120)
    print("\nLegend:")
    print("- PSNR: Peak Signal-to-Noise Ratio (dB) - Higher is better (>50 dB is excellent)")
    print("- SSIM: Structural Similarity Index (0-1) - Higher is better (>0.95 is excellent)")
    print("- MS-SSIM: Multi-Scale SSIM (0-1) - Higher is better")
    print("- FSIM: Feature Similarity Index (0-1) - Higher is better")
    print("- Accuracy: Bit extraction accuracy (%) - Higher is better (>95% is good, N/A if extraction failed)")
    print("- Capacity: Actual bits embedded")
    print("- Max Cap: Maximum theoretical capacity")
    print("- Util%: Capacity utilization percentage")
    print("- Status: Complete Success / Checksum Failed / Extraction Failed / Failed")

def main():
    # List of all images in Assets folder
    asset_folder = "Assets"
    images = [
        "airplane.bmp",
        "Baboon.tiff", 
        "lenna.bmp",
        "pepper.bmp",
        "Sailboat.tiff",
        "Splash.tiff"
    ]
    
    # Example secret message
    secret_message = "This is a test of custom block-based steganography method. This message includes some longer text to ensure we have enough data to embed and test the custom approach effectively."
    
    # Create output folder if needed
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize custom steganography system
    custom_system = CustomSteganographySystem(
        block_size=8, 
        sub_block_size=1,
        segment_length=2, 
        lfsr_seed=0x1234
    )
    
    print("="*80)
    print(" "*25 + "CUSTOM STEGANOGRAPHY - MULTI-IMAGE TEST")
    print("="*80)
    print(f"Processing {len(images)} images from Assets folder")
    print(f"Secret message length: {len(secret_message)} characters")
    print("="*80)
    
    results = []
    
    # Process each image
    for i, image_file in enumerate(images, 1):
        image_path = os.path.join(asset_folder, image_file)
        image_name = os.path.splitext(image_file)[0]  # Remove extension for cleaner display
        
        print(f"\n[{i}/{len(images)}] Processing: {image_file}")
        print("-" * 60)
        
        try:
            # Set output path
            custom_output_path = os.path.join(output_folder, f"{image_name}_custom_stego.png")
            
            # Embed data
            print(f"Embedding data into {image_file}...")
            custom_stego_img = custom_system.embed(image_path, secret_message, custom_output_path)
            if custom_stego_img is None:
                print(f"❌ Embedding failed for {image_file}")
                results.append({
                    'image_name': image_name,
                    'success': False,
                    'error': 'Embedding failed'
                })
                continue
            
            # Extract data
            print(f"Extracting data from {image_file}...")
            custom_extracted_data = custom_system.extract(stego_img=custom_stego_img)
            extraction_success = custom_extracted_data == secret_message
            
            # Always try to evaluate, even if extraction failed
            print(f"Evaluating {image_file}...")
            custom_evaluation = custom_system.evaluate(custom_stego_img)
            
            if custom_evaluation:
                print(f"✅ Visual quality metrics calculated for {image_file}")
                print(f"   PSNR: {custom_evaluation['psnr']:.2f} dB")
                print(f"   SSIM: {custom_evaluation['ssim']:.4f}")
                
                # Determine overall success status
                if custom_extracted_data is None:
                    print(f"❌ Extraction failed for {image_file}")
                    status = 'Extraction Failed'
                    success = False
                elif custom_evaluation.get('checksum_failed', False):
                    print(f"⚠️ Checksum verification failed for {image_file}")
                    print(f"   Bit Accuracy: {custom_evaluation['bit_accuracy']:.2f}%")
                    status = 'Checksum Failed'  
                    success = False
                elif custom_evaluation.get('extraction_failed', False):
                    print(f"⚠️ Data extraction failed for {image_file}")
                    status = 'Extraction Failed'
                    success = False
                else:
                    print(f"✓ Extraction success: {extraction_success}")
                    print(f"   Bit Accuracy: {custom_evaluation['bit_accuracy']:.2f}%")
                    status = 'Complete Success'
                    success = True
                    
                print(f"   Output saved: {custom_output_path}")
                
                results.append({
                    'image_name': image_name,
                    'success': success,
                    'evaluation': custom_evaluation,
                    'extraction_success': extraction_success,
                    'output_path': custom_output_path,
                    'status': status
                })
            else:
                print(f"❌ Evaluation failed for {image_file}")
                results.append({
                    'image_name': image_name,
                    'success': False,
                    'error': 'Evaluation failed'
                })
                
        except FileNotFoundError:
            print(f"❌ Error: Image file not found at {image_path}")
            results.append({
                'image_name': image_name,
                'success': False,
                'error': 'File not found'
            })
            
        except Exception as e:
            print(f"❌ Error processing {image_file}: {e}")
            results.append({
                'image_name': image_name,
                'success': False,
                'error': str(e)
            })
    
    # Print comprehensive results table
    print_results_table(results)

if __name__ == "__main__":
    main() 