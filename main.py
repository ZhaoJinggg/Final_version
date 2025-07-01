from comparison import compare_steganography_methods

def main():
    image_path = r"C:\Users\赵小镜\Desktop\UAV\Final_version\Assets\pepper.bmp"
    
    # Example secret message
    secret_message = "This is a test of LSB vs custom block-based steganography methods. This message includes some longer text to ensure we have enough data to embed and compare the two approaches effectively."
    
    # Run the comparison
    try:
        results = compare_steganography_methods(image_path, secret_message, "output")
        
        if results:
            lsb_stego_img, custom_stego_img, lsb_evaluation, custom_evaluation = results
            
            print("\n=== Comparison Complete ===")
            print("Results saved to 'output' folder")
            print("- lsb_stego.png: LSB steganography result")
            print("- custom_stego.png: Custom steganography result")
            
        else:
            print("Comparison failed due to errors in steganography processes")
            
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        print("Please update the image_path variable with a valid image file path")
        
    except Exception as e:
        print(f"Error occurred during comparison: {e}")
        
if __name__ == "__main__":
    main() 