from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import os

def ela_analysis(image_path, quality=90):
    # Open original image
    original = Image.open(image_path).convert("RGB")
    
    # Save at lower quality temporarily
    temp_path = "temp_ela.jpg"
    original.save(temp_path, "JPEG", quality=quality)
    
    # Open the re-saved version
    compressed = Image.open(temp_path).convert("RGB")
    
    # Find the difference
    diff = ImageChops.difference(original, compressed)
    
    # Amplify the difference so we can see it
    diff_array = np.array(diff)
    scale = 15
    amplified = np.clip(diff_array * scale, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(amplified)
    
    # Calculate suspicion score
    avg_brightness = np.mean(amplified)
    
    # Clean up temp file
    os.remove(temp_path)
    
    return ela_image, avg_brightness

def check_forgery_ela(image_path):
    print(f"Analyzing: {image_path}")
    
    ela_img, score = ela_analysis(image_path)
    
    # Show results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(Image.open(image_path))
    axes[0].set_title("Original Document")
    axes[0].axis("off")
    
    # ELA image
    axes[1].imshow(ela_img)
    axes[1].set_title(f"ELA Analysis\nSuspicion Score: {score:.2f}")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig("ela_result.png")
    plt.show()
    
    # Give verdict
    print(f"\nELA Suspicion Score: {score:.2f}")
    if score > 15:
        print("⚠️  SUSPICIOUS — Possible forgery detected!")
    else:
        print("✅  LOOKS CLEAN — No obvious tampering")
    
    print("Result saved as ela_result.png")

# Test it on your image
IMAGE = r"C:\Users\victus\OneDrive\New folder\forgery project\data\findit2\test\X00016469619.png"
check_forgery_ela(IMAGE)