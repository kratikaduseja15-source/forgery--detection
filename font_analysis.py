import easyocr
import cv2
import numpy as np
from PIL import Image
import json

def analyze_fonts(image_path):
    print(f"Analyzing fonts in: {image_path}")
    print("Loading OCR engine... (first time is slow)")

    # Load OCR — supports English + Hindi + other languages
    reader = easyocr.Reader(['en', 'hi'], gpu=False)

    # Read image
    img = cv2.imread(image_path)
    results = reader.readtext(image_path)

    if not results:
        print("No text found in document!")
        return

    # Collect all text box sizes
    heights = []
    widths  = []
    texts   = []

    for (bbox, text, confidence) in results:
        if confidence < 0.3:
            continue  # skip low confidence text

        # bbox = 4 corner points
        top_left     = bbox[0]
        top_right    = bbox[1]
        bottom_right = bbox[2]

        # Calculate height and width of each text box
        height = abs(bottom_right[1] - top_left[1])
        width  = abs(top_right[0]  - top_left[0])

        heights.append(height)
        widths.append(width)
        texts.append(text)

    if len(heights) < 2:
        print("Not enough text to analyze")
        return

    # Calculate statistics
    avg_height = np.mean(heights)
    std_height = np.std(heights)

    print(f"\n📊 Font Analysis Results:")
    print(f"Total text regions found: {len(heights)}")
    print(f"Average text height: {avg_height:.1f}px")
    print(f"Height variation: {std_height:.1f}px")

    # Find suspicious text (very different size)
    suspicious = []
    for i, (h, t) in enumerate(zip(heights, texts)):
        deviation = abs(h - avg_height)
        if deviation > avg_height * 0.4:  # 40% different from average
            suspicious.append({
                "text": t,
                "height": round(h, 1),
                "deviation": round(deviation, 1)
            })

    # Give verdict
    print(f"\nSuspicious text regions: {len(suspicious)}")

    if suspicious:
        print("\n⚠️  SUSPICIOUS TEXT FOUND:")
        for s in suspicious:
            print(f"  → '{s['text']}' "
                  f"(size: {s['height']}px, "
                  f"deviation: {s['deviation']}px)")
        print("\n❌ FONT INCONSISTENCY DETECTED — Possible forgery!")
        score = "HIGH RISK"
    else:
        print("\n✅ All text looks consistent — No font tampering detected!")
        score = "LOW RISK"

    # Draw boxes on image
    output_img = img.copy()
    for (bbox, text, conf) in results:
        if conf < 0.3:
            continue
        pts = np.array(bbox, dtype=np.int32)
        
        # Check if this text is suspicious
        h = abs(bbox[2][1] - bbox[0][1])
        deviation = abs(h - avg_height)
        
        if deviation > avg_height * 0.4:
            color = (0, 0, 255)    # Red = suspicious
        else:
            color = (0, 255, 0)    # Green = normal

        cv2.polylines(output_img, [pts], True, color, 2)

    # Save result image
    output_path = "font_analysis_result.png"
    cv2.imwrite(output_path, output_img)
    print(f"\nResult image saved: {output_path}")
    print(f"GREEN boxes = Normal text ✅")
    print(f"RED boxes   = Suspicious text ❌")
    print(f"\nOverall Risk: {score}")

    return score

# Test it
IMAGE = r"C:\Users\victus\OneDrive\New folder\forgery project\data\findit2\test\X00016469619.png"
analyze_fonts(IMAGE)