# import cv2  # Temporarily disabled due to NumPy 2.x compatibility issues
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import concurrent.futures
from tqdm import tqdm

def load_segmentation_model():
    """Load YOLOv8 large segmentation model."""
    model = YOLO('yolov8l-seg.pt')  # Large segmentation model
    return model

def segment_persons_from_image(image, segmentation_model, min_confidence=0.5):
    """
    Segment individual persons from image and return separate crops with white background.
    
    Args:
        image: PIL Image or numpy array
        segmentation_model: YOLOv8 segmentation model
        min_confidence: Minimum confidence threshold for person detection
    
    Returns:
        List of PIL Images, each containing one person with white background
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Run segmentation
    try:
        results = segmentation_model(img_array)
    except Exception as e:
        print(f"[ERROR] Segmentation model failed: {e}")
        return []
    
    person_crops = []
    
    # Process each detected person separately
    if len(results) > 0 and results[0].boxes is not None and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Check if it's a person (class 0) and confidence is high enough
            if len(box) >= 6 and int(box[5]) == 0 and box[4] >= min_confidence:
                try:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # Add some padding around the bounding box
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(img_array.shape[1], x2 + padding)
                    y2 = min(img_array.shape[0], y2 + padding)
                    
                    # Resize mask to image size using PIL instead of cv2
                    mask_pil = Image.fromarray(mask.astype(np.float32))
                    mask_resized = mask_pil.resize((img_array.shape[1], img_array.shape[0]), Image.Resampling.NEAREST)
                    mask_bool = np.array(mask_resized) > 0.5
                    
                    # Crop the image and mask to bounding box
                    cropped_image = img_array[y1:y2, x1:x2]
                    cropped_mask = mask_bool[y1:y2, x1:x2]
                    
                    # Create white background for the crop
                    white_background = np.ones_like(cropped_image) * 255
                    
                    # Apply mask: keep person pixels, white background for rest
                    segmented_crop = np.where(cropped_mask[..., np.newaxis], cropped_image, white_background)
                    
                    # Convert to PIL Image
                    person_crop = Image.fromarray(segmented_crop.astype(np.uint8))
                    person_crops.append(person_crop)
                except Exception as e:
                    print(f"[ERROR] Failed to process person {i}: {e}")
                    continue
    
    return person_crops

def process_single_image_segmentation(args):
    """Process a single image for segmentation (for multiprocessing)."""
    image, image_idx, segmentation_model = args
    
    try:
        # Segment the image into individual person crops
        person_crops = segment_persons_from_image(image, segmentation_model)
        
        return image_idx, person_crops
    except Exception as e:
        print(f"[ERROR] Failed to process image {image_idx}: {e}")
        return image_idx, []

def create_segmented_images(images, output_dir, max_workers=8):
    """
    Create segmented images with white background for each detected person.
    Each person becomes a separate image.
    
    Args:
        images: List of PIL Images
        output_dir: Directory to save segmented images
        max_workers: Number of parallel workers
    
    Returns:
        List of segmented PIL Images (one per detected person)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load segmentation model
    print("[INFO] Loading YOLOv8 large segmentation model...")
    segmentation_model = load_segmentation_model()
    
    all_segmented_images = []
    total_persons = 0
    
    # Process images sequentially for debugging
    print("[INFO] Processing images sequentially for debugging...")
    for i, image in enumerate(tqdm(images, desc="Creating segmented person crops")):
        try:
            # Segment the image into individual person crops
            person_crops = segment_persons_from_image(image, segmentation_model)
            
            # Save each person crop with unique filename
            for person_idx, person_crop in enumerate(person_crops):
                filename = f"image_{i:06d}_person_{person_idx:02d}.png"
                filepath = os.path.join(output_dir, filename)
                person_crop.save(filepath)
                all_segmented_images.append(person_crop)
                total_persons += 1
                
            if i % 100 == 0:  # Progress update every 100 images
                print(f"[DEBUG] Processed {i} images, found {total_persons} persons so far")
                
        except Exception as e:
            print(f"[ERROR] Failed to process image {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[INFO] Successfully created {total_persons} individual person crops from {len(images)} original images")
    if len(images) > 0:
        print(f"[INFO] Average: {total_persons/len(images):.1f} persons per original image")
    return all_segmented_images

def segment_person_crop(person_crop, segmentation_model=None):
    """
    Segment person in a crop from video frame.
    
    Args:
        person_crop: Numpy array of person crop from video frame
        segmentation_model: YOLOv8 segmentation model (loaded if None)
    
    Returns:
        Numpy array with person segmented and white background
    """
    if segmentation_model is None:
        segmentation_model = load_segmentation_model()
    
    # Convert to PIL and get segmented crops
    pil_crop = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
    person_crops = segment_persons_from_image(pil_crop, segmentation_model)
    
    # Return the first (and likely only) segmented crop, or original if none found
    if person_crops:
        segmented_pil = person_crops[0]
        # Convert back to BGR numpy array for video processing
        segmented_crop = cv2.cvtColor(np.array(segmented_pil), cv2.COLOR_RGB2BGR)
        return segmented_crop
    else:
        # If segmentation fails, return original crop
        return person_crop
