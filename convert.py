#!/usr/bin/env python3
"""
Image Converter Script

This script converts all images in ./samples/input directory to 768x1024 resolution
and saves them to samples/output directory with reverse sequential naming (img_999.png, img_998.png, etc.)

Usage:
    python convert.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--start-number START_NUM]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from PIL import Image, ImageOps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_image_files(input_dir):
    """Get list of all image files in input directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    try:
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in image_extensions:
                    image_files.append(file_path)
        
        # Sort files for consistent processing order
        image_files.sort()
        logger.info(f"Found {len(image_files)} image files in {input_dir}")
        return image_files
        
    except Exception as e:
        logger.error(f"Error reading input directory {input_dir}: {e}")
        return []

def convert_image(input_path, output_path, target_size=(768, 1024)):
    """Convert single image to target size and save"""
    try:
        # Open and convert image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get original size
            original_size = img.size
            logger.info(f"  Original size: {original_size[0]}x{original_size[1]}")
            
            # Resize image to target size
            # Using LANCZOS for high quality resampling
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save as PNG
            resized_img.save(output_path, 'PNG', optimize=True)
            logger.info(f"  Converted to: {target_size[0]}x{target_size[1]} -> {output_path}")
            return True
            
    except Exception as e:
        logger.error(f"  Error converting {input_path}: {e}")
        return False

def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(
        description="Convert images to 768x1024 with reverse sequential naming"
    )
    parser.add_argument(
        '--input', 
        default='samples/input',
        help='Input directory containing images (default: samples/input)'
    )
    parser.add_argument(
        '--output', 
        default='samples/output',
        help='Output directory for converted images (default: samples/output)'
    )
    parser.add_argument(
        '--start-number',
        type=int,
        default=999,
        help='Starting number for reverse naming (default: 999)'
    )
    parser.add_argument(
        '--target-width',
        type=int,
        default=768,
        help='Target width in pixels (default: 768)'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=1024,
        help='Target height in pixels (default: 1024)'
    )
    
    args = parser.parse_args()
    
    # Set up directories
    input_dir = args.input
    output_dir = args.output
    target_size = (args.target_width, args.target_height)
    
    logger.info("🖼️ Image Converter Started")
    logger.info("=" * 50)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target size: {target_size[0]}x{target_size[1]}")
    logger.info(f"Starting number: {args.start_number}")
    
    # Check input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"❌ Input directory does not exist: {input_dir}")
        logger.error("Please create the directory and add images to convert")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"✅ Output directory ready: {output_dir}")
    
    # Get list of image files
    image_files = get_image_files(input_dir)
    if not image_files:
        logger.error(f"❌ No image files found in {input_dir}")
        logger.error("Supported formats: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP")
        return 1
    
    # Convert images with reverse sequential naming
    logger.info(f"🔄 Converting {len(image_files)} images...")
    
    success_count = 0
    current_number = args.start_number
    
    for i, input_path in enumerate(image_files):
        input_filename = os.path.basename(input_path)
        output_filename = f"img_{current_number:03d}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        logger.info(f"Converting {i+1}/{len(image_files)}: {input_filename} -> {output_filename}")
        
        if convert_image(input_path, output_path, target_size):
            success_count += 1
            current_number -= 1  # Decrement for next image
        else:
            logger.warning(f"⚠️ Failed to convert: {input_filename}")
    
    # Summary
    logger.info("=" * 50)
    logger.info(f"🎉 Conversion completed!")
    logger.info(f"✅ Successfully converted: {success_count}/{len(image_files)} images")
    
    if success_count > 0:
        logger.info(f"📁 Output files: img_{args.start_number:03d}.png to img_{current_number+1:03d}.png")
        logger.info(f"📂 Saved to: {output_dir}")
    
    if success_count < len(image_files):
        failed_count = len(image_files) - success_count
        logger.warning(f"⚠️ Failed conversions: {failed_count}")
        return 1
    
    return 0

def check_dependencies():
    """Check if PIL/Pillow is available"""
    try:
        from PIL import Image
        logger.info("✅ PIL/Pillow is available")
        return True
    except ImportError:
        logger.error("❌ PIL/Pillow is not installed")
        logger.error("Install with: pip install Pillow")
        return False

if __name__ == "__main__":
    logger.info("🖼️ Image Converter Tool")
    logger.info("Converts images to 768x1024 with reverse sequential naming")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run conversion
    exit_code = main()
    sys.exit(exit_code)