#!/usr/bin/env python3
"""
Simplified video player for object detection demo
Based on translate_play.py but focused only on video playback
"""

import sys
import time
import gi
from pathlib import Path
import logging
import platform
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QColor
import os
from object_detection import DetectionCoordinator, ImageCropper
from image_embedding import ImageEmbedding
from segmentation_utils import segment_persons_from_image

# import cv2  # Removed to avoid OpenGL dependency
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PopupWindow(QMainWindow):
    def __init__(self, model_type=None):
        """
        Initialize the PyQt5-based pop-up window
        
        Args:
            model_type: Model type to use (YOLOV8S_SEG or YOLOV8L_SEG). If None, uses default.
        """
        super().__init__()
        self.video_player = None
        self.is_running = False
        
        # Initialize object detection coordinator with specified model type
        if model_type is None:
            # Use default model (YOLOv8l)
            from object_detection import ModelConfig
            model_type = ModelConfig.YOLOV8L_SEG
        
        self.detection_coordinator = DetectionCoordinator(model_type=model_type)
        self.image_cropper = ImageCropper()  # Initialize ImageCropper for bounding box cropping
        self.last_detection_results = None
        
        # Log current model configuration
        model_info = self.detection_coordinator.get_current_model_info()
        logger.info(f"🎯 PopupWindow: Initialized with model: {model_info['model_type']}")
        logger.info(f"🎯 PopupWindow: Model path: {model_info['model_path']}")
        logger.info(f"🎯 PopupWindow: Available models: {model_info['available_models']}")
        

        
        # Initialize FashionCLIP embedding system 
        # Note: Models and images should be prepared using embedding_model_prepare.py
        logger.info("Loading pre-prepared FashionCLIP embedding system...")
        self.image_embedding = ImageEmbedding(use_segmented=True)
        
        # Verify that models and images are ready
        if self.image_embedding.should_initialize():
            logger.error("❌ Models and images not prepared!")
            logger.error("🔧 Please run the preparation script first:")
            logger.error("   python embedding_model_prepare.py")
            logger.error("This will download models and images (~10-15 minutes)")
            sys.exit(1)
        
        # Pre-load embeddings for fast similarity search during video playback
        logger.info("Loading pre-computed FashionCLIP embeddings...")
        if not self.image_embedding.load_existing_embeddings():
            logger.error("❌ Failed to load embeddings!")
            logger.error("🔧 Please run the preparation script:")
            logger.error("   python embedding_model_prepare.py")
            sys.exit(1)
        else:
            logger.info("✅ FashionCLIP model and embeddings loaded successfully - ready for fast matching!")
        
        # Navigation state for bounding boxes (matching Kotlin's focus system)
        self.current_focus_index = 0
        self.detection_coordinates = []
        self.is_in_detection_mode = False
        
        # Store original frame for clean bounding box redrawing
        self.original_frame_pixmap = None
        self.cropped_images = []  # Store cropped images for each detected object
        
        logger.info("PopupWindow: Object detection coordinator initialized")
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Video Shopping Demo")
        
        # Start in fullscreen mode
        self.showFullScreen()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title section
        title_label = QLabel("🎬 Video Shopping Demo")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        main_layout.addWidget(title_label)
        
        # Create fixed split-screen layout container
        self.video_container_layout = QHBoxLayout()
        self.video_container_layout.setSpacing(2)  # Set minimal spacing between video and right panels
        
        # Video frame setup (left side)
        self.video_frame = QFrame()
        self.video_frame.setMinimumHeight(200)
        self.video_frame.setStyleSheet("background-color: #000000; border: 0; border-radius: 12px;")
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Make the video frame focusable and ready for video embedding
        self.video_frame.setFocusPolicy(Qt.StrongFocus)
        
        video_layout = QVBoxLayout(self.video_frame)
        self.video_placeholder = QLabel("📺 GStreamer Video Will Be Embedded Here")
        self.video_placeholder.setFont(QFont("Arial", 12))
        self.video_placeholder.setAlignment(Qt.AlignCenter)
        self.video_placeholder.setStyleSheet("color: #bdc3c7; padding: 20px; border-radius: 12px;")
        video_layout.addWidget(self.video_placeholder)
        
        # create ONE label and keep it forever
        self.video_label = QLabel(self.video_frame)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(False)  
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 12px;")
        video_layout.addWidget(self.video_label)
        
        # Right-side panels (always visible) - 3x4 grid layout
        self.right_box_container = QVBoxLayout()
        self.right_box_container.setSpacing(5)  # Reduce spacing between rows from default 10px to 5px
        
        # Row 1: Fashion match placeholders (will be updated with FashionCLIP results)
        self.row1_frame = QFrame()
        self.row1_frame.setStyleSheet("background-color: #ffffff; border: 2px solid #cccccc; border-radius: 10px;")
        self.row1_frame.setMinimumHeight(120)
        
        row1_layout = QHBoxLayout(self.row1_frame)
        row1_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins from 10px to 5px
        row1_layout.setSpacing(5)  # Reduced spacing from 10px to 5px
        
        # Row 1 image labels (for FashionCLIP matches 1-3)
        self.row1_labels = []
        for i in range(3):
            label = QLabel(f"Match {i+1}")
            label.setStyleSheet("color: #cccccc; font-size: 12px; border: 1px solid #ddd;")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(150, 150)  # Increased size for better visibility
            label.setMaximumSize(250, 250)  # Increased maximum size
            label.setScaledContents(True)  # Allow image scaling
            row1_layout.addWidget(label)
            self.row1_labels.append(label)
        
        # Row 2: Fashion match placeholders (will be updated with FashionCLIP results)
        self.row2_frame = QFrame()
        self.row2_frame.setStyleSheet("background-color: #ffffff; border: 2px solid #cccccc; border-radius: 10px;")
        self.row2_frame.setMinimumHeight(120)
        
        row2_layout = QHBoxLayout(self.row2_frame)
        row2_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins from 10px to 5px
        row2_layout.setSpacing(5)  # Reduced spacing from 10px to 5px
        
        # Row 2 image labels (for FashionCLIP matches 4-6)
        self.row2_labels = []
        for i in range(3):
            label = QLabel(f"Match {i+4}")
            label.setStyleSheet("color: #cccccc; font-size: 12px; border: 1px solid #ddd;")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(150, 150)  # Increased size for better visibility
            label.setMaximumSize(250, 250)  # Increased maximum size
            label.setScaledContents(True)  # Allow image scaling
            row2_layout.addWidget(label)
            self.row2_labels.append(label)  
        
        # Row 3: Fashion match placeholders (will be updated with FashionCLIP results)
        self.row3_frame = QFrame()
        self.row3_frame.setStyleSheet("background-color: #ffffff; border: 2px solid #cccccc; border-radius: 10px;")
        self.row3_frame.setMinimumHeight(120)
        
        row3_layout = QHBoxLayout(self.row3_frame)
        row3_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins from 10px to 5px
        row3_layout.setSpacing(5)  # Reduced spacing from 10px to 5px
        
        # Row 3 image labels (for FashionCLIP matches 7-9)
        self.row3_labels = []
        for i in range(3):
            label = QLabel(f"Match {i+7}")
            label.setStyleSheet("color: #cccccc; font-size: 12px; border: 1px solid #ddd;")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(150, 150)  # Increased size for better visibility
            label.setMaximumSize(250, 250)  # Increased maximum size
            label.setScaledContents(True)  # Allow image scaling
            row3_layout.addWidget(label)
            self.row3_labels.append(label)
        
        # Add all rows to right container
        self.right_box_container.addWidget(self.row1_frame)
        self.right_box_container.addWidget(self.row2_frame)
        self.right_box_container.addWidget(self.row3_frame)
        
        # Create right panel widget (always visible)
        self.right_box_widget = QWidget()
        self.right_box_widget.setLayout(self.right_box_container)
        self.right_box_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        # Set up fixed split-screen layout: video left, panels right
        self.video_container_layout.addWidget(self.video_frame, 2)  # Video takes 2/3 of space
        self.video_container_layout.addWidget(self.right_box_widget, 1)  # Panels take 1/3 of space
        
        main_layout.addLayout(self.video_container_layout)
        
        # Buttons section
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)  # Add spacing between buttons
        
        # Match Style button (same as space key)
        self.match_style_button = QPushButton("🎯 Match Style")
        self.match_style_button.setFont(QFont("Arial", 11))
        self.match_style_button.setFocusPolicy(Qt.NoFocus)  # Disable focus to prevent keyboard interference
        self.match_style_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                min-height: 45px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        button_layout.addWidget(self.match_style_button)
        
        # Previous button (same as left arrow)
        self.previous_button = QPushButton("⬅️ Previous")
        self.previous_button.setFont(QFont("Arial", 11))
        self.previous_button.setFocusPolicy(Qt.NoFocus)  # Disable focus to prevent keyboard interference
        self.previous_button.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                min-height: 45px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)
        button_layout.addWidget(self.previous_button)
        
        # Next button (same as right arrow)
        self.next_button = QPushButton("➡️ Next")
        self.next_button.setFont(QFont("Arial", 11))
        self.next_button.setFocusPolicy(Qt.NoFocus)  # Disable focus to prevent keyboard interference
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                min-height: 45px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)
        button_layout.addWidget(self.next_button)
        

        
        # Close Window button (same as Q key)
        self.close_button = QPushButton("❌ Close Window")
        self.close_button.setFont(QFont("Arial", 11))
        self.close_button.setFocusPolicy(Qt.NoFocus)  # Disable focus to prevent keyboard interference
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                min-height: 45px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        button_layout.addWidget(self.close_button)
        
        main_layout.addWidget(QWidget())  # Spacer
        main_layout.addLayout(button_layout)
        
        # Set window properties
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        
        # Ensure window can receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
        
        logger.info("PyQt5 pop-up window created successfully")
        
    def keyPressEvent(self, event):
        """Handle keyboard events in the pop-up window"""
        key = event.key()
        key_text = event.text()
        
        logger.info(f"Key pressed in pop-up window: {key_text} (code: {key})")
        
        if key == Qt.Key_Plus or key_text == '+' or key_text == '=':
            self.on_plus_key()
        elif key == Qt.Key_Space:
            self.on_space_key()
        elif key == Qt.Key_Left or key == Qt.Key_Right:
            self.navigate_bounding_boxes(key)
        elif key == Qt.Key_Q:
            self.on_quit_key()
        else:
            super().keyPressEvent(event)
        
    def on_space_key(self):
        """Handle space key press - toggle video pause"""
        if self.video_player:
            self.video_player.toggle_pause()
            status = "paused ⏸️" if self.video_player.is_paused else "playing ▶️"
            logger.info(f"Video {status}")
            
            # Update button label based on video state
            self.update_match_style_button_label()
            
            # Trigger object detection when video is paused
            if self.video_player.is_paused:
                logger.info("🔍 Video paused - triggering object detection...")
                
                # Debug layout sizes when paused (when user is actually looking at it)
                self.debug_layout_sizes()
                
                # Use model.synap for fast detection via DetectionCoordinator
                self.detection_coordinator.on_video_paused(self.video_frame, callback=self.on_detection_complete)
            else:
                # Only reset detection if we're not currently in detection mode
                # This prevents interrupting ongoing detection
                if not self.is_in_detection_mode:
                    self.detection_coordinator.reset()
                    logger.info("🔄 Video resumed - detection reset")
                    self.detection_coordinates = []
                    self.current_focus_index = 0
                    self.draw_single_bounding_box(-1)  # Clear bounding box
                else:
                    logger.info("🔄 Video resumed - keeping detection results active")
    
    def on_detection_complete(self, results):
        """Handle detection completion - store results and show first bounding box"""
        coordinates = results.get('coordinates', [])
        person_detections = results.get('person_detections', [])
        
        logger.info(f"🎯 Detection completed with {len(coordinates)} person detections")
        
        # Store detection results (already filtered to persons only by CoordinateProcessor)
        self.detection_coordinates = coordinates
        self.current_focus_index = 0 if coordinates else -1
        self.is_in_detection_mode = len(coordinates) > 0
        
        # Clear previous cropped images
        self.cropped_images.clear()
        
        # Store original frame for clean redrawing
        # Use the cached frame file as the original frame to avoid using frames with existing bounding boxes
        import os
        cached_frame_path = "cache/paused_frame.png"
        if os.path.exists(cached_frame_path):
            from PyQt5.QtGui import QPixmap
            self.original_frame_pixmap = QPixmap(cached_frame_path)
            logger.info(f"PopupWindow: Stored original frame from cache: {cached_frame_path}")
        else:
            # Fallback: Get from video label widget
            video_label = None
            if hasattr(self, '_video_label') and self._video_label:
                video_label = self._video_label
            else:
                # Look for QLabel children in the video frame
                from PyQt5.QtWidgets import QLabel
                for child in self.video_frame.findChildren(QLabel):
                    if child.pixmap() is not None:
                        video_label = child
                        break
            
            if video_label and video_label.pixmap():
                self.original_frame_pixmap = video_label.pixmap().copy()
                logger.info("PopupWindow: Stored original frame from video label (fallback)")
            else:
                logger.warning("PopupWindow: No original frame available for bounding box drawing")
        
        # Show first bounding box if objects detected
        if self.is_in_detection_mode:
            logger.info(f"PopupWindow: About to draw first bounding box. Detection mode: {self.is_in_detection_mode}, Coordinates count: {len(coordinates)}")
            logger.info(f"PopupWindow: Original frame pixmap available: {self.original_frame_pixmap is not None and not self.original_frame_pixmap.isNull()}")
            self.draw_single_bounding_box(0)
            logger.info(f"PopupWindow: Showing first bounding box (1/{len(coordinates)})")
        else:
            logger.warning(f"PopupWindow: Not in detection mode. Coordinates: {len(coordinates)}, Detection mode: {self.is_in_detection_mode}")
        
        # Log detailed detection results (persons only)
        for i, (x, y, width, height) in enumerate(coordinates):
            logger.info(f"👤 Person {i + 1} at ({x}, {y}) size {width}x{height}")
        
        for detection in person_detections:
            logger.info(f"👤 Detected person {detection.get('label', 'unknown')} (class {detection.get('class_index', -1)}) at ({detection.get('x', 0)}, {detection.get('y', 0)})")
        
        # Integrate FashionCLIP similarity search with object detection
        self.show_detection_at_index(0)
    
    def show_detection_at_index(self, index):
        """Show detection and find similar fashion items"""
        if not self.detection_coordinates or index >= len(self.detection_coordinates):
            logger.warning(f"Cannot show detection {index}: no coordinates available")
            return
        
        coord = self.detection_coordinates[index]
        x, y, width, height = coord  # coord is now a tuple (x, y, width, height)
        logger.info(f"Showing detection {index + 1}: ({x}, {y}, {width}, {height})")
        
        # Step 2: Generate segmented person crop for the highlighted person (on-demand)
        logger.info(f"🎯 Generating segmented person crop for highlighted person {index + 1}")
        crop_path = self.detection_coordinator.generate_mask_for_person(index)
        
        if crop_path:
            logger.info(f"✅ Successfully generated segmented person crop for person {index + 1} -> {crop_path}")
            
            # Load the segmented person crop as QPixmap
            from PyQt5.QtGui import QPixmap
            crop_pixmap = QPixmap(crop_path)
            
            if not crop_pixmap.isNull():
                logger.info(f"🎯 Successfully loaded segmented person crop for person {index + 1}, starting FashionCLIP search...")
                # Find similar fashion items using FashionCLIP with segmented person crop
                self.find_similar_fashion_items(crop_pixmap, index)
            else:
                logger.error(f"❌ Failed to load segmented person crop for person {index + 1}")
                # Fallback to old cropping method
                self._fallback_to_cropping(index, x, y, width, height)
        else:
            logger.warning(f"⚠️ Failed to generate segmented person crop for person {index + 1}, falling back to cropping method")
            # Fallback to old cropping method
            self._fallback_to_cropping(index, x, y, width, height)
    
    def _fallback_to_cropping(self, index, x, y, width, height):
        """Fallback method using old cropping approach"""
        logger.info(f"🔄 Using fallback cropping for detection {index + 1}")
        
        # Fallback to old cropping method
        cropped_image = self.crop_detection_region(x, y, width, height)
        if cropped_image is not None:
            logger.info(f"Successfully cropped image for detection {index + 1}, starting FashionCLIP search...")
            # Find similar fashion items using FashionCLIP
            self.find_similar_fashion_items(cropped_image, index)
        else:
            logger.error(f"Failed to crop image for detection {index + 1} - cropped_image is None")
    
    def crop_detection_region(self, x, y, width, height):
        """Crop the detection region from the current video frame and apply person segmentation"""
        try:
            logger.info(f"crop_detection_region: Starting crop for region ({x}, {y}, {width}, {height})")
            
            # First try to use the saved frame from cache
            import os
            cached_frame_path = "cache/paused_frame.png"
            current_frame = None
            
            if os.path.exists(cached_frame_path):
                from PyQt5.QtGui import QPixmap
                current_frame = QPixmap(cached_frame_path)
                logger.info(f"crop_detection_region: Using cached frame from {cached_frame_path}")
            else:
                # Fallback: Get current frame from video player
                current_frame = self.video_player.get_current_frame()
                logger.info("crop_detection_region: Using current frame from video player")
            
            if current_frame is None or current_frame.isNull():
                logger.error("crop_detection_region: Failed to get current frame")
                return None
            
            logger.info(f"crop_detection_region: Got frame: {current_frame.width()}x{current_frame.height()}")
            
            # Step 1: Use ImageCropper to crop the bounding box region (existing implementation)
            cropped_pixmap = self.image_cropper.crop_bounding_box(current_frame, x, y, width, height)
            
            if cropped_pixmap is None or cropped_pixmap.isNull():
                logger.error(f"crop_detection_region: Failed to crop region - cropped_pixmap is None or null")
                return None
            
            logger.info(f"crop_detection_region: Successfully cropped region: ({x}, {y}, {width}, {height})")
            
            # Step 2: Apply person segmentation (new logic)
            segmented_pixmap = self.apply_person_segmentation(cropped_pixmap)
            
            if segmented_pixmap is not None:
                logger.info("crop_detection_region: Successfully applied person segmentation")
                return segmented_pixmap
            else:
                logger.warning("crop_detection_region: Segmentation failed, returning original crop")
                return cropped_pixmap
                
        except Exception as e:
            logger.error(f"crop_detection_region: Error cropping detection region: {e}")
            import traceback
            logger.error(f"crop_detection_region: Traceback: {traceback.format_exc()}")
            return None
    
    def apply_person_segmentation(self, cropped_pixmap):
        """Apply person segmentation to cropped image with white background"""
        try:
            logger.info("apply_person_segmentation: Starting person segmentation...")
            
            # Since we are doing segmentation on the full captured frame,
            # when 1 person is highlighted, simple cropping would do the job
            # No need to do segmentation again
            logger.info("apply_person_segmentation: Using simple cropping since full frame segmentation is already done")
            
            # Return the cropped pixmap directly without additional segmentation
            return cropped_pixmap
                
        except Exception as e:
            logger.error(f"apply_person_segmentation: Error applying segmentation: {e}")
            import traceback
            logger.error(f"apply_person_segmentation: Traceback: {traceback.format_exc()}")
            return None
    

    
    def find_similar_fashion_items(self, cropped_image, detection_index):
        """Find similar fashion items using FashionCLIP embeddings (Steps 4 & 5)"""
        try:
            logger.info(f"🎯 Step 4: Starting similarity search for person {detection_index + 1}")
            
            # Step 4: Verify embeddings are available
            if not hasattr(self.image_embedding, 'embeddings') or self.image_embedding.embeddings is None:
                logger.error("❌ Embeddings not pre-loaded - this should not happen!")
                logger.info("🔄 Attempting to load embeddings as fallback...")
                if not self.image_embedding.load_existing_embeddings():
                    logger.error("❌ Failed to load embeddings for similarity search")
                    return
            
            logger.info(f"✅ Embeddings loaded successfully ({self.image_embedding.embeddings.shape[0]} items)")
            
            # Step 4: Generate embedding for the mask image
            logger.info(f"🎯 Step 4: Computing embedding for person {detection_index + 1} mask image")
            query_embedding = self.image_embedding.compute_image_embedding(cropped_image)
            
            if query_embedding is None:
                logger.error("❌ Failed to compute embedding for mask image")
                return
            
            logger.info(f"✅ Successfully computed embedding vector (shape: {query_embedding.shape})")
            
            # Step 4: Perform similarity search
            logger.info(f"🎯 Step 4: Performing similarity search for person {detection_index + 1}")
            similar_items = self.image_embedding.find_similar_images(cropped_image, top_k=12)
            
            if similar_items:
                logger.info(f"✅ Step 4: Found {len(similar_items)} similar fashion items for person {detection_index + 1}")
                for item in similar_items[:3]:  # Log top 3 for brevity
                    logger.info(f"  🏆 Rank {item['rank']}: {os.path.basename(item['image_path'])} (similarity: {item['similarity']:.3f})")
                if len(similar_items) > 3:
                    logger.info(f"  ... and {len(similar_items) - 3} more items")
                
                # Step 5: Display results in UI grid
                logger.info(f"🎯 Step 5: Displaying fashion matches for person {detection_index + 1}")
                self.display_fashion_matches(similar_items, detection_index)
            else:
                logger.warning(f"⚠️ Step 4: No similar fashion items found for person {detection_index + 1}")
                
        except Exception as e:
            logger.error(f"❌ Error in similarity search for person {detection_index + 1}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def display_fashion_matches(self, similar_items, detection_index):
        """Display fashion matching results in UI grid (Step 5)"""
        try:
            logger.info(f"🎯 Step 5: Starting UI display for person {detection_index + 1}")
            logger.info(f"📊 Query: Person Detection {detection_index + 1}")
            logger.info(f"📊 Found {len(similar_items)} fashion matches")
            
            # Step 5: Clear previous results
            logger.info(f"🧹 Step 5: Clearing previous fashion matches")
            for label in self.row1_labels + self.row2_labels + self.row3_labels:
                label.clear()
                label.setText("No Match")
                label.setStyleSheet("color: #cccccc; font-size: 12px; border: 1px solid #ddd;")
            
            # Step 5: Display matched images in UI grid
            logger.info(f"🎨 Step 5: Displaying {min(len(similar_items), 9)} fashion matches in UI grid")
            displayed_count = 0
            
            for i, item in enumerate(similar_items[:9]):  # Limit to 9 images
                logger.info(f"🎨 #{item['rank']}: {os.path.basename(item['image_path'])} "
                           f"(similarity: {item['similarity']:.3f})")
                
                # Load and display the matched image
                if os.path.exists(item['image_path']):
                    pixmap = QPixmap(item['image_path'])
                    if not pixmap.isNull():
                        # Scale image to fit the larger label size
                        scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        
                        # Update the appropriate label with default styling
                        if i < 3:
                            # Row 1 (matches 1-3)
                            self.row1_labels[i].setPixmap(scaled_pixmap)
                            self.row1_labels[i].setText("")  # Clear text
                            self.row1_labels[i].setStyleSheet("border: 2px solid #ddd; border-radius: 6px; background-color: #f9f9f9;")
                            logger.info(f"🎨 Row 1, Position {i+1}: Displayed match #{item['rank']}")
                        elif i < 6:
                            # Row 2 (matches 4-6)
                            self.row2_labels[i - 3].setPixmap(scaled_pixmap)
                            self.row2_labels[i - 3].setText("")  # Clear text
                            self.row2_labels[i - 3].setStyleSheet("border: 2px solid #ddd; border-radius: 6px; background-color: #f9f9f9;")
                            logger.info(f"🎨 Row 2, Position {i-2}: Displayed match #{item['rank']}")
                        else:
                            # Row 3 (matches 7-9)
                            self.row3_labels[i - 6].setPixmap(scaled_pixmap)
                            self.row3_labels[i - 6].setText("")  # Clear text
                            self.row3_labels[i - 6].setStyleSheet("border: 2px solid #ddd; border-radius: 6px; background-color: #f9f9f9;")
                            logger.info(f"🎨 Row 3, Position {i-5}: Displayed match #{item['rank']}")
                        
                        displayed_count += 1
                    else:
                        logger.warning(f"❌ Failed to load image: {item['image_path']}")
                else:
                    logger.warning(f"❌ Image file not found: {item['image_path']}")
            
            # Step 5: Summary and completion
            logger.info(f"✅ Step 5: Successfully displayed {displayed_count} fashion matches for person {detection_index + 1}")
            logger.info(f"🎯 Step 5: UI grid updated with fashion similarity results")
            
            # Add summary information
            if similar_items:
                top_similarity = similar_items[0]['similarity']
                avg_similarity = sum(item['similarity'] for item in similar_items[:displayed_count]) / displayed_count
                logger.info(f"📊 Step 5: Top similarity: {top_similarity:.3f}, Average similarity: {avg_similarity:.3f}")
            
            logger.info(f"🎉 Steps 4 & 5 completed successfully for person {detection_index + 1}")
            
            # Log mask cache information
            cache_info = self.detection_coordinator.get_mask_cache_info()
            logger.info(f"📦 Mask cache status: {cache_info['cache_size']} items cached")
            
        except Exception as e:
            logger.error(f"❌ Error displaying fashion matches for person {detection_index + 1}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def navigate_detections(self, direction):
        """Navigate between detected objects"""
        if not self.detection_coordinates:
            return
        
        if direction == "next":
            self.current_focus_index = (self.current_focus_index + 1) % len(self.detection_coordinates)
        elif direction == "prev":
            self.current_focus_index = (self.current_focus_index - 1) % len(self.detection_coordinates)
        
        logger.info(f"Navigating to detection {self.current_focus_index + 1}/{len(self.detection_coordinates)}")
        self.show_detection_at_index(self.current_focus_index)
    
    def draw_single_bounding_box(self, index):
        """Draw a single bounding box for the specified detection index"""
        logger.info(f"draw_single_bounding_box: Called with index {index}")
        logger.info(f"draw_single_bounding_box: Detection mode: {self.is_in_detection_mode}, Coordinates count: {len(self.detection_coordinates)}")
        
        if not self.is_in_detection_mode or index < 0 or index >= len(self.detection_coordinates):
            # Clear bounding boxes by restoring original frame
            logger.info(f"draw_single_bounding_box: Clearing bounding boxes (index {index} out of range or not in detection mode)")
            if self.original_frame_pixmap:
                video_label = self.get_video_label()
                if video_label:
                    video_label.setPixmap(self.original_frame_pixmap.scaled(
                        video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    logger.info("draw_single_bounding_box: Cleared bounding boxes by restoring original frame")
                else:
                    logger.warning("draw_single_bounding_box: No video label found for clearing")
            else:
                logger.warning("draw_single_bounding_box: No original frame pixmap available for clearing")
            return
        
        # Get coordinates for the specified index
        x, y, width, height = self.detection_coordinates[index]
        logger.info(f"draw_single_bounding_box: Drawing box at ({x}, {y}) size {width}x{height}")
        
        # Use original frame if available, otherwise current frame
        source_pixmap = self.original_frame_pixmap if self.original_frame_pixmap else None
        if not source_pixmap:
            logger.warning("draw_single_bounding_box: No original frame pixmap, trying to get from video label")
            video_label = self.get_video_label()
            if video_label and video_label.pixmap():
                source_pixmap = video_label.pixmap()
                logger.info("draw_single_bounding_box: Got source pixmap from video label")
            else:
                logger.error("draw_single_bounding_box: No source pixmap available, cannot draw bounding box")
                return
        else:
            logger.info(f"draw_single_bounding_box: Using original frame pixmap: {source_pixmap.width()}x{source_pixmap.height()}")
        
        # Create a copy for drawing bounding box
        frame_with_box = source_pixmap.copy()
        logger.info(f"draw_single_bounding_box: Created copy for drawing: {frame_with_box.width()}x{frame_with_box.height()}")
        
        # Draw bounding box
        painter = QPainter(frame_with_box)
        pen = QPen(Qt.red, 5)  # Red stroke, 5px width for better visibility
        painter.setPen(pen)
        painter.drawRect(x, y, width, height)
        painter.end()
        logger.info(f"draw_single_bounding_box: Drew red rectangle at ({x}, {y}) size {width}x{height}")
        
        # Update video frame display
        video_label = self.get_video_label()
        if video_label:
            scaled_pixmap = frame_with_box.scaled(
                video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            video_label.setPixmap(scaled_pixmap)
            logger.info(f"draw_single_bounding_box: Updated video label with bounding box, scaled to {scaled_pixmap.width()}x{scaled_pixmap.height()}")
        else:
            logger.error("draw_single_bounding_box: No video label found to update with bounding box")
        
        # Crop the bounding box region from original frame
        self.crop_current_bounding_box(index, source_pixmap, x, y, width, height)
        
        logger.info(f"PopupWindow: Drew bounding box {index + 1}/{len(self.detection_coordinates)} at ({x}, {y}) size {width}x{height}")
    
    def get_video_label(self):
        """Get the QLabel widget that displays video frames"""
        if hasattr(self, 'video_label') and self.video_label:
            return self.video_label
        
        # Look for QLabel children in the video frame
        from PyQt5.QtWidgets import QLabel
        for child in self.video_frame.findChildren(QLabel):
            if child.pixmap() is not None:
                return child
        return None
    
    def crop_current_bounding_box(self, index, source_pixmap, x, y, width, height):
        """Crop the current bounding box region and store it"""
        try:
            # Crop the bounding box region
            cropped_pixmap = ImageCropper.crop_bounding_box(source_pixmap, x, y, width, height)
            
            if not cropped_pixmap.isNull():
                # Store cropped image (expand list if needed)
                while len(self.cropped_images) <= index:
                    self.cropped_images.append(None)
                
                self.cropped_images[index] = cropped_pixmap
                logger.info(f"PopupWindow: Cropped bounding box {index + 1} - size: {cropped_pixmap.width()}x{cropped_pixmap.height()}")
                
                # Optionally save cropped image for debugging/testing
                # Uncomment the following lines if you want to save cropped images to files
                # import os
                # os.makedirs("cropped_images", exist_ok=True)
                # ImageCropper.save_cropped_image(cropped_pixmap, f"cropped_images/crop_{index + 1}.png")
            else:
                logger.error(f"PopupWindow: Failed to crop bounding box {index + 1}")
                
        except Exception as e:
            logger.error(f"PopupWindow: Error cropping bounding box {index + 1}: {e}")
    
    def get_cropped_image(self, index):
        """Get the cropped image for a specific detection index"""
        if 0 <= index < len(self.cropped_images):
            return self.cropped_images[index]
        return None
    
    def navigate_bounding_boxes(self, key):
        """Navigate between detected objects using arrow keys (matching Kotlin's onItemFocusChanged)"""
        if not self.is_in_detection_mode or not self.detection_coordinates:
            return
        
        old_index = self.current_focus_index
        
        if key == Qt.Key_Left:
            # Navigate to previous item (with wrap-around)
            self.current_focus_index = (self.current_focus_index - 1) % len(self.detection_coordinates)
            logger.info(f"🔙 Left arrow - navigating from item {old_index + 1} to item {self.current_focus_index + 1}")
        elif key == Qt.Key_Right:
            # Navigate to next item (with wrap-around)
            self.current_focus_index = (self.current_focus_index + 1) % len(self.detection_coordinates)
            logger.info(f"➡️ Right arrow - navigating from item {old_index + 1} to item {self.current_focus_index + 1}")
        
        # Redraw bounding box and trigger FashionCLIP search for the newly focused item
        self.draw_single_bounding_box(self.current_focus_index)
        
        # Also trigger FashionCLIP search for the newly focused detection
        self.show_detection_at_index(self.current_focus_index)

    def on_quit_key(self):
        """Handle quit key press"""
        logger.info("Quit key pressed in pop-up window")
        self.close_window()
        
    def on_match_style_button(self):
        """Handle match style button click (same as space key)"""
        logger.info("🎯 Match Style button clicked - calling on_space_key")
        self.on_space_key()
    
    def on_previous_button(self):
        """Handle previous button click (same as left arrow)"""
        logger.info("Previous button clicked")
        self.navigate_bounding_boxes(Qt.Key_Left)
    
    def on_next_button(self):
        """Handle next button click (same as right arrow)"""
        logger.info("Next button clicked")
        self.navigate_bounding_boxes(Qt.Key_Right)
    
    def update_match_style_button_label(self):
        """Update the Match Style button label based on video state"""
        if hasattr(self, 'match_style_button') and self.video_player:
            if self.video_player.is_paused:
                self.match_style_button.setText("▶️ Resume Video")
                self.match_style_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: none;
                        padding: 12px 20px;
                        border-radius: 6px;
                        min-height: 45px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:pressed {
                        background-color: #21618c;
                    }
                """)
                logger.info("🎯 Button label updated to: Resume Video")
            else:
                self.match_style_button.setText("🎯 Match Style")
                self.match_style_button.setStyleSheet("""
                    QPushButton {
                        background-color: #27ae60;
                        color: white;
                        border: none;
                        padding: 12px 20px;
                        border-radius: 6px;
                        min-height: 45px;
                    }
                    QPushButton:hover {
                        background-color: #229954;
                    }
                    QPushButton:pressed {
                        background-color: #1e8449;
                    }
                """)
                logger.info("🎯 Button label updated to: Match Style")
    
    def close_window(self):
        """Close the pop-up window"""
        logger.info("Closing PyQt5 pop-up window")
        self.is_running = False
        self.close()
        
    def set_video_player(self, player):
        """Set reference to video player for control integration"""
        self.video_player = player
        # Initialize button label based on initial video state
        self.update_match_style_button_label()
        
    def get_video_frame(self):
        """Get the video frame widget for future GStreamer embedding"""
        return self.video_frame
        
    def on_video_embedded(self):
        """Called when video is successfully embedded"""
        self.video_placeholder.setText("🎬 Video Playing (Embedded)")
        self.video_placeholder.setStyleSheet("color: #27ae60; padding: 20px;")
        logger.info("Video embedding UI updated")
        
    def on_video_embedding_failed(self):
        """Called when video embedding fails"""
        self.video_placeholder.setText("⚠️ Video Embedding Failed - Check Logs")
        self.video_placeholder.setStyleSheet("color: #e74c3c; padding: 20px;")
        logger.warning("Video embedding failed - UI updated")

    def event(self, event):
        """Handle custom events including frame updates"""
        if event.type() == QEvent.User:
            # This is our frame update event
            self.process_frame_update()
            return True
        return super().event(event)
    
    def process_frame_update(self):
        """Process pending frame data and update the video display"""
        try:
            if hasattr(self.video_player, '_pending_frame'):
                frame_data = self.video_player._pending_frame
                if frame_data:
                    # Now we can safely create QPixmap on the main thread
                    from PyQt5.QtGui import QImage, QPixmap
                    
                    # Create QImage from raw RGB data
                    image = QImage(
                        frame_data['data'], 
                        frame_data['width'], 
                        frame_data['height'], 
                        frame_data['width'] * 3,  # bytes per line for RGB
                        QImage.Format_RGB888
                    )
                    if image.isNull():
                        logger.error("Failed to create QImage from buffer data")
                        return
                    
                    pixmap = QPixmap.fromImage(image)
                    if pixmap.isNull():
                        logger.error("Failed to create QPixmap from QImage")
                        return
                    
                    # Use fixed video frame size (no more dynamic resizing)
                    target_size = self.video_label.size() 

                    mode = (Qt.SmoothTransformation if (self.video_player and self.video_player.is_paused)
                            else Qt.FastTransformation)
                    scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, mode)
                    self.video_label.setPixmap(scaled)

                    # Hide the placeholder text since we now have video
                    if hasattr(self, 'video_placeholder') and self.video_placeholder is not None:
                        try:
                            self.video_placeholder.hide()
                        except RuntimeError:
                            # Widget already deleted, ignore
                            pass
        except Exception as e:
            logger.error(f"Error processing frame update: {e}")
            import traceback
            logger.error(f"Frame processing traceback: {traceback.format_exc()}")
                         
    def _add_test_boxes(self):
        """Add visual test boxes on right side to confirm viewport size"""
        if not hasattr(self, '_test_boxes'):
            from PyQt5.QtWidgets import QLabel
            
            # Get viewport dimensions
            viewport_width = self.video_frame.width()
            viewport_height = self.video_frame.height()
            
            # Calculate box dimensions (right half of viewport, split into 2 boxes)
            box_width = viewport_width // 2
            box_height = viewport_height // 2
            
            # Create first test box (top right)
            self._test_box1 = QLabel(self.video_frame)
            self._test_box1.setStyleSheet("background-color: #ff6b6b; border: 2px solid #ff0000;")
            self._test_box1.setAlignment(Qt.AlignCenter)
            self._test_box1.setGeometry(viewport_width // 2, 0, box_width, box_height)
            logger.info(f"Test box 1 position: ({viewport_width // 2}, 0), size: {box_width}x{box_height}")
            
            # Create second test box (bottom right)
            self._test_box2 = QLabel(self.video_frame)
            self._test_box2.setStyleSheet("background-color: #4ecdc4; border: 2px solid #00ff00;")
            self._test_box2.setAlignment(Qt.AlignCenter)
            self._test_box2.setGeometry(viewport_width // 2, box_height, box_width, box_height)
            logger.info(f"Test box 2 position: ({viewport_width // 2}, {box_height}), size: {box_width}x{box_height}")
            
            # Store reference for cleanup
            self._test_boxes = [self._test_box1, self._test_box2]
            
            # Show the boxes
            self._test_box1.show()
            self._test_box2.show()
            
            logger.info(f"Added test boxes - Viewport: {viewport_width}x{viewport_height}, Box size: {box_width}x{box_height}")
    
    def _remove_test_boxes(self):
        """Remove visual test boxes"""
        if hasattr(self, '_test_boxes'):
            for box in self._test_boxes:
                if box:
                    box.hide()
                    box.deleteLater()
            delattr(self, '_test_boxes')
            if hasattr(self, '_test_box1'):
                delattr(self, '_test_box1')
            if hasattr(self, '_test_box2'):
                delattr(self, '_test_box2')
            logger.info("Removed test boxes")

    def resize_video_display(self, half_size=False):
        """Resize the video display immediately"""
        if hasattr(self, 'video_label') and self.video_label:
            # Get current pixmap
            current_pixmap = self.video_label.pixmap()
            if current_pixmap and not current_pixmap.isNull():
                # Calculate target size
                if half_size:
                    target_size = self.video_frame.size() / 2
                    logger.info(f"Resizing video to HALF size: {target_size.width()}x{target_size.height()}")
                else:
                    target_size = self.video_frame.size()
                    logger.info(f"Resizing video to FULL size: {target_size.width()}x{target_size.height()}")
                
                # Scale and update
                scaled_pixmap = current_pixmap.scaled(
                    target_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                logger.info("Video display resized successfully")

    def debug_layout_sizes(self):
        """Debug method to output actual widget sizes"""
        try:
            video_width = self.video_frame.width()
            right_panel_width = self.right_box_widget.width()
            total_width = video_width + right_panel_width
            
            if total_width > 0:
                video_percentage = (video_width / total_width) * 100
                right_panel_percentage = (right_panel_width / total_width) * 100
                
                logger.info("=== LAYOUT DEBUG INFO ===")
                logger.info(f"Video frame width: {video_width}px ({video_percentage:.1f}%)")
                logger.info(f"Right panel width: {right_panel_width}px ({right_panel_percentage:.1f}%)")
                logger.info(f"Total layout width: {total_width}px")
                logger.info(f"Expected: Video=66.7%, Right=33.3%")
                logger.info("=== END DEBUG INFO ===")
            else:
                logger.info("Layout sizes not yet available (widgets not rendered)")
        except Exception as e:
            logger.error(f"Error debugging layout sizes: {e}")
    
    def switch_model(self, model_type):
        """
        Switch to a different YOLO model
        
        Args:
            model_type: Model type to switch to (yolov8s_seg or yolov8l_seg)
        """
        try:
            success = self.detection_coordinator.switch_model(model_type)
            if success:
                logger.info(f"✅ Successfully switched to model: {model_type}")
                # Update UI to reflect the change
                model_info = self.detection_coordinator.get_current_model_info()
                logger.info(f"🎯 Current model: {model_info['model_type']}")
                logger.info(f"🎯 Model path: {model_info['model_path']}")
            else:
                logger.error(f"❌ Failed to switch to model: {model_type}")
        except Exception as e:
            logger.error(f"❌ Error switching model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_current_model_info(self):
        """Get information about the current model"""
        return self.detection_coordinator.get_current_model_info()


class VideoPlayer:
    def __init__(self, video_path: str, embed_widget=None, loop_video=False):
        """Initialize the video player"""
        self.video_path = Path(video_path)
        self.pipeline = None
        self.is_playing = False
        self.is_paused = False
        self.video_sink = None
        self.is_fullscreen = True
        self.use_subprocess = False
        self.embed_widget = embed_widget  # PyQt5 widget to embed video into
        self.popup_window = None  # Reference to popup window
        self._frame_count = 0  # Counter for reducing debug log frequency
        self.loop_video = loop_video  # Enable/disable video looping
        self.loop_count = 0  # Track number of loops completed
        
        # Initialize GStreamer
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst, GLib
        self.Gst = Gst
        self.GLib = GLib
        
        # Initialize GStreamer
        if not self.Gst.is_initialized():
            self.Gst.init(None)
            logger.info("GStreamer initialized successfully")
        
        # Store pending frame data for main thread processing
        self._pending_frame = None

    def set_embed_widget(self, widget):
        """Set the PyQt5 widget to embed video into"""
        self.embed_widget = widget
        if widget:
            logger.info(f"Video frame widget: {widget}, visible: {widget.isVisible()}, winId: {widget.winId()}")
            logger.info("Video embedding widget set")

    def set_popup_window(self, popup_window):
        """Set reference to popup window for resize functionality"""
        self.popup_window = popup_window
        popup_window.set_video_player(self)
        logger.info("Popup window set")

    def create_embedded_pipeline(self):
        """Create embedded video pipeline for PyQt5 widget"""
        try:
            logger.info("Creating embedded video pipeline for PyQt5 widget...")
            
            # Use appsink for direct frame capture and rendering
            # Ensure normal playback speed with proper sync and framerate control
            pipeline_str = f"""
                filesrc location="{self.video_path}" ! 
                  decodebin name=dec ! 
                queue ! videorate ! video/x-raw,framerate=30/1 ! 
                synavideoconvertscale ! video/x-raw,format=RGB !  
                appsink name=videosink emit-signals=true sync=true
                dec. ! queue ! audioconvert ! audio/x-raw,rate=48000,channels=2 ! alsasink device=hw:0,7 sync=false
            """
            
            logger.info(f"Pipeline string: {pipeline_str.strip()}")
            
            # Create pipeline
            self.pipeline = self.Gst.parse_launch(pipeline_str.replace('\n', '').replace('  ', ' '))
            
            # Get the appsink element
            self.video_sink = self.pipeline.get_by_name("videosink")
            if not self.video_sink:
                logger.error("Failed to get video sink from pipeline")
                return False
            
            logger.info(f"Got appsink: {self.video_sink}")
            
            # Configure appsink for widget rendering with proper sync
            caps = self.Gst.Caps.from_string("video/x-raw,format=RGB,framerate=30/1")
            self.video_sink.set_property("caps", caps)
            self.video_sink.set_property("emit-signals", True)
            self.video_sink.set_property("max-buffers", 3)  # More buffers for smooth playback
            self.video_sink.set_property("drop", False)  # Don't drop frames to maintain sync
            self.video_sink.set_property("sync", True)  # Enable sync for normal playback speed
            
            logger.info("Appsink configured with properties")
            
            # Connect the new-sample signal to our callback
            self.video_sink.connect("new-sample", self._on_new_sample)
            logger.info("Connected new-sample signal to callback")
            
            # Set up bus message handling for EOS (End of Stream) events
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            logger.info("Connected bus message handler for EOS detection")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embedded video pipeline: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def play(self, embedded=False):
        """Start video playback"""
        try:
            if embedded:
                logger.info("Attempting to start embedded video playback...")
                logger.info("Creating embedded video pipeline with direct frame rendering...")
                if not self.create_embedded_pipeline():
                    logger.error("Failed to create embedded pipeline")
                    logger.error("Embedded video playback failed - no fallback mode")
                    if self.popup_window:
                        self.popup_window.on_video_embedding_failed()
                    return False
            else:
                if not self.create_pipeline():
                    logger.error("Failed to create pipeline")
                    return False
            
            # Start playback
            logger.info("Starting video playback...")
            ret = self.pipeline.set_state(self.Gst.State.PLAYING)
            if ret == self.Gst.StateChangeReturn.FAILURE:
                logger.error("Failed to start video playback")
                return False
            
            self.is_playing = True
            self.is_paused = False
            logger.info("Video playback started successfully")
            
            if embedded and self.popup_window:
                self.popup_window.on_video_embedded()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video playback: {e}")
            return False
    
    def pause(self):
        """Pause video playback"""
        # Python GStreamer approach - works for both Linux and macOS
        if self.pipeline and not self.is_paused:
            logger.info("Pausing video...")
            self.pipeline.set_state(self.Gst.State.PAUSED)
            self.is_paused = True
            logger.info("Video paused")
    
    def resume(self):
        """Resume video playback"""
        # Python GStreamer approach - works for both Linux and macOS
        if self.pipeline and self.is_paused:
            logger.info("Resuming video...")
            self.pipeline.set_state(self.Gst.State.PLAYING)
            self.is_paused = False
            logger.info("Video resumed")
    
    def toggle_pause(self):
        """Toggle between pause and resume"""
        if self.is_paused:
            self.resume()
        else:
            self.pause()
    
    def stop(self):
        """Stop video playback and cleanup"""
        if self.pipeline:
            logger.info("Stopping video...")
            self.pipeline.set_state(self.Gst.State.NULL)
            self.is_playing = False
            self.is_paused = False
            logger.info("Video stopped")
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages for errors and EOS"""
        try:
            msg_type = message.type
            
            if msg_type == self.Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                logger.error(f"GStreamer error: {err}, debug: {debug}")
            
            elif msg_type == self.Gst.MessageType.EOS:
                logger.info("🎬 End of stream reached")
                if self.loop_video:
                    self.loop_count += 1
                    logger.info(f"🔄 Video loop #{self.loop_count} completed - restarting...")
                    self._restart_video()
                else:
                    logger.info("📺 Video finished - stopping playback")
                    self.stop()
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling bus message: {e}")
            return True
    
    def _restart_video(self):
        """Restart video playback from the beginning"""
        try:
            logger.info("🔄 Restarting video playback...")
            
            # Stop current pipeline
            if self.pipeline:
                self.pipeline.set_state(self.Gst.State.NULL)
            
            # Create new pipeline
            if not self.create_embedded_pipeline():
                logger.error("Failed to recreate pipeline for loop")
                return False
            
            # Start playback
            ret = self.pipeline.set_state(self.Gst.State.PLAYING)
            if ret == self.Gst.StateChangeReturn.FAILURE:
                logger.error("Failed to restart video playback")
                return False
            
            self.is_playing = True
            self.is_paused = False
            logger.info(f"✅ Video restarted successfully (loop #{self.loop_count})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error restarting video: {e}")
            import traceback
            logger.error(f"Restart traceback: {traceback.format_exc()}")
            return False
    
    def get_current_frame(self):
        """Get current video frame as QPixmap for cropping"""
        try:
            if hasattr(self, '_pending_frame') and self._pending_frame:
                frame_data = self._pending_frame
                
                # Create QImage from raw RGB data
                from PyQt5.QtGui import QImage, QPixmap
                
                # Create QImage from raw RGB data
                image = QImage(
                    frame_data['data'], 
                    frame_data['width'], 
                    frame_data['height'], 
                    frame_data['width'] * 3,  # bytes per line for RGB
                    QImage.Format_RGB888
                )
                
                # Convert to QPixmap
                pixmap = QPixmap.fromImage(image)
                logger.info(f"Retrieved current frame: {pixmap.width()}x{pixmap.height()}")
                return pixmap
            else:
                logger.warning("No current frame data available")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current frame: {e}")
            return None

    def _on_new_sample(self, sink):
        """Callback to render video frames directly to PyQt5 widget"""
        try:
            self._frame_count += 1
            
            # Get the sample from the sink
            sample = sink.emit("pull-sample")
            if not sample:
                logger.error("Failed to pull sample from sink")
                return self.Gst.FlowReturn.ERROR
            
            # Get the buffer from the sample
            buffer = sample.get_buffer()
            if not buffer:
                logger.error("Failed to get buffer from sample")
                return self.Gst.FlowReturn.ERROR
            
            # Get the caps (format information)
            caps = sample.get_caps()
            if not caps:
                logger.error("Failed to get caps from sample")
                return self.Gst.FlowReturn.ERROR
            
            # Extract format information
            structure = caps.get_structure(0)
            width = structure.get_int("width")[1]
            height = structure.get_int("height")[1]
            
            # Map the buffer to get raw data
            success, map_info = buffer.map(self.Gst.MapFlags.READ)
            if not success:
                logger.error("Failed to map buffer")
                return self.Gst.FlowReturn.ERROR
            
            try:
                # Copy the raw image data (since buffer will be unmapped)
                image_data = bytes(map_info.data)
                
                # Store frame data for main thread processing
                self._pending_frame = {
                    'data': image_data,
                    'width': width,
                    'height': height
                }
                
                # Use QApplication.postEvent to ensure main thread execution
                from PyQt5.QtWidgets import QApplication
                from PyQt5.QtCore import QEvent
                
                class FrameUpdateEvent(QEvent):
                    def __init__(self):
                        super().__init__(QEvent.User)
                
                # Post event to main thread via popup window
                if QApplication.instance() and self.popup_window:
                    QApplication.postEvent(self.popup_window, FrameUpdateEvent())
                else:
                    logger.warning("Cannot post frame event - no QApplication or popup window reference")
                
            finally:
                buffer.unmap(map_info)
            
            return self.Gst.FlowReturn.OK
            
        except Exception as e:
            logger.error(f"Error in video frame callback: {e}")
            import traceback
            logger.error(f"Callback traceback: {traceback.format_exc()}")
            return self.Gst.FlowReturn.ERROR


def simple_keyboard_monitor(player):
    """Simple keyboard monitoring for macOS when NSRunLoop is already running"""
    import select
    import sys
    
    logger.info("Keyboard monitoring active. Press 'q' to quit, SPACE to pause/resume")
    
    while player.is_playing:
        # Check if there's input available (non-blocking)
        if select.select([sys.stdin], [], [], 0.1)[0]:
            try:
                key = sys.stdin.read(1).lower()
                if key == 'q':
                    logger.info("Quit key pressed")
                    player.stop()
                    break
                elif key == ' ':
                    logger.info("Space key pressed - toggling pause")
                    player.toggle_pause()
            except:
                # Handle any input errors gracefully
                pass
        
        time.sleep(0.1)


def main():
    """Main function to run the video player with PyQt5 pop-up window"""
    
    # Parse command line arguments
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Shopping Demo with Qt-embedded Video Player')
    parser.add_argument('--video', '-v', 
                       default='samples/clip.mp4',
                       help='Path to video file relative to current directory (default: samples/clip.mp4)')
    parser.add_argument('--model', '-m',
                       choices=['yolov8s_seg', 'yolov8l_seg'],
                       default='yolov8s_seg',
                       help='YOLO model to use for detection (default: yolov8s_seg)')
    parser.add_argument('--loop', '-l',
                       action='store_true',
                       help='Enable video looping (video will restart automatically when finished)')
    
    args = parser.parse_args()
    
    # Set up video path (relative to current directory where shopping.py is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, args.video)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        logger.error(f"Please check the path relative to: {script_dir}")
        logger.error("Available video files in samples/:")
        samples_dir = os.path.join(script_dir, "samples")
        if os.path.exists(samples_dir):
            for f in os.listdir(samples_dir):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    logger.error(f"  - samples/{f}")
        return 1
    
    logger.info(f"Using video file: {video_path}")
    
    try:
        # Initialize GStreamer first
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        logger.info("GStreamer initialized successfully")
        
        # Create PyQt5 application on main thread
        qt_app = QApplication(sys.argv)
        logger.info("PyQt5 QApplication created on main thread")
        
        # Create and show PyQt5 pop-up window on main thread
        logger.info("Creating PyQt5 pop-up window on main thread...")
        logger.info(f"Using model: {args.model}")
        popup_window = PopupWindow(model_type=args.model)
        popup_window.show()
        logger.info("PyQt5 pop-up window created successfully")
        
        # Connect button signals
        popup_window.match_style_button.clicked.connect(popup_window.on_match_style_button)
        popup_window.previous_button.clicked.connect(popup_window.on_previous_button)
        popup_window.next_button.clicked.connect(popup_window.on_next_button)
        popup_window.close_button.clicked.connect(popup_window.close_window)
        
        # Set window properties
        popup_window.setWindowTitle("🎬 Video Shopping Demo")
        
        # Debug layout sizes after window is shown
        QTimer.singleShot(100, popup_window.debug_layout_sizes)  # Delay to ensure widgets are rendered
        
        logger.info("PyQt5 pop-up window created successfully")
        
        # Create video player
        logger.info(f"Creating video player for: {video_path}")
        logger.info(f"Loop mode: {'enabled' if args.loop else 'disabled'}")
        player = VideoPlayer(video_path, loop_video=args.loop)
        
        # Set up video embedding
        player.set_embed_widget(popup_window.get_video_frame())
        player.set_popup_window(popup_window)
        
        # Start embedded video playback
        if not player.play(embedded=True):
            logger.error("Failed to start embedded video playback")
            return 1
        
        loop_status = "enabled" if args.loop else "disabled"
        logger.info("Application ready. Controls: SPACE: pause/resume, q: quit, +: plus key (in popup)")
        logger.info(f"Video looping: {loop_status}")
        logger.info("Starting PyQt5 event loop on main thread...")
        
        # Start Qt event loop (this blocks until application exits)
        qt_app.exec_()
        
        # Cleanup
        player.stop()
        logger.info("Application exited cleanly")
        return 0
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    logger.info("🛍️ Starting Shopping Demo with Qt-embedded video...")
    logger.info("📝 Usage: python shopping.py [--video path/to/video.mp4] [--model yolov8s_seg|yolov8l_seg] [--loop]")
    logger.info("📝 Example: python shopping.py --video samples/clip_1.mp4 --model yolov8s_seg --loop")
    logger.info("📝 Available models: yolov8s_seg (faster), yolov8l_seg (more accurate)")
    logger.info("📝 Use --loop to enable video looping (video restarts automatically when finished)")
    sys.exit(main())
