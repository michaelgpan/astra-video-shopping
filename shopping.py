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
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QColor
import os
from object_detection import DetectionCoordinator, ImageCropper
from image_embedding import ImageEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PopupWindow(QMainWindow):
    def __init__(self):
        """Initialize the PyQt5-based pop-up window"""
        super().__init__()
        self.video_player = None
        self.is_running = False
        
        # Initialize object detection coordinator
        self.detection_coordinator = DetectionCoordinator()
        self.image_cropper = ImageCropper()  # Initialize ImageCropper for bounding box cropping
        self.last_detection_results = None
        
        # Initialize FashionCLIP embedding system 
        # Note: Models and images should be prepared using embedding_model_prepare.py
        logger.info("Loading pre-prepared FashionCLIP embedding system...")
        self.image_embedding = ImageEmbedding()
        
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
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title section
        title_label = QLabel("🎬 Video Shopping Demo")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # Create fixed split-screen layout container
        self.video_container_layout = QHBoxLayout()
        
        # Video frame setup (left side)
        self.video_frame = QFrame()
        self.video_frame.setMinimumHeight(200)
        self.video_frame.setStyleSheet("background-color: #34495e; border: 2px dashed #7f8c8d; border-radius: 10px;")
        
        # Make the video frame focusable and ready for video embedding
        self.video_frame.setFocusPolicy(Qt.StrongFocus)
        self.video_frame.setAttribute(Qt.WA_NativeWindow, True)  # Important for video embedding
        
        video_layout = QVBoxLayout(self.video_frame)
        self.video_placeholder = QLabel("📺 GStreamer Video Will Be Embedded Here")
        self.video_placeholder.setFont(QFont("Arial", 12))
        self.video_placeholder.setAlignment(Qt.AlignCenter)
        self.video_placeholder.setStyleSheet("color: #bdc3c7; padding: 20px;")
        video_layout.addWidget(self.video_placeholder)
        
        # Right-side panels (always visible) - 3x4 grid layout
        self.right_box_container = QVBoxLayout()
        
        # Row 1: Fashion match placeholders (will be updated with FashionCLIP results)
        self.row1_frame = QFrame()
        self.row1_frame.setStyleSheet("background-color: #ffffff; border: 2px solid #cccccc; border-radius: 10px;")
        self.row1_frame.setMinimumHeight(120)
        
        row1_layout = QHBoxLayout(self.row1_frame)
        row1_layout.setContentsMargins(10, 10, 10, 10)  # Same margins as other rows
        row1_layout.setSpacing(10)  # Same spacing as other rows
        
        # Row 1 image labels (for FashionCLIP matches 1-4)
        self.row1_labels = []
        for i in range(4):
            label = QLabel(f"Match {i+1}")
            label.setStyleSheet("color: #cccccc; font-size: 12px; border: 1px solid #ddd;")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(120, 120)  # Same size as other rows
            label.setMaximumSize(200, 200)  # Set maximum size to prevent expansion
            label.setScaledContents(True)  # Allow image scaling
            row1_layout.addWidget(label)
            self.row1_labels.append(label)
        
        # Row 2: Fashion match placeholders (will be updated with FashionCLIP results)
        self.row2_frame = QFrame()
        self.row2_frame.setStyleSheet("background-color: #ffffff; border: 2px solid #cccccc; border-radius: 10px;")
        self.row2_frame.setMinimumHeight(120)
        
        row2_layout = QHBoxLayout(self.row2_frame)
        row2_layout.setContentsMargins(10, 10, 10, 10)  # Same margins as row 1
        row2_layout.setSpacing(10)  # Same spacing as row 1
        
        # Row 2 image labels (for FashionCLIP matches 5-8)
        self.row2_labels = []
        for i in range(4):
            label = QLabel(f"Match {i+5}")
            label.setStyleSheet("color: #cccccc; font-size: 12px; border: 1px solid #ddd;")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(120, 120)  # Same size as row 1
            label.setMaximumSize(200, 200)  # Set maximum size to prevent expansion
            label.setScaledContents(True)  # Allow image scaling
            row2_layout.addWidget(label)
            self.row2_labels.append(label)  
        
        # Row 3: Fashion match placeholders (will be updated with FashionCLIP results)
        self.row3_frame = QFrame()
        self.row3_frame.setStyleSheet("background-color: #ffffff; border: 2px solid #cccccc; border-radius: 10px;")
        self.row3_frame.setMinimumHeight(120)
        
        row3_layout = QHBoxLayout(self.row3_frame)
        row3_layout.setContentsMargins(10, 10, 10, 10)  # Same margins as row 1
        row3_layout.setSpacing(10)  # Same spacing as row 1
        
        # Row 3 image labels (for FashionCLIP matches 9-12)
        self.row3_labels = []
        for i in range(4):
            label = QLabel(f"Match {i+9}")
            label.setStyleSheet("color: #cccccc; font-size: 12px; border: 1px solid #ddd;")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(120, 120)  # Same size as row 1
            label.setMaximumSize(200, 200)  # Set maximum size to prevent expansion
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
        
        # Set up fixed split-screen layout: video left, panels right
        self.video_container_layout.addWidget(self.video_frame, 2)  # Video takes 2/3 of space
        self.video_container_layout.addWidget(self.right_box_widget, 1)  # Panels take 1/3 of space
        
        main_layout.addLayout(self.video_container_layout)
        
        # Buttons section
        button_layout = QHBoxLayout()
        
        self.test_button = QPushButton("🧪 Test Button")
        self.test_button.setFont(QFont("Arial", 10))
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.test_button.clicked.connect(self.on_test_button)
        button_layout.addWidget(self.test_button)
        
        self.close_button = QPushButton("❌ Close Window")
        self.close_button.setFont(QFont("Arial", 10))
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.close_button.clicked.connect(self.close_window)
        button_layout.addWidget(self.close_button)
        
        main_layout.addWidget(QWidget())  # Spacer
        main_layout.addLayout(button_layout)
        
        # Set window properties
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        
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
            
            # Trigger object detection when video is paused
            if self.video_player.is_paused:
                logger.info("🔍 Video paused - triggering object detection...")
                
                # Debug layout sizes when paused (when user is actually looking at it)
                self.debug_layout_sizes()
                
                self.detection_coordinator.on_video_paused(
                    self.video_frame, 
                    callback=self.on_detection_complete
                )
            else:
                # Reset detection when video resumes
                self.detection_coordinator.reset()
                logger.info("🔄 Video resumed - detection reset")
                self.is_in_detection_mode = False
                self.detection_coordinates = []
                self.current_focus_index = 0
                self.draw_single_bounding_box(-1)  # Clear bounding box
    
    def on_detection_complete(self, results):
        """Handle detection completion - store results and show first bounding box"""
        coordinates = results.get('coordinates', [])
        detections = results.get('detections', [])
        
        logger.info(f"PopupWindow: Detection completed with {len(coordinates)} objects detected")
        
        # Store detection results
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
        
        # Log detailed detection results
        for i, (x, y, width, height) in enumerate(coordinates):
            logger.info(f"PopupWindow: Object {i + 1} at ({x}, {y}) size {width}x{height}")
        
        for detection in detections:
            logger.info(f"PopupWindow: Detected {detection.label} (class {detection.class_index}) at ({detection.x}, {detection.y})")
        
        # Integrate FashionCLIP similarity search with object detection
        self.show_detection_at_index(0)
    
    def show_detection_at_index(self, index):
        """Show detection and find similar fashion items"""
        if not self.detection_coordinates or index >= len(self.detection_coordinates):
            logger.warning(f"Cannot show detection {index}: no coordinates available")
            return
        
        x, y, width, height = self.detection_coordinates[index]
        logger.info(f"Showing detection {index + 1}: ({x}, {y}, {width}, {height})")
        
        # Crop the detected region from current frame
        logger.info(f"Attempting to crop detection region for index {index}")
        cropped_image = self.crop_detection_region(x, y, width, height)
        if cropped_image is not None:
            logger.info(f"Successfully cropped image for detection {index + 1}, starting FashionCLIP search...")
            # Find similar fashion items using FashionCLIP
            self.find_similar_fashion_items(cropped_image, index)
        else:
            logger.error(f"Failed to crop image for detection {index + 1} - cropped_image is None")
    
    def crop_detection_region(self, x, y, width, height):
        """Crop the detection region from the current video frame"""
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
            
            # Use ImageCropper to crop the bounding box region
            cropped_pixmap = self.image_cropper.crop_bounding_box(current_frame, x, y, width, height)
            
            if cropped_pixmap is not None and not cropped_pixmap.isNull():
                logger.info(f"crop_detection_region: Successfully cropped region: ({x}, {y}, {width}, {height})")
                return cropped_pixmap
            else:
                logger.error(f"crop_detection_region: Failed to crop region - cropped_pixmap is None or null")
                return None
                
        except Exception as e:
            logger.error(f"crop_detection_region: Error cropping detection region: {e}")
            import traceback
            logger.error(f"crop_detection_region: Traceback: {traceback.format_exc()}")
            return None
    
    def find_similar_fashion_items(self, cropped_image, detection_index):
        """Find similar fashion items using FashionCLIP embeddings"""
        try:
            logger.info("Searching for similar fashion items...")
            
            # Embeddings should already be pre-loaded during initialization
            if not hasattr(self.image_embedding, 'embeddings') or self.image_embedding.embeddings is None:
                logger.error("Embeddings not pre-loaded - this should not happen!")
                logger.info("Attempting to load embeddings as fallback...")
                if not self.image_embedding.load_existing_embeddings():
                    logger.error("Failed to load embeddings for similarity search")
                    return
            
            # Find top 12 similar images
            similar_items = self.image_embedding.find_similar_images(cropped_image, top_k=12)
            
            if similar_items:
                logger.info(f"Found {len(similar_items)} similar fashion items:")
                for item in similar_items:
                    logger.info(f"  Rank {item['rank']}: {item['image_path']} (similarity: {item['similarity']:.3f})")
                
                # Display results in UI grid
                self.display_fashion_matches(similar_items, detection_index)
            else:
                logger.warning("No similar fashion items found")
                
        except Exception as e:
            logger.error(f"Error finding similar fashion items: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def display_fashion_matches(self, similar_items, detection_index):
        """Display fashion matching results in UI grid (all 3 rows)"""
        try:
            logger.info("=== FASHION MATCHING RESULTS ===")
            logger.info(f"Query: Detection {detection_index + 1}")
            logger.info(f"Found {len(similar_items)} matches:")
            
            # Clear previous images first
            for label in self.row1_labels + self.row2_labels + self.row3_labels:
                label.clear()
                label.setText("No Match")
                label.setStyleSheet("color: #cccccc; font-size: 12px; border: 1px solid #ddd;")
            
            # Display up to 12 matched images in the grid
            for i, item in enumerate(similar_items[:12]):  # Limit to 12 images
                logger.info(f"#{item['rank']}: {os.path.basename(item['image_path'])} "
                           f"(similarity: {item['similarity']:.3f})")
                
                # Load and display the matched image
                if os.path.exists(item['image_path']):
                    pixmap = QPixmap(item['image_path'])
                    if not pixmap.isNull():
                        # Scale image to fit the smaller label size
                        scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        
                        # Update the appropriate label
                        if i < 4:
                            # Row 1 (matches 1-4)
                            self.row1_labels[i].setPixmap(scaled_pixmap)
                            self.row1_labels[i].setText("")  # Clear text
                            self.row1_labels[i].setStyleSheet("border: 2px solid #4CAF50; border-radius: 5px;")
                        elif i < 8:
                            # Row 2 (matches 5-8)
                            self.row2_labels[i - 4].setPixmap(scaled_pixmap)
                            self.row2_labels[i - 4].setText("")  # Clear text
                            self.row2_labels[i - 4].setStyleSheet("border: 2px solid #4CAF50; border-radius: 5px;")
                        else:
                            # Row 3 (matches 9-12)
                            self.row3_labels[i - 8].setPixmap(scaled_pixmap)
                            self.row3_labels[i - 8].setText("")  # Clear text
                            self.row3_labels[i - 8].setStyleSheet("border: 2px solid #4CAF50; border-radius: 5px;")
                    else:
                        logger.warning(f"Failed to load image: {item['image_path']}")
                else:
                    logger.warning(f"Image file not found: {item['image_path']}")
            
            logger.info("=== END RESULTS ===")
            logger.info(f"Displayed {min(len(similar_items), 12)} fashion matches in UI grid")
            
        except Exception as e:
            logger.error(f"Error displaying fashion matches: {e}")
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
        pen = QPen(Qt.white, 5)  # White stroke, 5px width (matching Kotlin)
        painter.setPen(pen)
        painter.drawRect(x, y, width, height)
        painter.end()
        logger.info(f"draw_single_bounding_box: Drew white rectangle at ({x}, {y}) size {width}x{height}")
        
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
        if hasattr(self, '_video_label') and self._video_label:
            return self._video_label
        
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
        
    def on_test_button(self):
        """Handle test button click"""
        logger.info("Test button clicked")
        logger.info("Test button clicked! ✅")
        
    def close_window(self):
        """Close the pop-up window"""
        logger.info("Closing PyQt5 pop-up window")
        self.is_running = False
        self.close()
        
    def set_video_player(self, player):
        """Set reference to video player for control integration"""
        self.video_player = player
        
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
                    target_size = self.video_frame.size()
                    
                    # Update the video frame widget
                    if hasattr(self.video_frame, 'setPixmap'):
                        # If it's a QLabel, set pixmap directly
                        scaled_pixmap = pixmap.scaled(
                            target_size, 
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        self.video_frame.setPixmap(scaled_pixmap)
                        
                    else:
                        # If it's a QFrame, we need to add a QLabel child
                        if not hasattr(self, '_video_label'):
                            from PyQt5.QtWidgets import QLabel, QVBoxLayout
                            logger.info("Creating video label for QFrame")
                            self._video_label = QLabel()
                            self._video_label.setStyleSheet("background-color: black;")
                            
                            # Clear existing layout if any
                            if self.video_frame.layout():
                                QWidget().setLayout(self.video_frame.layout())
                            
                            layout = QVBoxLayout(self.video_frame)
                            layout.setContentsMargins(0, 0, 0, 0)
                            layout.addWidget(self._video_label)
                            self.video_frame.setLayout(layout)
                        
                        scaled_pixmap = pixmap.scaled(
                            target_size, 
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        self._video_label.setPixmap(scaled_pixmap)
                        
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
        if hasattr(self, '_video_label') and self._video_label:
            # Get current pixmap
            current_pixmap = self._video_label.pixmap()
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
                self._video_label.setPixmap(scaled_pixmap)
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


class VideoPlayer:
    def __init__(self, video_path: str, embed_widget=None):
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
            # Let GStreamer auto-detect the original framerate (could be 60fps)
            pipeline_str = f"""
                filesrc location="{self.video_path}" ! 
                decodebin ! 
                videoconvert ! 
                videoscale ! 
                video/x-raw,format=RGB,width=640,height=480 ! 
                appsink name=videosink emit-signals=true max-buffers=2 drop=true sync=false
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
            
            # Configure appsink for widget rendering - auto-detect framerate
            caps = self.Gst.Caps.from_string("video/x-raw,format=RGB")
            self.video_sink.set_property("caps", caps)
            self.video_sink.set_property("emit-signals", True)
            self.video_sink.set_property("max-buffers", 2)  # Slightly more buffers for 60fps
            self.video_sink.set_property("drop", True)
            self.video_sink.set_property("sync", False)  # Disable sync for smoother playback
            
            logger.info("Appsink configured with properties")
            
            # Connect the new-sample signal to our callback
            self.video_sink.connect("new-sample", self._on_new_sample)
            logger.info("Connected new-sample signal to callback")
            
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
    
    # Set up video path
    video_path = "samples/clip.mp4"
    
    try:
        # macOS GUI context setup
        logger.info("Setting up macOS GUI context...")
        
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
        popup_window = PopupWindow()
        popup_window.show()
        logger.info("PyQt5 pop-up window created successfully")
        
        # Connect button signals
        popup_window.test_button.clicked.connect(popup_window.on_test_button)
        popup_window.close_button.clicked.connect(popup_window.close_window)
        
        # Set window properties
        popup_window.setWindowTitle("🎬 Video Shopping Demo")
        popup_window.setGeometry(100, 100, 800, 600)
        popup_window.show()
        
        # Debug layout sizes after window is shown
        QTimer.singleShot(100, popup_window.debug_layout_sizes)  # Delay to ensure widgets are rendered
        
        logger.info("PyQt5 pop-up window created successfully")
        
        # Create video player
        logger.info(f"Creating video player for: {video_path}")
        player = VideoPlayer(video_path)
        
        # Set up video embedding
        player.set_embed_widget(popup_window.get_video_frame())
        player.set_popup_window(popup_window)
        
        # Start embedded video playback
        if not player.play(embedded=True):
            logger.error("Failed to start embedded video playback")
            return 1
        
        logger.info("Application ready. Controls: SPACE: pause/resume, q: quit, +: plus key (in popup)")
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
    sys.exit(main())
