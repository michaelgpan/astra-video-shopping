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
            import sys
            sys.exit(1)
        
        # Pre-load embeddings for fast similarity search during video playback
        logger.info("Loading pre-computed FashionCLIP embeddings...")
        if not self.image_embedding.load_existing_embeddings():
            logger.error("❌ Failed to load embeddings!")
            logger.error("🔧 Please run the preparation script:")
            logger.error("   python embedding_model_prepare.py")
            import sys
            sys.exit(1)
        else:
            logger.info("✅ FashionCLIP model and embeddings loaded successfully - first match will be fast!")
        
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
        
        # Control buttons section (4 buttons for video control)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)  # Add spacing between buttons
        
        # Common button style
        button_style = """
            QPushButton {
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 12px;
                min-width: 120px;
            }
            QPushButton:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            QPushButton:pressed {
                transform: translateY(0px);
            }
        """
        
        # 1. Play/Pause button (Space key function)
        self.play_pause_button = QPushButton("⏸️ Pause")
        self.play_pause_button.setFont(QFont("Arial", 11))
        self.play_pause_button.setStyleSheet(button_style + """
            QPushButton {
                background-color: #27ae60;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.play_pause_button.clicked.connect(self.on_play_pause_button)
        button_layout.addWidget(self.play_pause_button)
        
        # 2. Previous detection button (Left arrow key function)
        self.prev_button = QPushButton("⬅️ Previous")
        self.prev_button.setFont(QFont("Arial", 11))
        self.prev_button.setStyleSheet(button_style + """
            QPushButton {
                background-color: #f39c12;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)
        self.prev_button.clicked.connect(self.on_prev_button)
        button_layout.addWidget(self.prev_button)
        
        # 3. Next detection button (Right arrow key function)
        self.next_button = QPushButton("➡️ Next")
        self.next_button.setFont(QFont("Arial", 11))
        self.next_button.setStyleSheet(button_style + """
            QPushButton {
                background-color: #3498db;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.next_button.clicked.connect(self.on_next_button)
        button_layout.addWidget(self.next_button)
        
        # 4. Exit button (Q key function)
        self.exit_button = QPushButton("❌ Exit")
        self.exit_button.setFont(QFont("Arial", 11))
        self.exit_button.setStyleSheet(button_style + """
            QPushButton {
                background-color: #e74c3c;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.exit_button.clicked.connect(self.on_exit_button)
        button_layout.addWidget(self.exit_button)
        
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
                logger.error("⚠️ Embeddings not pre-loaded - this should not happen with new initialization!")
                logger.error("This indicates an initialization problem - fashion matching may fail")
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
        
        # Update video frame display - NO SCALING to preserve coordinate accuracy
        video_label = self.get_video_label()
        if video_label:
            # Set the pixmap directly without scaling to preserve 1280x720 coordinates
            video_label.setPixmap(frame_with_box)
            logger.info(f"draw_single_bounding_box: Updated video label with bounding box (no scaling): {frame_with_box.width()}x{frame_with_box.height()}")
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
        
    def on_play_pause_button(self):
        """Handle play/pause button click (Space key function)"""
        logger.info("🎮 Play/Pause button clicked")
        if hasattr(self, 'video_player') and self.video_player:
            self.video_player.toggle_pause()
            # Update button text based on current state
            if self.video_player.is_paused:
                self.play_pause_button.setText("▶️ Play")
            else:
                self.play_pause_button.setText("⏸️ Pause")
        else:
            logger.warning("No video player available for play/pause control")
    
    def on_prev_button(self):
        """Handle previous button click (Left arrow key function)"""
        logger.info("🎮 Previous button clicked")
        # Simulate left arrow key press
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QKeyEvent
        from PyQt5.QtWidgets import QApplication
        
        key_event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Left, Qt.NoModifier)
        QApplication.postEvent(self, key_event)
    
    def on_next_button(self):
        """Handle next button click (Right arrow key function)"""
        logger.info("🎮 Next button clicked")
        # Simulate right arrow key press
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QKeyEvent
        from PyQt5.QtWidgets import QApplication
        
        key_event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
        QApplication.postEvent(self, key_event)
    
    def on_exit_button(self):
        """Handle exit button click (Q key function)"""
        logger.info("🎮 Exit button clicked")
        # Simulate Q key press or directly quit
        if hasattr(self, 'video_player') and self.video_player:
            logger.info("Stopping video player...")
        QApplication.instance().quit()
        
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
    
    def display_captured_frame(self, frame_path):
        """Display captured frame from file path in the video frame"""
        try:
            logger.info(f"Loading captured frame from: {frame_path}")
            
            # Clear any existing video label to ensure fresh display
            if hasattr(self, '_video_label') and self._video_label:
                self._video_label.clear()
                logger.info("Cleared previous frame display")
            
            # Load the captured frame
            from PyQt5.QtGui import QPixmap
            pixmap = QPixmap(frame_path)
            
            if pixmap.isNull():
                logger.error(f"Failed to load image from: {frame_path}")
                return False
            
            logger.info(f"Loaded frame: {pixmap.width()}x{pixmap.height()}")
            
            # Use existing method to show the frame
            self.show_with_captured_frame(pixmap)
            
            return True
            
        except Exception as e:
            logger.error(f"Error displaying captured frame: {e}")
            import traceback
            logger.error(f"Display frame traceback: {traceback.format_exc()}")
            return False
    
    def display_captured_pixmap(self, pixmap):
        """Display captured QPixmap directly from memory and trigger object detection"""
        try:
            logger.info(f"🖼️ Displaying captured frame from memory: {pixmap.width()}x{pixmap.height()}")
            logger.info(f"🖼️ Pixmap device pixel ratio: {pixmap.devicePixelRatio()}")
            logger.info(f"🖼️ Pixmap depth: {pixmap.depth()}")
            
            # Clear any existing video label to ensure fresh display
            if hasattr(self, '_video_label') and self._video_label:
                self._video_label.clear()
                logger.info("Cleared previous frame display")
            
            # Use existing method to show the pixmap
            self.show_with_captured_frame(pixmap)
            
            # Trigger object detection on the captured frame
            if self.image_embedding:
                logger.info("🎯 Starting object detection on captured frame...")
                self.run_detection_on_pixmap(pixmap)
            else:
                logger.warning("⚠️ Image embedding model not available for object detection")
            
            return True
            
        except Exception as e:
            logger.error(f"Error displaying captured pixmap: {e}")
            import traceback
            logger.error(f"Display pixmap traceback: {traceback.format_exc()}")
            return False
    
    def run_detection_on_pixmap(self, pixmap):
        """Run object detection on QPixmap directly from memory"""
        try:
            import tempfile
            import os
            
            # Save pixmap to temporary file for object detection
            # (Object detection service expects file path)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Save pixmap to temporary file
            if pixmap.save(temp_path, 'PNG'):
                logger.info(f"💾 Saved frame to temp file for detection: {temp_path}")
                
                # Use ObjectDetectionService directly for more control
                from object_detection import ObjectDetectionService
                detection_service = ObjectDetectionService()
                
                # Define callback for detection completion
                def detection_callback(json_result):
                    try:
                        # Clean up temp file first
                        try:
                            os.unlink(temp_path)
                            logger.info("🗑️ Cleaned up temporary detection file")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temp file: {cleanup_error}")
                        
                        # Process detection results
                        self._on_detection_complete_pixmap(json_result)
                        
                    except Exception as e:
                        logger.error(f"Error in detection callback: {e}")
                
                # Run detection asynchronously
                model_path = "/usr/share/synap/models/object_detection/coco/model/yolov8s-640x384/model.synap"
                detection_service.find_objects_from_image_async(
                    image_path=temp_path,
                    model_path=model_path,
                    callback=detection_callback
                )
                
            else:
                logger.error("Failed to save pixmap to temporary file for detection")
                
        except Exception as e:
            logger.error(f"Error running detection on pixmap: {e}")
            import traceback
            logger.error(f"Detection pixmap traceback: {traceback.format_exc()}")
    
    def _on_detection_complete_pixmap(self, json_result):
        """Handle detection results from pixmap-based detection"""
        try:
            logger.info(f"🎯 Object detection completed, processing JSON result...")
            
            # Process JSON result to extract coordinates
            from object_detection import CoordinateProcessor
            coordinate_processor = CoordinateProcessor()
            
            # Parse JSON result directly to filter by class before processing coordinates
            import json
            detection_list = json.loads(json_result)
            logger.info(f"🎯 Received {len(detection_list)} total detections from YOLO")
            
            # Filter to keep only class 0 (person) detections
            person_detections = []
            filtered_out = 0
            for detection in detection_list:
                class_index = detection.get('class_index', -1)
                if class_index == 0:  # Only keep person detections (class 0)
                    person_detections.append(detection)
                    logger.info(f"✅ Keeping person detection (class {class_index}): {detection.get('label', 'unknown')}")
                else:
                    filtered_out += 1
                    logger.info(f"🚫 Filtering out class {class_index}: {detection.get('label', 'unknown')}")
            
            logger.info(f"👤 Filtered to {len(person_detections)} person detections (class 0 only)")
            logger.info(f"🚫 Filtered out {filtered_out} non-person detections")
            
            # Convert filtered person detections back to JSON string for coordinate processing
            person_json_result = json.dumps(person_detections)
            
            # Process coordinates only for person detections
            person_coordinates = coordinate_processor.process_detection_results(person_json_result)
            
            # Store filtered detection results (only persons)
            self.last_detection_results = person_coordinates
            self.detection_coordinates = person_coordinates
            
            # Enable detection mode and reset focus
            self.is_in_detection_mode = True
            self.current_focus_index = 0
            
            # Draw bounding boxes on the displayed frame (only for persons)
            if person_coordinates:
                logger.info(f"Drawing {len(person_coordinates)} person bounding boxes")
                self.draw_single_bounding_box(0)  # Start with first person detection
                
                # Trigger fashion matching for the first person detection
                self.trigger_fashion_matching(0)
            else:
                logger.info("No person detections found in frame (class 0 only)")
                
        except Exception as e:
            logger.error(f"Error handling detection results: {e}")
            import traceback
            logger.error(f"Detection results traceback: {traceback.format_exc()}")
    
    def trigger_fashion_matching(self, detection_index):
        """Trigger fashion matching for a specific detection"""
        try:
            if not self.detection_coordinates or detection_index >= len(self.detection_coordinates):
                logger.warning(f"Invalid detection index: {detection_index}")
                return
            
            # Get the detection coordinates
            x, y, w, h = self.detection_coordinates[detection_index]
            logger.info(f"🔍 Triggering fashion matching for detection {detection_index}: ({x}, {y}, {w}, {h})")
            
            # Get the original frame pixmap (without bounding boxes) for cropping
            if hasattr(self, 'original_frame_pixmap') and self.original_frame_pixmap:
                source_pixmap = self.original_frame_pixmap
                logger.info("🔍 Using original frame pixmap for cropping (without bounding boxes)")
                
                # Crop the detection area
                cropped_image = self.image_cropper.crop_bounding_box(
                    source_pixmap, x, y, w, h
                )
            elif hasattr(self, '_video_label') and self._video_label and self._video_label.pixmap():
                source_pixmap = self._video_label.pixmap()
                logger.warning("🔍 Using video label pixmap for cropping (may have bounding boxes)")
                
                # Crop the detection area
                cropped_image = self.image_cropper.crop_bounding_box(
                    source_pixmap, x, y, w, h
                )
            else:
                logger.error("🔍 No source pixmap available for cropping")
                cropped_image = None
            
            if cropped_image and self.image_embedding:
                logger.info(f"🎯 Running fashion matching for detection {detection_index + 1}")
                logger.info(f"🔍 Cropped image size: {cropped_image.width()}x{cropped_image.height()}")
                logger.info(f"🔍 Image embedding model available: {self.image_embedding is not None}")
                
                # Run fashion matching in background
                def matching_callback(similar_items):
                    try:
                        logger.info(f"✅ Fashion matching completed for detection {detection_index + 1}: {len(similar_items)} matches")
                        self.display_fashion_matches(similar_items, detection_index)
                    except Exception as e:
                        logger.error(f"Error in matching callback: {e}")
                
                # Start matching in background thread
                import threading
                matching_thread = threading.Thread(
                    target=lambda: self.run_fashion_matching(cropped_image, matching_callback),
                    daemon=True
                )
                matching_thread.start()
            else:
                logger.warning("No cropped image or embedding model available for matching")
                
        except Exception as e:
            logger.error(f"Error triggering fashion matching: {e}")
            import traceback
            logger.error(f"Fashion matching traceback: {traceback.format_exc()}")
    
    def run_fashion_matching(self, cropped_image, callback):
        """Run fashion matching on cropped image"""
        try:
            if not self.image_embedding:
                logger.error("No image embedding model available")
                return
            
            logger.info("🚀 Using pre-loaded FashionCLIP model - fast matching!")
            
            # Convert QPixmap to temporary file for embedding
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            if cropped_image.save(temp_path, 'PNG'):
                logger.info(f"💾 Saved cropped image for matching: {temp_path}")
                
                # Load the saved image as PIL Image for embedding
                from PIL import Image
                pil_image = Image.open(temp_path)
                logger.info(f"📸 Loaded PIL image for embedding: {pil_image.size}, mode: {pil_image.mode}")
                
                # Get similar items using PIL image directly
                logger.info(f"🔍 Calling find_similar_images with PIL image: {type(pil_image)}")
                similar_items = self.image_embedding.find_similar_images(pil_image, top_k=12)
                logger.info(f"🎯 Fashion matching returned {len(similar_items)} results")
                
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up matching temp file: {cleanup_error}")
                
                # Call callback with results
                callback(similar_items)
            else:
                logger.error("Failed to save cropped image for matching")
                
        except Exception as e:
            logger.error(f"Error in fashion matching: {e}")
            import traceback
            logger.error(f"Fashion matching traceback: {traceback.format_exc()}")
    
    def display_fashion_matches(self, similar_items, detection_index):
        """Display fashion matching results in UI grid"""
        try:
            logger.info(f"🎨 Displaying {len(similar_items)} fashion matches for detection {detection_index + 1}")
            
            # Clear existing matches
            self.clear_matching_results()
            
            # Display up to 12 matches (4 per row, 3 rows)
            for i, item in enumerate(similar_items[:12]):
                if i < 4:
                    row = 1  # First row
                    col = i
                elif i < 8:
                    row = 2  # Second row
                    col = i - 4
                else:
                    row = 3  # Third row
                    col = i - 8
                
                self.display_single_match(item, row, col, i)
            
            logger.info(f"✅ Displayed {min(len(similar_items), 12)} fashion matches")
            
        except Exception as e:
            logger.error(f"Error displaying fashion matches: {e}")
            import traceback
            logger.error(f"Display matches traceback: {traceback.format_exc()}")
    
    def clear_matching_results(self):
        """Clear existing matching results"""
        try:
            # Clear all match labels using the actual UI structure
            if hasattr(self, 'row1_labels'):
                for label in self.row1_labels:
                    label.clear()
                    label.setText("Match")
            
            if hasattr(self, 'row2_labels'):
                for label in self.row2_labels:
                    label.clear()
                    label.setText("Match")
            
            if hasattr(self, 'row3_labels'):
                for label in self.row3_labels:
                    label.clear()
                    label.setText("Match")
            
            logger.info("🧹 Cleared existing matching results")
            
        except Exception as e:
            logger.error(f"Error clearing matching results: {e}")
    
    def display_single_match(self, item, row, col, index):
        """Display a single matching item"""
        try:
            # Get the appropriate label from the UI structure
            label = None
            if row == 1 and hasattr(self, 'row1_labels') and col < len(self.row1_labels):
                label = self.row1_labels[col]
            elif row == 2 and hasattr(self, 'row2_labels') and col < len(self.row2_labels):
                label = self.row2_labels[col]
            elif row == 3 and hasattr(self, 'row3_labels') and col < len(self.row3_labels):
                label = self.row3_labels[col]
            
            if label:
                # Load and display the image
                from PyQt5.QtGui import QPixmap
                from PyQt5.QtCore import Qt
                pixmap = QPixmap(item['image_path'])
                
                if not pixmap.isNull():
                    # Scale to fit label
                    scaled_pixmap = pixmap.scaled(
                        label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    label.setPixmap(scaled_pixmap)
                    
                    # Set tooltip with similarity score
                    similarity = item.get('similarity', 0.0)
                    label.setToolTip(f"Match {index + 1}\nSimilarity: {similarity:.3f}")
                    
                    logger.info(f"📸 Displayed match {index + 1} in row {row}, col {col} (similarity: {similarity:.3f})")
                else:
                    logger.warning(f"Failed to load image: {item['image_path']}")
            else:
                logger.warning(f"Label for row {row}, col {col} not found")
                
        except Exception as e:
            logger.error(f"Error displaying single match: {e}")
    
    def show_with_captured_frame(self, captured_frame):
        """Show Qt overlay with captured video frame in left 2/3"""
        try:
            from PyQt5.QtCore import Qt
            logger.info("Showing Qt overlay with captured frame...")
            
            # Frame is already captured at 1280x720, so no scaling needed
            FIXED_VIDEO_WIDTH = 1280  # Fixed width for consistency
            FIXED_VIDEO_HEIGHT = 720   # 16:9 aspect ratio
            
            logger.info(f"Using captured frame directly at: {FIXED_VIDEO_WIDTH}x{FIXED_VIDEO_HEIGHT} (no scaling needed)")
            
            # Check if frame is already the correct size
            frame_size = captured_frame.size()
            logger.info(f"📏 Input frame size: {frame_size.width()}x{frame_size.height()}")
            logger.info(f"📏 Target size: {FIXED_VIDEO_WIDTH}x{FIXED_VIDEO_HEIGHT}")
            
            if frame_size.width() == FIXED_VIDEO_WIDTH and frame_size.height() == FIXED_VIDEO_HEIGHT:
                logger.info("✅ Frame is already correct size, using directly")
                scaled_frame = captured_frame
            else:
                logger.info(f"🔄 Frame size mismatch: {frame_size.width()}x{frame_size.height()}, scaling to {FIXED_VIDEO_WIDTH}x{FIXED_VIDEO_HEIGHT}")
                from PyQt5.QtCore import QSize
                scaled_frame = captured_frame.scaled(
                    QSize(FIXED_VIDEO_WIDTH, FIXED_VIDEO_HEIGHT),
                    Qt.IgnoreAspectRatio,  # Force 16:9 ratio
                    Qt.SmoothTransformation
                )
                logger.info(f"✅ Scaled frame size: {scaled_frame.width()}x{scaled_frame.height()}")
            
            # Create video label if it doesn't exist
            if not hasattr(self, '_video_label'):
                from PyQt5.QtWidgets import QLabel, QVBoxLayout
                
                self._video_label = QLabel()
                self._video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
                self._video_label.setScaledContents(False)  # Don't auto-scale, we'll set exact pixmap size
                self._video_label.setAlignment(Qt.AlignCenter)
                
                # Clear existing layout if any
                if self.video_frame.layout():
                    from PyQt5.QtWidgets import QWidget
                    QWidget().setLayout(self.video_frame.layout())
                
                layout = QVBoxLayout(self.video_frame)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self._video_label)
                self.video_frame.setLayout(layout)
            
            # Display the captured frame
            self._video_label.setPixmap(scaled_frame)
            
            # Store the original frame for bounding box drawing (without any bounding boxes)
            self.original_frame_pixmap = scaled_frame.copy()
            logger.info(f"💾 Stored original frame pixmap for bounding box drawing: {self.original_frame_pixmap.width()}x{self.original_frame_pixmap.height()}")
            
            # Get actual label size after setting pixmap
            label_size = self._video_label.size()
            pixmap_size = self._video_label.pixmap().size() if self._video_label.pixmap() else "None"
            
            logger.info(f"📺 QLabel size: {label_size.width()}x{label_size.height()}")
            logger.info(f"📺 QLabel pixmap size: {pixmap_size}")
            logger.info(f"📺 Video frame widget size: {self.video_frame.size().width()}x{self.video_frame.size().height()}")
            
            # Show the window
            self.show()
            self.raise_()
            self.activateWindow()
            
            logger.info("✅ Qt overlay displayed with captured frame")
            
        except Exception as e:
            logger.error(f"Error showing captured frame: {e}")
            import traceback
            logger.error(f"Show frame traceback: {traceback.format_exc()}")
        
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


def test_fullscreen_decode():
    """Quick test of fullscreen video decoding for 4K 60FPS performance"""
    
    logger.info("🚀 Testing fullscreen 4K 60FPS video decoding...")
    
    # Set up video path
    video_path = "samples/clip.mp4"  # Use clip.mp4 for 60FPS testing
    
    try:
        # Initialize GStreamer
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        logger.info("GStreamer initialized for fullscreen test")
        
        # Create fullscreen player
        player = FullscreenVideoPlayer(video_path)
        
        # Start fullscreen playback
        if player.play_fullscreen():
            logger.info("✅ Fullscreen video started successfully!")
            logger.info("Press CTRL+C to stop...")
            
            # Simple keyboard monitoring
            import time
            try:
                while player.is_playing:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Stopping fullscreen video...")
                player.stop()
                logger.info("✅ Fullscreen test completed")
        else:
            logger.error("❌ Failed to start fullscreen video")
            return 1
            
    except Exception as e:
        logger.error(f"Error in fullscreen test: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0

class TinyFocusWindow(QMainWindow):
    """A tiny transparent window to capture keyboard events while video plays fullscreen"""
    
    def __init__(self, player, popup_window):
        super().__init__()
        self.player = player
        self.popup_window = popup_window
        self.init_ui()
    
    def init_ui(self):
        """Initialize the tiny focus window"""
        # Set window properties
        self.setWindowTitle("Focus Window")
        self.setGeometry(10, 10, 50, 50)  # Very small window in top-left corner
        
        # Make window transparent and always on top (Wayland compatible)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowOpacity(0.7)  # More visible but still transparent
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.Tool |
            Qt.X11BypassWindowManagerHint  # Help with Wayland compositors
        )
        
        # Set background color with alpha transparency
        self.setStyleSheet("""
            QMainWindow {
                background-color: rgba(255, 50, 50, 180);
                border: 2px solid rgba(255, 100, 100, 200);
                border-radius: 5px;
            }
        """)
        
        # Add a small label to show it's the focus window
        from PyQt5.QtWidgets import QLabel
        label = QLabel("⌨️", self)
        label.setGeometry(15, 15, 20, 20)
        label.setStyleSheet("""
            color: white; 
            font-size: 16px; 
            font-weight: bold;
            text-shadow: 1px 1px 2px black;
            background-color: transparent;
        """)
        
        # Make sure window can receive keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        
        logger.info("🎯 Tiny focus window created for keyboard input")
    
    def keyPressEvent(self, event):
        """Handle keyboard events in the tiny focus window"""
        key = event.key()
        key_text = event.text()
        
        logger.info(f"🔤 Key pressed in focus window: '{key_text}' (code: {key})")
        
        if key == Qt.Key_Space:
            logger.info("⏸️ Space key - toggling video pause/resume")
            self.player.toggle_pause()
        elif key == Qt.Key_Q:
            logger.info("🚪 Q key - exiting application")
            self.player.stop()
            QApplication.instance().quit()
        else:
            logger.info(f"🔤 Unhandled key: '{key_text}' (code: {key})")
            super().keyPressEvent(event)
    
    def show_and_focus(self):
        """Show the window and give it keyboard focus"""
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()
        logger.info("✅ Focus window shown and activated")

def main():
    """Main function to run the video player with PyQt5 pop-up window"""
    
    # Parse command line arguments
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='4K 60FPS Video Player with Qt Overlay')
    parser.add_argument('--video', '-v', 
                       default='samples/clip.mp4',
                       help='Path to video file relative to current directory (default: samples/clip.mp4)')
    
    args = parser.parse_args()
    
    # Set up video path (relative to current directory where video_test.py is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, args.video)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        logger.error(f"Please check the path relative to: {script_dir}")
        return 1
    
    logger.info(f"Using video file: {video_path}")
    
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
        
        # Connect video control button signals
        # Buttons are already connected in init_ui method
        
        # Set window properties with fixed size
        popup_window.setWindowTitle("🎬 Video Shopping Demo")
        popup_window.setGeometry(100, 100, 1200, 800)  # Larger default size
        popup_window.setMinimumSize(1200, 800)  # Ensure minimum size
        popup_window.show()
        
        # Debug layout sizes after window is shown
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, popup_window.debug_layout_sizes)  # Delay to ensure widgets are rendered
        
        logger.info("PyQt5 pop-up window created successfully")
        
        # Create fullscreen video player for 4K 60FPS performance test
        logger.info(f"Creating fullscreen video player for: {video_path}")
        player = FullscreenVideoPlayer(video_path, popup_window)
        
        # Set video player reference in popup window for button control
        popup_window.set_video_player(player)
        
        # Connect popup window to player for keyboard events
        popup_window.video_player = player
        
        # Hide Qt window initially (video starts in fullscreen)
        popup_window.hide()
        
        # Create tiny focus window for keyboard input
        logger.info("Creating tiny focus window for keyboard input...")
        focus_window = TinyFocusWindow(player, popup_window)
        
        # Connect focus window to player
        player.set_focus_window(focus_window)
        
        # Start fullscreen video playback - no Qt integration, pure performance
        if not player.play():
            logger.error("Failed to start fullscreen video playback")
            return 1
        
        # Show the tiny focus window after video starts
        QTimer.singleShot(500, focus_window.show_and_focus)  # Delay to ensure video is playing
        
        logger.info("Application ready. Controls: SPACE: pause/resume, q: quit")
        logger.info("🎯 Using tiny focus window for keyboard input")
        logger.info("📝 Look for the small red window in the top-left corner - that's where keyboard focus is")
        
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


class FullscreenVideoPlayer:
    """4K 60FPS optimized video player - fullscreen Wayland with Qt overlay on pause"""
    
    def __init__(self, video_path: str, qt_overlay_window=None):
        """Initialize the fullscreen video player"""
        self.video_path = Path(video_path)
        self.pipeline = None
        self.is_playing = False
        self.is_paused = False
        self.qt_overlay_window = qt_overlay_window  # Qt window to show on pause
        self.focus_window = None  # Tiny focus window for keyboard input
        self.fullscreen_mode = True  # Start in fullscreen mode
        self.captured_frame = None   # Store captured frame on pause
        self.capture_pipeline = None  # Separate pipeline for frame capture
        
        # Initialize GStreamer
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        self.Gst = Gst
        
        if not self.Gst.is_initialized():
            self.Gst.init(None)
            logger.info("GStreamer initialized for fullscreen playback")
    
    def create_fullscreen_pipeline(self):
        """Create fullscreen Wayland pipeline for maximum 4K 60FPS performance"""
        try:
            logger.info("Creating fullscreen Wayland pipeline for 4K 60FPS performance...")
            
            # Use absolute path for better compatibility
            abs_path = self.video_path.resolve()
            
            # Simple, high-performance pipeline for 4K 60FPS
            # Use waylandsink with fullscreen for maximum performance
            pipeline_str = f"""
                filesrc location="{abs_path}" !
                decodebin name=dec !
                videoconvert ! waylandsink name=videosink fullscreen=true
            """
            
            logger.info(f"Fullscreen Wayland pipeline: {pipeline_str.strip()}")
            
            # Create pipeline
            self.pipeline = self.Gst.parse_launch(pipeline_str.replace('\n', '').replace('  ', ' '))
            
            # Get the video sink element
            self.video_sink = self.pipeline.get_by_name("videosink")
            if self.video_sink:
                logger.info("Got waylandsink element")
            else:
                logger.warning("Could not get waylandsink element")
            

            
            # Set up basic bus message handling for errors
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            logger.info("Connected to GStreamer bus for error handling")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fullscreen Wayland pipeline: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def create_capture_pipeline(self):
        """Create separate pipeline for frame capture on pause"""
        try:
            logger.info("Creating frame capture pipeline...")
            
            # Pipeline to capture current frame to file
            pipeline_str = f"""
                filesrc location="{self.video_path}" ! 
                decodebin ! 
                videoconvert ! 
                videoscale ! 
                video/x-raw,format=RGB ! 
                appsink name=capturesink emit-signals=true max-buffers=1 drop=true sync=false
            """
            
            self.capture_pipeline = self.Gst.parse_launch(pipeline_str.replace('\n', '').replace('  ', ' '))
            
            # Get capture sink
            self.capture_sink = self.capture_pipeline.get_by_name("capturesink")
            if self.capture_sink:
                self.capture_sink.connect("new-sample", self._on_capture_sample)
                logger.info("Frame capture pipeline created successfully")
                return True
            else:
                logger.error("Failed to get capture sink")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create capture pipeline: {e}")
            return False
    
    def _on_capture_sample(self, sink):
        """Capture frame sample for Qt overlay"""
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                return self.Gst.FlowReturn.ERROR
            
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Extract frame info
            structure = caps.get_structure(0)
            width = structure.get_int("width")[1]
            height = structure.get_int("height")[1]
            
            # Map buffer
            success, map_info = buffer.map(self.Gst.MapFlags.READ)
            if success:
                try:
                    # Create QPixmap from captured frame
                    from PyQt5.QtGui import QImage, QPixmap
                    
                    image_data = bytes(map_info.data)
                    image = QImage(image_data, width, height, width * 3, QImage.Format_RGB888)
                    
                    if not image.isNull():
                        self.captured_frame = QPixmap.fromImage(image)
                        logger.info(f"Captured frame: {width}x{height}")
                        
                        # Stop capture pipeline - we only need one frame
                        self.capture_pipeline.set_state(self.Gst.State.NULL)
                        
                finally:
                    buffer.unmap(map_info)
            
            return self.Gst.FlowReturn.OK
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return self.Gst.FlowReturn.ERROR
    
    def set_focus_window(self, focus_window):
        """Set reference to the tiny focus window"""
        self.focus_window = focus_window
        logger.info("🎯 Focus window reference set in player")
    
    def capture_current_frame(self):
        """Simple frame capture - safe but shows video start frame"""
        try:
            import time
            
            logger.info("📸 Simple frame capture - safe method (shows video start)")
            
            # Ensure main pipeline is paused
            if not self.is_paused:
                logger.warning("Pipeline not paused, pausing now")
                self.pipeline.set_state(self.Gst.State.PAUSED)
                self.is_paused = True
                time.sleep(0.1)
            
            # Get current position for logging only
            success, position = self.pipeline.query_position(self.Gst.Format.TIME)
            if success:
                logger.info(f"Main pipeline position: {position / self.Gst.SECOND:.3f}s (actual position)")
            
            # Create simple capture pipeline - captures from start for safety
            abs_path = self.video_path.resolve()
            capture_pipeline_str = f"""
                filesrc location="{abs_path}" !
                decodebin !
                videoconvert !
                videoscale !
                video/x-raw,format=RGB,width=1280,height=720 !
                appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false
            """
            
            logger.info("Creating simple capture pipeline (no seeking - safe)")
            
            # Create and configure capture pipeline
            capture_pipeline = self.Gst.parse_launch(capture_pipeline_str.replace('\n', '').replace('  ', ' '))
            
            # Get appsink element
            appsink = capture_pipeline.get_by_name("sink")
            if not appsink:
                logger.error("Failed to get appsink element")
                return None
            
            # Set up callback for new sample
            self._captured_pixmap = None
            self._frame_captured = False
            appsink.connect("new-sample", self._on_capture_sample_memory)
            
            # Start capture pipeline directly
            logger.info("Starting simple capture (from beginning - stable)")
            capture_pipeline.set_state(self.Gst.State.PLAYING)
            
            # Wait for frame to be captured
            max_wait = 0.8
            wait_time = 0.0
            while not self._frame_captured and wait_time < max_wait:
                time.sleep(0.02)
                wait_time += 0.02
            
            # Stop capture pipeline
            capture_pipeline.set_state(self.Gst.State.NULL)
            
            if self._frame_captured and self._captured_pixmap:
                logger.info(f"✅ Frame captured safely: {self._captured_pixmap.width()}x{self._captured_pixmap.height()}")
                logger.info("⚠️ Note: Shows video start frame (safe method)")
                return self._captured_pixmap
            else:
                logger.error("Frame capture failed")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            import traceback
            logger.error(f"Frame capture traceback: {traceback.format_exc()}")
            return None
    
    def _on_capture_sample_memory(self, appsink):
        """Handle captured frame sample - store directly in memory"""
        try:
            sample = appsink.emit("pull-sample")
            if sample:
                buffer = sample.get_buffer()
                caps = sample.get_caps()
                
                # Extract frame info
                structure = caps.get_structure(0)
                width = structure.get_int("width")[1]
                height = structure.get_int("height")[1]
                
                # Map buffer
                success, map_info = buffer.map(self.Gst.MapFlags.READ)
                if success:
                    try:
                        # Create QPixmap directly from raw RGB data
                        from PyQt5.QtGui import QImage, QPixmap
                        
                        image_data = bytes(map_info.data)
                        image = QImage(image_data, width, height, width * 3, QImage.Format_RGB888)
                        
                        if not image.isNull():
                            self._captured_pixmap = QPixmap.fromImage(image)
                            logger.info(f"✅ Frame captured to memory: {width}x{height}")
                            logger.info(f"✅ QImage created: {image.width()}x{image.height()}")
                            logger.info(f"✅ QPixmap created: {self._captured_pixmap.width()}x{self._captured_pixmap.height()}")
                            self._frame_captured = True
                        else:
                            logger.error("❌ Failed to create QImage from buffer")
                        
                    finally:
                        buffer.unmap(map_info)
            
            return self.Gst.FlowReturn.OK
            
        except Exception as e:
            logger.error(f"Error in memory capture callback: {e}")
            return self.Gst.FlowReturn.ERROR
    

    def _on_capture_sample(self, appsink):
        """Handle captured frame sample"""
        try:
            sample = appsink.emit("pull-sample")
            if sample:
                buffer = sample.get_buffer()
                caps = sample.get_caps()
                
                # Extract frame info
                structure = caps.get_structure(0)
                width = structure.get_int("width")[1]
                height = structure.get_int("height")[1]
                
                # Map buffer
                success, map_info = buffer.map(self.Gst.MapFlags.READ)
                if success:
                    try:
                        # Save raw RGB data as PNG
                        from PyQt5.QtGui import QImage, QPixmap
                        
                        image_data = bytes(map_info.data)
                        image = QImage(image_data, width, height, width * 3, QImage.Format_RGB888)
                        
                        if not image.isNull():
                            pixmap = QPixmap.fromImage(image)
                            if pixmap.save(self._capture_output_path, "PNG"):
                                logger.info(f"Frame saved: {width}x{height} -> {self._capture_output_path}")
                                self._frame_captured = True
                            else:
                                logger.error("Failed to save frame to file")
                        
                    finally:
                        buffer.unmap(map_info)
            
            return self.Gst.FlowReturn.OK
            
        except Exception as e:
            logger.error(f"Error in capture sample callback: {e}")
            return self.Gst.FlowReturn.ERROR
    
    def _cleanup_old_frames(self, cache_dir):
        """Clean up old captured frames to save space"""
        try:
            import glob
            pattern = os.path.join(cache_dir, "paused_frame_*.png")
            files = glob.glob(pattern)
            
            # Keep only the 3 most recent frames
            if len(files) > 3:
                files.sort(key=os.path.getctime)
                for old_file in files[:-3]:
                    os.remove(old_file)
                    logger.info(f"Cleaned up old frame: {os.path.basename(old_file)}")
                    
        except Exception as e:
            logger.warning(f"Error cleaning up old frames: {e}")
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages for errors and EOS"""
        try:
            msg_type = message.type
            
            if msg_type == self.Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                logger.error(f"GStreamer error: {err}, debug: {debug}")
            
            elif msg_type == self.Gst.MessageType.EOS:
                logger.info("End of stream reached")
                self.stop()
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling bus message: {e}")
            return True
    
    def play(self):
        """Start video playback (wrapper for play_fullscreen)"""
        return self.play_fullscreen()
    
    def play_fullscreen(self):
        """Start fullscreen video playback"""
        try:
            logger.info("Starting fullscreen 4K 60FPS video playback...")
            
            if not self.create_fullscreen_pipeline():
                logger.error("Failed to create fullscreen pipeline")
                return False
            
            # Start fullscreen playback
            ret = self.pipeline.set_state(self.Gst.State.PLAYING)
            if ret == self.Gst.StateChangeReturn.FAILURE:
                logger.error("Failed to start fullscreen playback")
                return False
            
            self.is_playing = True
            self.is_paused = False
            self.fullscreen_mode = True
            
            logger.info("✅ Fullscreen video started - maximum performance mode!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start fullscreen playback: {e}")
            return False
    
    def pause_and_show_qt(self):
        """Pause video and switch to Qt overlay with captured frame"""
        try:
            import os
            logger.info("🔄 Pausing fullscreen video and capturing frame...")
            
            # 1. Pause main video pipeline
            if self.pipeline:
                self.pipeline.set_state(self.Gst.State.PAUSED)
                self.is_paused = True
                logger.info("✅ Video pipeline paused")
            
            # 2. Capture current frame directly to memory
            captured_pixmap = self.capture_current_frame()
            
            # 3. Show Qt overlay window with captured frame
            if self.qt_overlay_window:
                logger.info("Showing Qt overlay window...")
                self.qt_overlay_window.show()
                self.qt_overlay_window.raise_()
                self.qt_overlay_window.activateWindow()
                
                # Display captured frame in Qt video frame (direct memory)
                if captured_pixmap:
                    self.qt_overlay_window.display_captured_pixmap(captured_pixmap)
                    logger.info(f"✅ Captured frame displayed from memory: {captured_pixmap.width()}x{captured_pixmap.height()}")
                else:
                    logger.warning("No captured frame to display")
                
                logger.info("✅ Qt overlay window shown and activated")
            
            # 4. Hide focus window during pause (Qt window will handle keyboard)
            if hasattr(self, 'focus_window') and self.focus_window:
                self.focus_window.hide()
                logger.info("🎯 Focus window hidden during pause")
            
            self.fullscreen_mode = False
            logger.info("✅ Switched to Qt overlay mode - video is now paused")
            
        except Exception as e:
            logger.error(f"Failed to pause and show Qt: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def resume_fullscreen(self):
        """Hide Qt overlay and resume fullscreen video"""
        try:
            logger.info("🔄 Hiding Qt overlay and resuming fullscreen video...")
            
            # 1. Hide Qt overlay window
            if self.qt_overlay_window:
                logger.info("Hiding Qt overlay window...")
                self.qt_overlay_window.hide()
                logger.info("✅ Qt overlay window hidden")
            
            # 2. Show focus window again for keyboard input
            if hasattr(self, 'focus_window') and self.focus_window:
                self.focus_window.show_and_focus()
                logger.info("🎯 Focus window shown for keyboard input")
            
            # 3. Resume fullscreen video
            if self.pipeline:
                logger.info("Resuming video pipeline...")
                self.pipeline.set_state(self.Gst.State.PLAYING)
                self.is_paused = False
                self.fullscreen_mode = True
                logger.info("✅ Video pipeline resumed")
            
            logger.info("✅ Resumed fullscreen video playback")
            
        except Exception as e:
            logger.error(f"Failed to resume fullscreen: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def toggle_pause(self):
        """Toggle between fullscreen video and Qt overlay"""
        if self.is_paused:
            self.resume_fullscreen()
        else:
            self.pause_and_show_qt()
    
    def stop(self):
        """Stop video playback"""
        try:
            if self.pipeline:
                self.pipeline.set_state(self.Gst.State.NULL)
                logger.info("Fullscreen video stopped")
            
            if self.capture_pipeline:
                self.capture_pipeline.set_state(self.Gst.State.NULL)
                logger.info("Capture pipeline stopped")
                
            self.is_playing = False
            
        except Exception as e:
            logger.error(f"Error stopping video: {e}")


if __name__ == "__main__":
    # Run the full application with Qt and focus window
    logger.info("🎬 Starting 4K 60FPS video application with Qt overlay...")
    logger.info("📝 Usage: python video_test.py [--video path/to/video.mp4]")
    logger.info("📝 Example: python video_test.py --video samples/clip_1.mp4")
    sys.exit(main())
