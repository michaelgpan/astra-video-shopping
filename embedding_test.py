#!/usr/bin/env python3

import sys
import os
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QFrame, QGridLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QKeySequence

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_embedding import ImageEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_index = 0  # Start with img_000.png
        self.images_dir = "images"
        self.image_embedding = None
        
        # Initialize UI
        self.init_ui()
        
        # Initialize image embedding system
        self.init_image_embedding()
        
        # Load initial image and find similarities
        self.update_display()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Fashion Similarity Test")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for query image (similar to video panel)
        self.left_panel = QFrame()
        self.left_panel.setStyleSheet("border: 2px solid gray; background-color: white;")
        self.left_panel.setFixedWidth(400)
        
        left_layout = QVBoxLayout(self.left_panel)
        
        # Query image label
        self.query_label = QLabel("Query Image")
        self.query_label.setAlignment(Qt.AlignCenter)
        self.query_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(self.query_label)
        
        # Query image display
        self.query_image_label = QLabel()
        self.query_image_label.setAlignment(Qt.AlignCenter)
        self.query_image_label.setStyleSheet("border: 1px solid lightgray; background-color: white;")
        self.query_image_label.setMinimumSize(350, 350)
        left_layout.addWidget(self.query_image_label)
        
        # Instructions
        instructions = QLabel("Press → to go to next image (+10)\nPress ← to go to previous image (-10)")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("margin: 10px; color: gray;")
        left_layout.addWidget(instructions)
        
        # Right panel for similar images
        self.right_panel = QFrame()
        self.right_panel.setStyleSheet("border: 2px solid gray; background-color: white;")
        
        right_layout = QVBoxLayout(self.right_panel)
        
        # Similar images title
        self.similar_label = QLabel("Top 12 Similar Images")
        self.similar_label.setAlignment(Qt.AlignCenter)
        self.similar_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        right_layout.addWidget(self.similar_label)
        
        # Grid for similar images (3x4 grid)
        self.similar_grid = QGridLayout()
        self.similar_image_labels = []
        
        for row in range(4):
            for col in range(3):
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 1px solid lightgray; background-color: white;")
                label.setFixedSize(150, 150)
                label.setScaledContents(True)
                self.similar_image_labels.append(label)
                self.similar_grid.addWidget(label, row, col)
        
        right_layout.addLayout(self.similar_grid)
        
        # Add panels to main layout
        main_layout.addWidget(self.left_panel, 1)  # 1/3 width
        main_layout.addWidget(self.right_panel, 2)  # 2/3 width
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        
        logger.info("Test UI initialized successfully")
        
    def init_image_embedding(self):
        """Initialize the image embedding system"""
        try:
            logger.info("Initializing ImageEmbedding system...")
            self.image_embedding = ImageEmbedding()
            logger.info("ImageEmbedding system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ImageEmbedding: {e}")
            
    def update_display(self):
        """Update both query image and similar images display"""
        try:
            # Load and display query image
            image_filename = f"img_{self.current_image_index:03d}.png"
            image_path = os.path.join(self.images_dir, image_filename)
            
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return
                
            # Display query image
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale image to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.query_image_label.setPixmap(scaled_pixmap)
                
                # Update title
                self.query_label.setText(f"Query Image: {image_filename}")
                
                logger.info(f"Loaded query image: {image_filename}")
                
                # Find and display similar images
                self.find_and_display_similar_images(pixmap)
            else:
                logger.error(f"Failed to load image: {image_path}")
                
        except Exception as e:
            logger.error(f"Error updating display: {e}")
            
    def find_and_display_similar_images(self, query_pixmap):
        """Find and display top 12 similar images"""
        try:
            if self.image_embedding is None:
                logger.error("ImageEmbedding system not initialized")
                return
                
            logger.info("Searching for similar images...")
            
            # Find top 12 similar images
            results = self.image_embedding.find_similar_images(query_pixmap, top_k=12)
            
            if not results:
                logger.error("No similar images found")
                return
                
            logger.info(f"Found {len(results)} similar images")
            
            # Display results in grid
            for i, result in enumerate(results):
                if i < len(self.similar_image_labels):
                    image_path = result['image_path']
                    similarity = result['similarity']
                    
                    if os.path.exists(image_path):
                        pixmap = QPixmap(image_path)
                        if not pixmap.isNull():
                            # Scale to fit grid cell
                            scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            self.similar_image_labels[i].setPixmap(scaled_pixmap)
                            
                            # Set tooltip with similarity score
                            filename = os.path.basename(image_path)
                            self.similar_image_labels[i].setToolTip(f"{filename}\nSimilarity: {similarity:.3f}")
                            
                            logger.info(f"  #{i+1}: {filename} (similarity: {similarity:.3f})")
                        else:
                            logger.error(f"Failed to load similar image: {image_path}")
                    else:
                        logger.error(f"Similar image not found: {image_path}")
                        
            # Clear any remaining grid cells
            for i in range(len(results), len(self.similar_image_labels)):
                self.similar_image_labels[i].clear()
                self.similar_image_labels[i].setToolTip("")
                
            logger.info("Similar images display updated")
            
        except Exception as e:
            logger.error(f"Error finding similar images: {e}")
            
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        try:
            if event.key() == Qt.Key_Right:
                # Go to next image (+10)
                self.current_image_index = min(999, self.current_image_index + 10)
                logger.info(f"Next image: img_{self.current_image_index:03d}.png")
                self.update_display()
                
            elif event.key() == Qt.Key_Left:
                # Go to previous image (-10)
                self.current_image_index = max(0, self.current_image_index - 10)
                logger.info(f"Previous image: img_{self.current_image_index:03d}.png")
                self.update_display()
                
            elif event.key() == Qt.Key_Q:
                # Quit
                logger.info("Quitting application...")
                self.close()
                
            else:
                super().keyPressEvent(event)
                
        except Exception as e:
            logger.error(f"Error handling key press: {e}")

def main():
    """Main function"""
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Create and show main window
        window = TestWindow()
        window.show()
        
        logger.info("Application started. Controls: → next (+10), ← previous (-10), Q quit")
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
