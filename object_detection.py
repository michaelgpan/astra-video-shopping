#!/usr/bin/env python3
"""
Object Detection Service for Video Player
Implements frame capture and object detection functionality matching Kotlin implementation
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import threading
import time
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QRect

# Configure logging
logger = logging.getLogger(__name__)

# Data classes matching Kotlin implementation
@dataclass
class Clothes:
    """Data class representing detected clothing item"""
    class_index: int
    height: int
    label: str
    width: int
    x: int
    y: int

@dataclass
class Person:
    """Data class representing detected person with clothing predictions"""
    clothe_prediction: List[Clothes]
    height: int
    width: int
    x: int
    y: int

class ObjectDetectionService:
    """Object detection service with hardcoded coordinates for testing"""
    
    def __init__(self):
        self.interrupted = False
        self.detection_thread = None
    
    def yolo_od(self, image_path: str, model_path: str = "model.synap") -> str:
        """
        Run YOLO object detection using synap on a given image and return results
        as a direct list of detections.
        
        Schema returned:
        [
          {
            "class_index": int,
            "confidence": float,
            "height": int,
            "width": int,
            "x": int,
            "y": int,
            "label": str
          }, ...
        ]
        """
        try:
            # Lazy import to avoid hard dependency at module import time
            from synap import Network
            from synap.preprocessor import Preprocessor
            from synap.postprocessor import Detector
        except Exception as e:
            logger.error(f"ObjectDetectionService: synap dependency not available: {e}")
            return "[]"
        
        import os
        if not os.path.exists(model_path):
            logger.error(f"ObjectDetectionService: Model not found at '{model_path}'")
            return "[]"
        if not os.path.exists(image_path):
            logger.error(f"ObjectDetectionService: Image not found at '{image_path}'")
            return "[]"
        
        try:
            logger.info(f"ObjectDetectionService: Running YOLO (synap) on {image_path} with model {model_path}")
            network = Network(model_path)
            preprocessor = Preprocessor()
            detector = Detector()
            
            assigned_rect = preprocessor.assign(network.inputs, image_path)
            _ = network.predict()
            result = detector.process(_, assigned_rect)
            
            print("#   Score  Class   Position        Size  Description     Landmarks")
            for i, item in enumerate(result.items):
                bb = item.bounding_box
                print(
                    f"{i:<3}  {item.confidence:.2f} {item.class_index:>6}  {bb.origin.x:>4},{bb.origin.y:>4}   {bb.size.x:>4},{bb.size.y:>4}  {'':<16}",
                    end="",
                )
                for lm in item.landmarks:
                    print(f" {lm}", end="")
                print()


            # Map synap results directly - each detection as a separate entry
            detection_entries = []
            for item in getattr(result, 'items', []):
                bb = item.bounding_box
                entry = {
                    'class_index': int(getattr(item, 'class_index', 0)),
                    'confidence': float(getattr(item, 'confidence', 0.0)),
                    'height': int(getattr(getattr(bb, 'size', None), 'y', 0) if getattr(bb, 'size', None) else 0),
                    'width': int(getattr(getattr(bb, 'size', None), 'x', 0) if getattr(bb, 'size', None) else 0),
                    'x': int(getattr(getattr(bb, 'origin', None), 'x', 0) if getattr(bb, 'origin', None) else 0),
                    'y': int(getattr(getattr(bb, 'origin', None), 'y', 0) if getattr(bb, 'origin', None) else 0),
                    'label': f"class_{int(getattr(item, 'class_index', 0))}"
                }
                detection_entries.append(entry)
            
            json_result = json.dumps(detection_entries)
            logger.info(f"ObjectDetectionService: YOLO returned {len(detection_entries)} detection(s)")
            return json_result
        except Exception as e:
            logger.error(f"ObjectDetectionService: Error running synap detection: {e}")
            return "[]"
    
    def find_objects_from_image(self, image_path: str, model_path: str = "model.synap") -> str:
        """Convenience wrapper to run detection on a static image and return JSON schema"""
        return self.yolo_od(image_path=image_path, model_path=model_path)
    
    def find_objects_from_image_async(self, image_path: str, model_path: str = "model.synap", callback=None):
        """Run YOLO detection asynchronously on the given image and invoke callback with JSON result"""
        # Reset interrupted flag for new detection attempt
        self.interrupted = False
        
        def detection_worker():
            try:
                result = self.yolo_od(image_path=image_path, model_path=model_path)
                if not self.interrupted and callback:
                    callback(result)
                elif self.interrupted:
                    logger.info("ObjectDetectionService: Image detection was interrupted, skipping callback")
            except Exception as e:
                logger.error(f"ObjectDetectionService: Error in image async detection: {e}")
                if callback:
                    callback("[]")
        
        self.detection_thread = threading.Thread(target=detection_worker)
        self.detection_thread.start()
    
    def find_objects(self) -> str:
        """
        Simulate object detection API call with hardcoded coordinates
        Returns JSON string matching Kotlin implementation format
        """
        # Reset interrupted flag for new detection attempt
        self.interrupted = False
        logger.info("ObjectDetectionService: Finding objects (hardcoded for testing)")
        
        # Simulate processing time like real API
        time.sleep(0.5)
        
        if self.interrupted:
            logger.info("ObjectDetectionService: Detection interrupted")
            return "[]"
        
        # Hardcoded detection results matching video content
        # These coordinates represent detected clothing items in the video frame
        # Adjusted for 640x480 video frame size
        hardcoded_results = [
            {
                "clothe_prediction": [
                    {
                        "class_index": 1,
                        "height": 220,
                        "label": "yellow_top",
                        "width": 300,  # Reduced from 360 to fit frame
                        "x": 180,
                        "y": 270
                    },
                    {
                        "class_index": 2,
                        "height": 400,  # Reduced from 540 to fit frame
                        "label": "purple_dress",
                        "width": 280,   # Reduced from 360 to fit frame
                        "x": 200,       # Reduced from 540 to fit within frame
                        "y": 50         # Adjusted to fit within frame height
                    }
                ],
                "height": 400,
                "width": 300,
                "x": 150,
                "y": 100
            }
        ]
        
        # Convert to JSON string
        json_result = json.dumps(hardcoded_results)
        logger.info(f"ObjectDetectionService: Returning {len(hardcoded_results)} person(s) with clothing predictions")
        
        return json_result
    
    def find_objects_async(self, callback):
        """
        Asynchronous object detection matching Kotlin's thread-based approach
        """
        def detection_worker():
            try:
                result = self.find_objects()
                if not self.interrupted:
                    callback(result)
            except Exception as e:
                logger.error(f"ObjectDetectionService: Error in async detection: {e}")
                callback("[]")
        
        self.detection_thread = threading.Thread(target=detection_worker)
        self.detection_thread.start()
    
    def interrupt(self):
        """Interrupt the object detection service"""
        self.interrupted = True
        logger.info("ObjectDetectionService: Interrupted")
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)

class FrameCaptureService:
    """Service for capturing video frames when paused"""
    
    def __init__(self):
        self.last_captured_frame = None
        self.frame_captured = False
        self.last_captured_frame_path = None
    
    def capture_frame(self, video_widget) -> bool:
        """
        Capture current video frame for object detection
        Returns True if frame captured successfully
        """
        logger.info("FrameCaptureService: Capturing current video frame")
        
        try:
            # Try to locate a QLabel child that holds the current video pixmap
            from PyQt5.QtWidgets import QLabel
            target_label = None
            for child in video_widget.findChildren(QLabel):
                if child.pixmap() is not None and not child.pixmap().isNull():
                    target_label = child
                    break
            
            captured_pixmap = None
            if target_label is not None:
                captured_pixmap = target_label.pixmap()
            else:
                # Fallback: grab the widget content directly from screen
                try:
                    from PyQt5.QtGui import QGuiApplication
                    screen = QGuiApplication.primaryScreen()
                    if screen is not None and hasattr(video_widget, 'winId'):
                        captured_pixmap = screen.grabWindow(int(video_widget.winId()))
                        if captured_pixmap and not captured_pixmap.isNull():
                            logger.info("FrameCaptureService: Captured frame via screen grab fallback")
                except Exception as grab_e:
                    logger.error(f"FrameCaptureService: Screen grab failed: {grab_e}")
            
            if captured_pixmap is not None and not captured_pixmap.isNull():
                # Ensure cache directory exists (relative to current working directory)
                import os
                cache_dir = "cache"
                os.makedirs(cache_dir, exist_ok=True)
                image_path = os.path.join(cache_dir, "paused_frame.png")
                
                if captured_pixmap.save(image_path, "PNG"):
                    self.last_captured_frame = {
                        "width": captured_pixmap.width(),
                        "height": captured_pixmap.height(),
                        "format": "PNG",
                        "data": None
                    }
                    self.last_captured_frame_path = image_path
                    self.frame_captured = True
                    logger.info(f"FrameCaptureService: Frame captured and saved to {image_path}")
                    return True
                else:
                    logger.error("FrameCaptureService: Failed to save captured frame to disk")
            
            # Fallback: simulate frame capture if no pixmap available
            logger.info("FrameCaptureService: No pixmap found - using simulated frame capture")
            self.last_captured_frame = {
                "width": 640,
                "height": 480,
                "format": "RGB",
                "data": "simulated_frame_data"
            }
            self.last_captured_frame_path = None
            self.frame_captured = True
            return True
            
        except Exception as e:
            logger.error(f"FrameCaptureService: Error capturing frame: {e}")
            return False
    
    def get_last_frame(self):
        """Get the last captured frame"""
        return self.last_captured_frame
    
    def get_last_frame_path(self) -> Optional[str]:
        """Get disk path of the last saved frame image if available"""
        return self.last_captured_frame_path
    
    def reset_capture_flag(self):
        """Reset the frame captured flag"""
        self.frame_captured = False

class CoordinateProcessor:
    """Process detection coordinates matching Kotlin logic"""
    
    @staticmethod
    def process_detection_results(json_result: str) -> List[Tuple[int, int, int, int]]:
        """
        Process JSON detection results and extract final coordinates
        Returns list of (x, y, width, height) tuples for each detected item
        """
        try:
            detection_list = json.loads(json_result)
            coordinates = []
            
            for detection in detection_list:
                # Direct mapping from YOLO detection results
                x = detection.get('x', 0)
                y = detection.get('y', 0)
                width = detection.get('width', 0)
                height = detection.get('height', 0)
                
                # Ensure coordinates are non-negative
                x = max(0, x)
                y = max(0, y)
                
                coordinates.append((x, y, width, height))
                
                logger.info(f"Processed coordinates for {detection.get('label', 'unknown')} "
                          f"(class {detection.get('class_index', 0)}, confidence {detection.get('confidence', 0.0):.2f}): "
                          f"({x}, {y}, {width}, {height})")
            
            return coordinates
            
        except Exception as e:
            logger.error(f"CoordinateProcessor: Error processing detection results: {e}")
            return []
    
    @staticmethod
    def create_python_objects(json_result: str) -> List[Clothes]:
        """
        Convert JSON result to Python Clothes objects (direct YOLO detections)
        """
        try:
            detection_list = json.loads(json_result)
            detections = []
            
            for detection_data in detection_list:
                detection = Clothes(
                    class_index=detection_data.get('class_index', 0),
                    height=detection_data.get('height', 0),
                    label=detection_data.get('label', ''),
                    width=detection_data.get('width', 0),
                    x=detection_data.get('x', 0),
                    y=detection_data.get('y', 0)
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"CoordinateProcessor: Error creating Python objects: {e}")
            return []

class ImageCropper:
    """Handles cropping of images within bounding box coordinates"""
    
    @staticmethod
    def crop_bounding_box(pixmap, x, y, width, height):
        """
        Crop a region from QPixmap using bounding box coordinates
        
        Args:
            pixmap: QPixmap source image (video frame)
            x, y: Top-left coordinates of bounding box
            width, height: Dimensions of bounding box
            
        Returns:
            QPixmap: Cropped image region
        """
        if not pixmap or pixmap.isNull():
            logging.error("ImageCropper: Invalid source pixmap")
            return QPixmap()
        
        # Bounds checking - ensure crop doesn't exceed frame dimensions
        # Matches Kotlin logic: if (x + width > pauseFrameBitmap.width) width = pauseFrameBitmap.width - x
        if x + width > pixmap.width():
            width = pixmap.width() - x
        if y + height > pixmap.height():
            height = pixmap.height() - y
            
        # Ensure coordinates are not negative
        if x < 0:
            x = 0
        if y < 0:
            y = 0
            
        # Ensure width and height are positive
        if width <= 0 or height <= 0:
            logging.error(f"ImageCropper: Invalid crop dimensions: width={width}, height={height}")
            return QPixmap()
        
        logging.info(f"ImageCropper: Cropping region at ({x}, {y}) with size {width}x{height}")
        
        # Create crop rectangle and extract region
        crop_rect = QRect(x, y, width, height)
        cropped_pixmap = pixmap.copy(crop_rect)
        
        return cropped_pixmap
    
    @staticmethod
    def save_cropped_image(cropped_pixmap, filepath):
        """
        Save cropped image to file
        
        Args:
            cropped_pixmap: QPixmap to save
            filepath: Path where to save the image
            
        Returns:
            bool: True if saved successfully
        """
        if not cropped_pixmap or cropped_pixmap.isNull():
            logging.error("ImageCropper: Cannot save null pixmap")
            return False
            
        success = cropped_pixmap.save(filepath, "PNG")
        if success:
            logging.info(f"ImageCropper: Saved cropped image to {filepath}")
        else:
            logging.error(f"ImageCropper: Failed to save cropped image to {filepath}")
            
        return success

# Main detection coordinator class
class DetectionCoordinator:
    
    def __init__(self):
        self.frame_service = FrameCaptureService()
        self.detection_service = ObjectDetectionService()
        self.coordinate_processor = CoordinateProcessor()
        self.detection_results = []
        self.load_thread = None
    
    def on_video_paused(self, video_widget, callback=None):
        """
        Handle video pause event - capture frame and detect objects
        Matches Kotlin's onIsPlayingChanged logic
        """
        logger.info("DetectionCoordinator: Video paused, starting detection process")
        
        # Always capture frame and run detection when video is paused
        # Reset the capture flag first to ensure fresh detection
        self.frame_service.reset_capture_flag()
        
        # Capture frame
        if self.frame_service.capture_frame(video_widget):
            
            # Start object detection in background thread
            def detection_callback(json_result):
                try:
                    # Process coordinates
                    coordinates = self.coordinate_processor.process_detection_results(json_result)
                    
                    # Create Python objects (direct YOLO detections)
                    detections = self.coordinate_processor.create_python_objects(json_result)
                    
                    # Store results in memory
                    self.detection_results = {
                        'coordinates': coordinates,
                        'detections': detections,
                        'json_result': json_result
                    }
                    
                    logger.info(f"DetectionCoordinator: Detection complete, found {len(coordinates)} items")
                    
                    # Call callback if provided
                    if callback:
                        callback(self.detection_results)
                        
                except Exception as e:
                    logger.error(f"DetectionCoordinator: Error in detection callback: {e}")
            
            # Prefer YOLO detection on the saved frame if available; fallback to hardcoded detection
            image_path = self.frame_service.get_last_frame_path()
            if image_path:
                logger.info(f"DetectionCoordinator: Starting YOLO detection on captured frame {image_path}")
                # Use hardcoded model path for target machine
                model_path = "/usr/share/synap/models/object_detection/coco/model/yolov8s-640x384/model.synap"
                self.detection_service.find_objects_from_image_async(image_path, model_path=model_path, callback=detection_callback)
            else:
                logger.warning("DetectionCoordinator: No saved frame path available, using fallback detection")
                self.detection_service.find_objects_async(detection_callback)
        
        else:
            logger.error("DetectionCoordinator: Failed to capture frame")
    
    def get_detection_results(self):
        """Get the last detection results"""
        return self.detection_results
    
    def reset(self):
        """Reset detection state"""
        self.frame_service.reset_capture_flag()
        self.detection_results = []
        if self.detection_service:
            self.detection_service.interrupt()
