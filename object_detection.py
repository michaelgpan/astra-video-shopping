#!/usr/bin/env python3
"""
Object Detection Service for Video Player
"""

import json
import logging
from typing import List, Optional, Tuple
import threading
import time
import cv2
import numpy as np
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QRect

# Configure logging
logger = logging.getLogger(__name__)


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
            "label": str,
            "mask": {
              "data": str,      # Base64 encoded mask data
              "shape": tuple,   # Mask array shape (height, width)
              "dtype": str      # Mask data type
            } | null            # null if no mask available
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
                
                # Extract mask data if available
                mask_data = None
                if hasattr(item, 'mask') and item.mask is not None:
                    try:
                        # Convert mask to numpy array and encode as base64 for JSON serialization
                        import base64
                        mask_array = np.array(item.mask.data) if hasattr(item.mask, 'data') else None
                        if mask_array is not None:
                            # Encode mask as base64 string for JSON serialization
                            mask_bytes = mask_array.tobytes()
                            mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')
                            mask_data = {
                                'data': mask_b64,
                                'shape': mask_array.shape,
                                'dtype': str(mask_array.dtype)
                            }
                            logger.info(f"ObjectDetectionService: Extracted mask for detection {len(detection_entries)}, shape: {mask_array.shape}")
                    except Exception as e:
                        logger.warning(f"ObjectDetectionService: Failed to extract mask: {e}")
                
                entry = {
                    'class_index': int(getattr(item, 'class_index', 0)),
                    'confidence': float(getattr(item, 'confidence', 0.0)),
                    'height': int(getattr(getattr(bb, 'size', None), 'y', 0) if getattr(bb, 'size', None) else 0),
                    'width': int(getattr(getattr(bb, 'size', None), 'x', 0) if getattr(bb, 'size', None) else 0),
                    'x': int(getattr(getattr(bb, 'origin', None), 'x', 0) if getattr(bb, 'origin', None) else 0),
                    'y': int(getattr(getattr(bb, 'origin', None), 'y', 0) if getattr(bb, 'origin', None) else 0),
                    'label': f"class_{int(getattr(item, 'class_index', 0))}",
                    'mask': mask_data
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
    
    def yolo_od_raw(self, image_path: str, model_path: str = "model.synap"):
        """
        Run YOLO object detection using synap and return raw synap result objects
        This method returns the original synap detection objects with masks intact
        """
        try:
            # Lazy import to avoid hard dependency at module import time
            from synap import Network
            from synap.preprocessor import Preprocessor
            from synap.postprocessor import Detector
        except Exception as e:
            logger.error(f"ObjectDetectionService: synap dependency not available: {e}")
            return None
        
        if not os.path.exists(model_path):
            logger.error(f"ObjectDetectionService: Model not found at '{model_path}'")
            return None
        if not os.path.exists(image_path):
            logger.error(f"ObjectDetectionService: Image not found at '{image_path}'")
            return None
        
        try:
            logger.info(f"ObjectDetectionService: Running YOLO (synap) raw on {image_path} with model {model_path}")
            network = Network(model_path)
            preprocessor = Preprocessor()
            detector = Detector()
            
            assigned_rect = preprocessor.assign(network.inputs, image_path)
            _ = network.predict()
            result = detector.process(_, assigned_rect)
            
            logger.info(f"ObjectDetectionService: Raw YOLO returned {len(result.items)} detection(s)")
            return result
            
        except Exception as e:
            logger.error(f"ObjectDetectionService: Error running synap detection: {e}")
            return None
    
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
                # Filter to keep only person detections (class_index == 0)
                class_index = detection.get('class_index', -1)
                if class_index != 0:
                    logger.info(f"Filtering out non-person detection: class {class_index} ({detection.get('label', 'unknown')})")
                    continue
                
                # Direct mapping from YOLO detection results
                x = detection.get('x', 0)
                y = detection.get('y', 0)
                width = detection.get('width', 0)
                height = detection.get('height', 0)
                
                # Ensure coordinates are non-negative
                x = max(0, x)
                y = max(0, y)
                
                coordinates.append((x, y, width, height))
                
                logger.info(f"✅ Processed person coordinates for {detection.get('label', 'unknown')} "
                          f"(class {class_index}, confidence {detection.get('confidence', 0.0):.2f}): "
                          f"({x}, {y}, {width}, {height})")
            
            logger.info(f"👤 Filtered to {len(coordinates)} person detections (class 0 only)")
            return coordinates
            
        except Exception as e:
            logger.error(f"CoordinateProcessor: Error processing detection results: {e}")
            return []
    
    @staticmethod
    def filter_person_detections(json_result: str) -> List[dict]:
        """
        Filter JSON result to keep only person detections (class_index == 0)
        Returns the filtered detection data as dictionaries
        """
        try:
            detection_list = json.loads(json_result)
            person_detections = []
            
            for detection_data in detection_list:
                # Filter to keep only person detections (class_index == 0)
                class_index = detection_data.get('class_index', -1)
                if class_index != 0:
                    logger.info(f"Filtering out non-person object: class {class_index} ({detection_data.get('label', 'unknown')})")
                    continue
                
                person_detections.append(detection_data)
            
            logger.info(f"👤 Filtered {len(person_detections)} person detections (class 0 only)")
            return person_detections
            
        except Exception as e:
            logger.error(f"CoordinateProcessor: Error filtering person detections: {e}")
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

# Model configuration class
class ModelConfig:
    """Configuration for YOLO model selection"""
    
    # Available models
    YOLOV8S_SEG = "yolov8s_seg"  # YOLOv8s segmentation model
    YOLOV8L_SEG = "yolov8l_seg"  # YOLOv8l segmentation model
    
    # Model paths
    MODEL_PATHS = {
        YOLOV8S_SEG: "/usr/share/synap/models/object_detection/coco/model/yolov8s-seg-640x352/yolov8s_seg.synap",
        YOLOV8L_SEG: "/usr/share/synap/models/object_detection/coco/model/yolov8l-seg-640x352/yolov8l_seg.synap"
    }
    
    @classmethod
    def get_model_path(cls, model_type):
        """Get the model path for the specified model type"""
        return cls.MODEL_PATHS.get(model_type, cls.MODEL_PATHS[cls.YOLOV8L_SEG])  # Default to YOLOv8l
    
    @classmethod
    def get_available_models(cls):
        """Get list of available model types"""
        return list(cls.MODEL_PATHS.keys())

# Main detection coordinator class
class DetectionCoordinator:
    
    def __init__(self, model_type=ModelConfig.YOLOV8L_SEG):
        """
        Initialize DetectionCoordinator with specified model type
        
        Args:
            model_type: Model type to use (YOLOV8S_SEG or YOLOV8L_SEG)
        """
        self.frame_service = FrameCaptureService()
        self.detection_service = ObjectDetectionService()
        self.coordinate_processor = CoordinateProcessor()
        self.mask_processor = MaskProcessor()
        self.detection_results = []
        self.raw_detection_results = None  # Store raw synap results for mask processing
        self.segmented_full_image = None  # Store the full segmented image with white background
        self.load_thread = None
        
        # Set model type and path
        self.model_type = model_type
        self.model_path = ModelConfig.get_model_path(model_type)
        logger.info(f"DetectionCoordinator: Initialized with model type: {model_type}")
        logger.info(f"DetectionCoordinator: Using model path: {self.model_path}")
    
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
                    
                    # Filter person detections from JSON result
                    person_detections = self.coordinate_processor.filter_person_detections(json_result)
                    
                    # Store results in memory
                    self.detection_results = {
                        'coordinates': coordinates,
                        'person_detections': person_detections,
                        'json_result': json_result
                    }
                    
                    logger.info(f"DetectionCoordinator: Detection complete, found {len(coordinates)} items")
                    
                    # Call callback if provided
                    if callback:
                        callback(self.detection_results)
                        
                except Exception as e:
                    logger.error(f"DetectionCoordinator: Error in detection callback: {e}")
            
            # Prefer YOLO detection on the saved frame if available; fallback to empty result
            image_path = self.frame_service.get_last_frame_path()
            if image_path:
                logger.info(f"DetectionCoordinator: Starting YOLO detection on captured frame {image_path}")
                logger.info(f"DetectionCoordinator: Using model: {self.model_type} at {self.model_path}")
                
                # First, get raw detection results for mask processing
                raw_result = self.detection_service.yolo_od_raw(image_path, model_path=self.model_path)
                if raw_result:
                    self.raw_detection_results = raw_result
                    logger.info(f"DetectionCoordinator: Stored raw detection results with {len(raw_result.items)} items")
                    
                    # Generate the full segmented image once for all persons
                    self.generate_full_segmented_image(image_path, raw_result)
                
                # Then run async detection for JSON results
                self.detection_service.find_objects_from_image_async(image_path, model_path=self.model_path, callback=detection_callback)
            else:
                logger.warning("DetectionCoordinator: No saved frame path available, returning empty detection result")
                detection_callback("[]")
        
        else:
            logger.error("DetectionCoordinator: Failed to capture frame")
    
    def get_detection_results(self):
        """Get the last detection results"""
        return self.detection_results
    
    def reset(self):
        """Reset detection state"""
        self.frame_service.reset_capture_flag()
        self.detection_results = []
        self.raw_detection_results = None  # Clear raw detection results
        self.segmented_full_image = None  # Clear full segmented image
        if self.detection_service:
            self.detection_service.interrupt()
    
    def generate_mask_for_person(self, person_index: int) -> str:
        """
        Generate person crop for a specific person detection using segmentation masks
        
        Args:
            person_index: Index of the person in detection results
            
        Returns:
            str: Path to generated segmented person crop image, or None if failed
        """
        try:
            # Get original image path
            original_image_path = self.frame_service.get_last_frame_path()
            if not original_image_path:
                logger.error("DetectionCoordinator: No original image path available")
                return None
            
            # Try to use the pre-generated full segmented image for efficiency
            if self.segmented_full_image and os.path.exists(self.segmented_full_image):
                logger.info(f"DetectionCoordinator: Using pre-generated full segmented image for person {person_index}")
                
                # Get person detection coordinates for cropping
                if self.raw_detection_results and self.raw_detection_results.items:
                    person_detections = [d for d in self.raw_detection_results.items if d.class_index == 0]
                    
                    if person_index >= len(person_detections):
                        logger.warning(f"DetectionCoordinator: Person index {person_index} out of range (total: {len(person_detections)})")
                        return None
                    
                    # Get bounding box coordinates
                    detection = person_detections[person_index]
                    bb = detection.bounding_box
                    x = bb.origin.x
                    y = bb.origin.y
                    width = bb.size.x
                    height = bb.size.y
                    
                    # Load the full segmented image
                    import cv2
                    full_segmented_img = cv2.imread(self.segmented_full_image)
                    if full_segmented_img is None:
                        logger.error(f"DetectionCoordinator: Failed to load full segmented image")
                        return None
                    
                    # Crop the person region from the full segmented image
                    padding = 20  # Add some padding around the person
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(full_segmented_img.shape[1], x + width + padding)
                    y2 = min(full_segmented_img.shape[0], y + height + padding)
                    
                    if x2 > x1 and y2 > y1:
                        cropped_img = full_segmented_img[y1:y2, x1:x2]
                        
                        # Generate output path
                        cache_dir = "cache"
                        os.makedirs(cache_dir, exist_ok=True)
                        cache_key = f"person_segmented_{person_index}"
                        output_path = os.path.join(cache_dir, f"{cache_key}.png")
                        
                        # Save the cropped image
                        success = cv2.imwrite(output_path, cropped_img)
                        if success:
                            logger.info(f"DetectionCoordinator: Generated person crop {person_index} from full segmented image -> {output_path}")
                            return output_path
                        else:
                            logger.error(f"DetectionCoordinator: Failed to save cropped image to {output_path}")
                    else:
                        logger.warning(f"DetectionCoordinator: Invalid crop coordinates for person {person_index}")
            
            # Fallback to individual segmentation if full image not available
            if self.raw_detection_results and self.raw_detection_results.items:
                logger.info(f"DetectionCoordinator: Using individual segmentation for person {person_index}")
                
                # Filter to person detections only (class_index == 0)
                person_detections = [d for d in self.raw_detection_results.items if d.class_index == 0]
                
                if person_index >= len(person_detections):
                    logger.warning(f"DetectionCoordinator: Person index {person_index} out of range (total: {len(person_detections)})")
                    return None
                
                # Use the new color_bg function for precise segmentation
                try:
                    # Load the original image
                    import cv2
                    img = cv2.imread(original_image_path)
                    if img is None:
                        logger.error(f"DetectionCoordinator: Failed to load image {original_image_path}")
                        return None
                    
                    # Create a subset of detections with only the target person
                    target_detection = person_detections[person_index]
                    
                    # Create a mock DetectorResultItems-like object with only the target detection
                    class SingleDetectionWrapper:
                        def __init__(self, detection):
                            self.items = [detection]
                    
                    single_detection = SingleDetectionWrapper(target_detection)
                    
                    # Use the mask processor's color_bg function
                    segmented_img = self.mask_processor.color_bg(
                        img=img,
                        detections=single_detection,
                        bg_color=(255, 255, 255),  # White background
                        mode="apply_fg"  # Create new image, don't modify original
                    )
                    
                    # Generate output path
                    cache_dir = "cache"
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_key = f"person_segmented_{person_index}"
                    output_path = os.path.join(cache_dir, f"{cache_key}.png")
                    
                    # Save the segmented image
                    success = cv2.imwrite(output_path, segmented_img)
                    if success:
                        logger.info(f"DetectionCoordinator: Generated segmented person crop for person {person_index} using raw masks -> {output_path}")
                        return output_path
                    else:
                        logger.error(f"DetectionCoordinator: Failed to save segmented image to {output_path}")
                        
                except Exception as e:
                    logger.error(f"DetectionCoordinator: Error using raw detection results: {e}")
                    import traceback
                    logger.error(f"DetectionCoordinator: Traceback: {traceback.format_exc()}")
                    # Fall back to JSON-based method
            
            # Fallback to JSON-based method if raw results not available
            if not self.detection_results or 'json_result' not in self.detection_results:
                logger.warning("DetectionCoordinator: No detection results available")
                return None
            
            # Parse JSON results to get detection data
            detection_list = json.loads(self.detection_results['json_result'])
            
            # Filter to person detections only (class_index == 0)
            person_detections = [d for d in detection_list if d.get('class_index', -1) == 0]
            
            if person_index >= len(person_detections):
                logger.warning(f"DetectionCoordinator: Person index {person_index} out of range (total: {len(person_detections)})")
                return None
            
            # Get the specific person detection
            person_detection = person_detections[person_index]
            
            # Generate segmented person crop for this person using old method
            crop_path = self.mask_processor.generate_person_mask(
                detection_data=person_detection,
                original_image_path=original_image_path
            )
            
            if crop_path:
                logger.info(f"DetectionCoordinator: Generated segmented person crop for person {person_index} (fallback) -> {crop_path}")
            else:
                logger.error(f"DetectionCoordinator: Failed to generate segmented person crop for person {person_index}")
            
            return crop_path
            
        except Exception as e:
            logger.error(f"DetectionCoordinator: Error generating mask for person {person_index}: {e}")
            return None
    
    def get_mask_cache_info(self):
        """Get information about the mask processor cache"""
        return self.mask_processor.get_cache_info()
    
    def generate_full_segmented_image(self, image_path, raw_result):
        """Generate full segmented image with white background for all persons at once"""
        try:
            import cv2
            
            # Load the original image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"DetectionCoordinator: Failed to load image {image_path}")
                return
            
            # Filter to person detections only
            person_detections = [d for d in raw_result.items if d.class_index == 0]
            if not person_detections:
                logger.warning("DetectionCoordinator: No person detections found for full segmentation")
                return
            
            logger.info(f"DetectionCoordinator: Generating full segmented image for {len(person_detections)} persons")
            
            # Create white background
            white_bg = np.full(img.shape, (255, 255, 255), dtype=np.uint8)
            
            # Apply all person masks to the white background
            for i, detection in enumerate(person_detections):
                if detection.mask:
                    try:
                        mask_w, mask_h = detection.mask.width, detection.mask.height
                        mask = self.mask_processor.create_mask(
                            detection.mask.buffer(), mask_w, mask_h, img.shape[1], img.shape[0]
                        )
                        
                        if mask is not None:
                            # Copy person pixels to white background
                            white_bg[mask] = img[mask]
                            logger.info(f"DetectionCoordinator: Applied mask for person {i + 1}")
                        else:
                            logger.warning(f"DetectionCoordinator: Failed to create mask for person {i + 1}")
                    except Exception as e:
                        logger.error(f"DetectionCoordinator: Error applying mask for person {i + 1}: {e}")
            
            # Save the full segmented image
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)
            full_segmented_path = os.path.join(cache_dir, "full_segmented_image.png")
            
            success = cv2.imwrite(full_segmented_path, white_bg)
            if success:
                self.segmented_full_image = full_segmented_path
                logger.info(f"DetectionCoordinator: Generated full segmented image -> {full_segmented_path}")
            else:
                logger.error(f"DetectionCoordinator: Failed to save full segmented image")
                
        except Exception as e:
            logger.error(f"DetectionCoordinator: Error generating full segmented image: {e}")
    
    def clear_mask_cache(self):
        """Clear the mask processor cache"""
        self.mask_processor.clear_cache()
        logger.info("DetectionCoordinator: Mask cache cleared")
    
    def switch_model(self, model_type):
        """
        Switch to a different model type
        
        Args:
            model_type: New model type (YOLOV8S_SEG or YOLOV8L_SEG)
        """
        if model_type not in ModelConfig.get_available_models():
            logger.error(f"DetectionCoordinator: Invalid model type: {model_type}")
            logger.info(f"DetectionCoordinator: Available models: {ModelConfig.get_available_models()}")
            return False
        
        old_model = self.model_type
        self.model_type = model_type
        self.model_path = ModelConfig.get_model_path(model_type)
        
        logger.info(f"DetectionCoordinator: Switched from {old_model} to {model_type}")
        logger.info(f"DetectionCoordinator: New model path: {self.model_path}")
        
        # Clear cached results when switching models
        self.detection_results = []
        self.raw_detection_results = None
        self.segmented_full_image = None
        self.clear_mask_cache()
        
        return True
    
    def get_current_model_info(self):
        """Get information about the current model"""
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'available_models': ModelConfig.get_available_models()
        }


class MaskProcessor:
    """Process detection masks for individual persons using Synap segmentation"""
    
    def __init__(self):
        self.mask_cache = {}  # Cache for generated masks
        self.colors = {
            0: [255, 255, 255],  # White for person class
        }
    
    def filter_detections(self, detections):
        """Filter to get only human detections (class_index == 0)"""
        # Handle both DetectorResultItems and our wrapper
        if hasattr(detections, 'items'):
            items = detections.items
        else:
            items = detections
            
        if len(items) > 1:
            logger.info(f"MaskProcessor: Multiple detections found, selecting first human detection")
        
        for d in items:
            if d.class_index == 0:
                return d
            logger.info(f"MaskProcessor: Skipping non-human class ({d.class_index})")
        
        raise ValueError(f"No valid human detections found")
    
    def create_mask(self, mask_data, mask_w, mask_h, inp_w, inp_h, thresh=0):
        """Create boolean mask from segmentation probability data"""
        try:
            import cv2
            prob_mask = np.array(mask_data, dtype=np.float32).reshape((mask_h, mask_w))
            prob_mask = cv2.resize(prob_mask, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)
            return prob_mask > thresh
        except Exception as e:
            logger.error(f"MaskProcessor: Error creating mask: {e}")
            return None
    
    def apply_fg(self, img, mask, color):
        """Color-in background by copying over img pixels wherever mask is True"""
        new_img = np.full(img.shape, color, dtype=np.uint8)
        new_img[mask] = img[mask]
        return new_img
    
    def edit_bg(self, img, mask, color):
        """Color-in background by coloring pixels to color wherever mask is False"""
        img[~mask] = color
        return img
    
    def color_bg(self, img, detections, bg_color=(255, 255, 255), mode="edit_bg"):
        """Main function to color background using segmentation masks"""
        if mode not in ["apply_fg", "edit_bg"]:
            raise ValueError(f"Invalid bg coloring mode '{mode}'")
        if any(c < 0 or c > 255 for c in bg_color):
            raise ValueError(f"Invalid bg_color {bg_color}; all values must be 0-255")
        
        if not isinstance(img, np.ndarray):
            img = cv2.imread(img)
        else:
            if img.ndim == 4:
                img = img.squeeze(axis=0)
        
        inp_h, inp_w, _ = img.shape
        
        try:
            detection = self.filter_detections(detections)
            logger.info(f"MaskProcessor: Filtered detection class_index: {detection.class_index}")
            
            if not detection.mask:
                logger.warning("MaskProcessor: Detection doesn't contain mask data, returning original image")
                return img
            
            logger.info(f"MaskProcessor: Mask found, width: {detection.mask.width}, height: {detection.mask.height}")
            mask_w, mask_h = detection.mask.width, detection.mask.height
            mask = self.create_mask(detection.mask.buffer(), mask_w, mask_h, inp_w, inp_h)
            
            if mask is None:
                logger.warning("MaskProcessor: Failed to create mask, returning original image")
                return img
            
            if mode == "apply_fg":
                return self.apply_fg(img, mask, bg_color)
            return self.edit_bg(img, mask, bg_color)
            
        except Exception as e:
            logger.error(f"MaskProcessor: Error in color_bg: {e}")
            return img
    
    def generate_person_mask(self, detection_data: dict, original_image_path: str, output_path: str = None) -> str:
        """
        Generate segmented person crop with white background for a single person detection
        
        Args:
            detection_data: Single detection from YOLO result
            original_image_path: Path to original captured frame
            output_path: Optional output path for mask image
            
        Returns:
            str: Path to generated segmented person crop image
        """
        try:
            # Create cache key
            cache_key = f"{detection_data.get('x', 0)}_{detection_data.get('y', 0)}_{detection_data.get('width', 0)}_{detection_data.get('height', 0)}"
            
            # Check cache first
            if cache_key in self.mask_cache:
                logger.info(f"MaskProcessor: Using cached segmented person crop for detection {cache_key}")
                return self.mask_cache[cache_key]
            
            # Load original image
            if not os.path.exists(original_image_path):
                logger.error(f"MaskProcessor: Original image not found at {original_image_path}")
                return None
            
            img = cv2.imread(original_image_path)
            if img is None:
                logger.error(f"MaskProcessor: Failed to load image {original_image_path}")
                return None
            
            inp_h, inp_w, _ = img.shape
            
            # Extract detection coordinates
            x = detection_data.get('x', 0)
            y = detection_data.get('y', 0)
            width = detection_data.get('width', 0)
            height = detection_data.get('height', 0)
            
            # Calculate crop boundaries with padding
            padding = 20  # Add some padding around the person
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(inp_w, x + width + padding)
            y2 = min(inp_h, y + height + padding)
            
            if x2 > x1 and y2 > y1:
                # Crop the person region from the original image
                cropped_img = img[y1:y2, x1:x2]
                
                # Create a white background image with the same size as the cropped region
                white_bg = np.ones((y2-y1, x2-x1, 3), dtype=np.uint8) * 255
                
                # Try to use real segmentation mask if available
                mask = None
                if 'mask' in detection_data and detection_data['mask'] is not None:
                    try:
                        # Decode mask from base64
                        import base64
                        mask_info = detection_data['mask']
                        mask_b64 = mask_info['data']
                        mask_shape = mask_info['shape']
                        mask_dtype = mask_info['dtype']
                        
                        # Decode base64 to bytes, then to numpy array
                        mask_bytes = base64.b64decode(mask_b64)
                        mask_array = np.frombuffer(mask_bytes, dtype=np.dtype(mask_dtype)).reshape(mask_shape)
                        
                        # Resize mask to image size using PIL
                        from PIL import Image
                        mask_pil = Image.fromarray(mask_array.astype(np.float32))
                        mask_resized = mask_pil.resize((inp_w, inp_h), Image.Resampling.NEAREST)
                        mask_bool = np.array(mask_resized) > 0.5
                        
                        # Crop mask to the same region as the image
                        cropped_mask = mask_bool[y1:y2, x1:x2]
                        
                        logger.info(f"MaskProcessor: Using real segmentation mask, shape: {mask_array.shape}")
                        mask = cropped_mask
                        
                    except Exception as e:
                        logger.warning(f"MaskProcessor: Failed to decode real mask, falling back to elliptical mask: {e}")
                
                # Fallback to elliptical mask if no real mask available
                if mask is None:
                    logger.info(f"MaskProcessor: No real mask available, using elliptical mask")
                    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                    
                    # Calculate the person region within the cropped area
                    person_x1 = max(0, x - x1)  # Person start relative to crop
                    person_y1 = max(0, y - y1)  # Person start relative to crop
                    person_x2 = min(x2-x1, x + width - x1)  # Person end relative to crop
                    person_y2 = min(y2-y1, y + height - y1)  # Person end relative to crop
                    
                    # Create a simple elliptical mask for the person region
                    center_x = (person_x1 + person_x2) // 2
                    center_y = (person_y1 + person_y2) // 2
                    radius_x = (person_x2 - person_x1) // 2
                    radius_y = (person_y2 - person_y1) // 2
                    
                    # Create coordinate grids
                    y_coords, x_coords = np.ogrid[:y2-y1, :x2-x1]
                    
                    # Create elliptical mask
                    mask = ((x_coords - center_x) ** 2 / radius_x ** 2 + 
                           (y_coords - center_y) ** 2 / radius_y ** 2) <= 1.0
                    mask = mask.astype(np.uint8)
                
                # Apply mask: keep person pixels where mask is 1, white background elsewhere
                result_img = white_bg.copy()
                result_img[mask == 1] = cropped_img[mask == 1]
                
                # Generate output path if not provided
                if output_path is None:
                    cache_dir = "cache"
                    os.makedirs(cache_dir, exist_ok=True)
                    output_path = os.path.join(cache_dir, f"person_segmented_{cache_key}.png")
                
                # Save the result
                success = cv2.imwrite(output_path, result_img)
                if success:
                    logger.info(f"MaskProcessor: Generated segmented person crop at ({x}, {y}) size {width}x{height} -> {output_path}")
                    logger.info(f"MaskProcessor: Segmented crop size: {result_img.shape[1]}x{result_img.shape[0]}")
                    # Cache the result
                    self.mask_cache[cache_key] = output_path
                    return output_path
                else:
                    logger.error(f"MaskProcessor: Failed to save segmented person crop to {output_path}")
                    return None
            else:
                logger.warning(f"MaskProcessor: Invalid detection coordinates ({x}, {y}, {width}, {height})")
                return None
                
        except Exception as e:
            logger.error(f"MaskProcessor: Error generating segmented person crop: {e}")
            return None
    
    def clear_cache(self):
        """Clear the mask cache"""
        self.mask_cache.clear()
        logger.info("MaskProcessor: Cache cleared")
    
    def get_cache_info(self):
        """Get information about the current cache"""
        return {
            'cache_size': len(self.mask_cache),
            'cached_keys': list(self.mask_cache.keys())
        }
