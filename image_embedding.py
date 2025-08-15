#!/usr/bin/env python3
"""
FashionCLIP embedding system adapted from fashion_clip_embedding/main.py
Handles setup, image loading, model selection and embedding generation
"""

import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ImageEmbedding:
    """
    FashionCLIP embedding system adapted from fashion_clip_embedding/main.py
    Handles setup, image loading, model selection and embedding generation
    """
    
    def __init__(self, base_dir=None):
        """Initialize ImageEmbedding system"""
        self.base_dir = base_dir or os.getcwd()
        self.images_dir = os.path.join(self.base_dir, "images")
        self.embeddings_dir = os.path.join(self.base_dir, "embeddings")
        self.onnx_dir = os.path.join(self.base_dir, "onnx")
        
        # Model components
        self.infer_fn = None
        self.model = None
        self.processor = None
        self.images = []
        self.embeddings = None
        
        logger.info(f"ImageEmbedding: Initialized with base directory: {self.base_dir}")
    
    def should_initialize(self):
        """Check if initialization is needed (images directory doesn't exist)"""
        return not os.path.exists(self.images_dir)
    
    def setup_directories(self):
        """Setup cache and output directories"""
        logger.info("ImageEmbedding: Setting up directories...")
        
        # Create necessary directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)
        
        # Setup cache for datasets
        root_cache = os.path.join(self.base_dir, "cache")
        datasets_cache = os.path.join(root_cache, "datasets")
        os.makedirs(datasets_cache, exist_ok=True)
        
        return datasets_cache
    
    def load_deepfashion_images(self, data_split="train[:1000]", cache_dir=None):
        """Load images from DeepFashion dataset"""
        try:
            from datasets import load_dataset
            
            logger.info(f"ImageEmbedding: Loading DeepFashion dataset split: {data_split}")
            
            ds = load_dataset(
                "lirus18/deepfashion",
                split=data_split,
                cache_dir=cache_dir
            )
            
            self.images = [example["image"].convert("RGB") for example in ds]
            logger.info(f"ImageEmbedding: Loaded {len(self.images)} images")
            
            return self.images
            
        except Exception as e:
            logger.error(f"ImageEmbedding: Error loading DeepFashion dataset: {e}")
            return []
    
    def save_images(self, max_workers=8):
        """Save loaded images to disk with threading"""
        if not self.images:
            logger.warning("ImageEmbedding: No images to save")
            return
        
        try:
            from concurrent.futures import ThreadPoolExecutor
            from tqdm import tqdm
            
            logger.info(f"ImageEmbedding: Saving {len(self.images)} images...")
            
            def save_one(idx_img):
                idx, img = idx_img
                img.save(os.path.join(self.images_dir, f"img_{idx:03d}.png"))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(
                    executor.map(save_one, enumerate(self.images)), 
                    total=len(self.images), 
                    desc="Saving images (threaded)"
                ))
            
            logger.info("ImageEmbedding: Images saved successfully")
            
        except Exception as e:
            logger.error(f"ImageEmbedding: Error saving images: {e}")
    
    def load_model(self):
        """Load ONNX or HuggingFace FashionCLIP model"""
        try:
            int8_onnx_path = os.path.join(self.onnx_dir, "fashion_clip_image_int8_matmul.onnx")
            
            # Check if bitsandbytes is available for quantization
            try:
                import bitsandbytes
                bitsandbytes_available = True
            except ImportError:
                bitsandbytes_available = False
            
            # Try ONNX model first (faster inference)
            if os.path.exists(int8_onnx_path):
                logger.info("ImageEmbedding: Loading INT8 ONNX vision encoder from disk...")
                self._load_onnx_model(int8_onnx_path)
            else:
                logger.info("ImageEmbedding: Loading HuggingFace FashionCLIP model...")
                self._load_huggingface_model(bitsandbytes_available)
            
            logger.info("ImageEmbedding: Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"ImageEmbedding: Error loading model: {e}")
            return False
    
    def _load_onnx_model(self, onnx_path):
        """Load ONNX quantized model"""
        import onnxruntime as ort
        from transformers import CLIPProcessor
        
        self.ort_session = ort.InferenceSession(onnx_path)
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        
        def onnx_infer(images):
            inputs = self.processor(images=images, return_tensors="np")
            ort_inputs = {k: v for k, v in inputs.items() if k == "pixel_values"}
            ort_outs = self.ort_session.run(None, ort_inputs)
            return ort_outs[0]
        
        self.infer_fn = onnx_infer
    
    def _load_huggingface_model(self, bitsandbytes_available):
        """Load HuggingFace model with optional quantization"""
        import torch
        from transformers import CLIPModel, CLIPProcessor
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ImageEmbedding: Using device: {device}")
        
        if bitsandbytes_available and device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = CLIPModel.from_pretrained(
                "patrickjohncyh/fashion-clip",
                quantization_config=quantization_config,
                device_map="auto"
            )
            logger.info("ImageEmbedding: Using 8-bit quantization")
        else:
            self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
        
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        
        def hf_infer(images):
            inputs = self.processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                embeds = self.model.get_image_features(**inputs)
                return embeds.cpu().numpy()
        
        self.infer_fn = hf_infer
    
    def generate_embeddings(self):
        """Generate embeddings for loaded images"""
        if not self.images or not self.infer_fn:
            logger.error("ImageEmbedding: No images or model loaded")
            return None
        
        try:
            logger.info("ImageEmbedding: Computing image embeddings...")
            self.embeddings = self.infer_fn(self.images)
            
            # L2 normalize embeddings
            import numpy as np
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / norms
            
            logger.info(f"ImageEmbedding: Generated and normalized {self.embeddings.shape[0]} embeddings")
            
            # Save embeddings to disk
            self._save_embeddings()
            
            return self.embeddings
            
        except Exception as e:
            logger.error(f"ImageEmbedding: Error generating embeddings: {e}")
            return None
    
    def _save_embeddings(self):
        """Save embeddings to disk"""
        try:
            import numpy as np
            embeddings_file = os.path.join(self.embeddings_dir, "embeddings.npy")
            np.save(embeddings_file, self.embeddings)
            logger.info(f"ImageEmbedding: Embeddings saved to {embeddings_file}")
        except Exception as e:
            logger.error(f"ImageEmbedding: Error saving embeddings: {e}")
    
    def initialize_full_system(self):
        """Run full initialization: setup, load, save, model, embeddings"""
        if not self.should_initialize():
            logger.info("ImageEmbedding: Images directory exists, skipping initialization")
            return True
        
        try:
            # Step 1: Setup directories
            datasets_cache = self.setup_directories()
            
            # Step 2: Load images from DeepFashion
            if not self.load_deepfashion_images(cache_dir=datasets_cache):
                logger.error("ImageEmbedding: Failed to load images")
                return False
            
            # Step 3: Save images to disk
            self.save_images()
            
            # Step 4: Load model
            if not self.load_model():
                logger.error("ImageEmbedding: Failed to load model")
                return False
            
            # Step 5: Generate embeddings
            if self.generate_embeddings() is None:
                logger.error("ImageEmbedding: Failed to generate embeddings")
                return False
            
            logger.info("ImageEmbedding: Full system initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ImageEmbedding: Error during full initialization: {e}")
            return False
    
    def load_existing_embeddings(self):
        """Load existing embeddings from disk for similarity search"""
        try:
            import numpy as np
            embeddings_file = os.path.join(self.embeddings_dir, "embeddings.npy")
            
            if not os.path.exists(embeddings_file):
                logger.error("ImageEmbedding: No existing embeddings found")
                return False
            
            self.embeddings = np.load(embeddings_file)
            logger.info(f"ImageEmbedding: Loaded {self.embeddings.shape[0]} embeddings from disk")
            
            # Load model if not already loaded
            if self.infer_fn is None:
                if not self.load_model():
                    logger.error("ImageEmbedding: Failed to load model for similarity search")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"ImageEmbedding: Error loading existing embeddings: {e}")
            return False
    
    def compute_image_embedding(self, image):
        """Compute embedding for a single image (PIL Image or QPixmap)"""
        try:
            import numpy as np
            # Convert QPixmap to PIL Image if needed
            if hasattr(image, 'toImage'):  # QPixmap
                from PyQt5.QtGui import QImage
                from PIL import Image
                qimage = image.toImage()
                # Convert QImage to PIL Image
                width = qimage.width()
                height = qimage.height()
                ptr = qimage.bits()
                ptr.setsize(qimage.byteCount())
                arr = np.array(ptr).reshape(height, width, 4)  # RGBA
                pil_image = Image.fromarray(arr[:, :, :3])  # Remove alpha channel
            else:
                pil_image = image
            
            # Ensure RGB format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Generate embedding
            embedding = self.infer_fn([pil_image])
            
            # L2 normalize
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            normalized_embedding = embedding / norm
            
            return normalized_embedding[0]  # Return single embedding vector
            
        except Exception as e:
            logger.error(f"ImageEmbedding: Error computing image embedding: {e}")
            return None
    
    def find_similar_images(self, query_image, top_k=6):
        """Find top-k most similar images to the query image"""
        try:
            # Ensure embeddings are loaded
            if self.embeddings is None:
                if not self.load_existing_embeddings():
                    logger.error("ImageEmbedding: Cannot perform similarity search without embeddings")
                    return []
            
            # Compute embedding for query image
            query_embedding = self.compute_image_embedding(query_image)
            if query_embedding is None:
                logger.error("ImageEmbedding: Failed to compute query embedding")
                return []
            
            # Compute cosine similarities
            import numpy as np
            similarities = np.dot(self.embeddings, query_embedding)
            
            # Get top-k most similar indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_similarities = similarities[top_indices]
            
            # Prepare results
            results = []
            for i, (idx, similarity) in enumerate(zip(top_indices, top_similarities)):
                result = {
                    'rank': i + 1,
                    'index': int(idx),
                    'similarity': float(similarity),
                    'image_path': os.path.join(self.images_dir, f"img_{idx:03d}.png")
                }
                results.append(result)
            
            logger.info(f"ImageEmbedding: Found {len(results)} similar images (top-{top_k})")
            return results
            
        except Exception as e:
            logger.error(f"ImageEmbedding: Error in similarity search: {e}")
            return []
    
    def get_image_paths(self):
        """Get list of all image paths in the database"""
        try:
            image_paths = []
            if os.path.exists(self.images_dir):
                for i in range(1000):  # We have 1000 images
                    image_path = os.path.join(self.images_dir, f"img_{i:03d}.png")
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
            return image_paths
        except Exception as e:
            logger.error(f"ImageEmbedding: Error getting image paths: {e}")
            return []
