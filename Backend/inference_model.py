import torch
import pennylane as qml
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
N_QUBITS = 8
NUM_Q_LAYERS = 5
IMG_SIZE = 224
MODEL_SAVE_PATH = 'best_quantum_hybrid_resnet_model.pth'
BATCH_SIZE = 1
NUM_WORKERS = 4

# Class names and their descriptions
CLASS_NAMES = ["Normal", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
CLASS_DESCRIPTIONS = {
    "Normal": "No tumor detected in the brain scan",
    "glioma_tumor": "Glioma tumor detected - a type of tumor that occurs in the brain and spinal cord",
    "meningioma_tumor": "Meningioma tumor detected - a tumor that arises from the meninges",
    "pituitary_tumor": "Pituitary tumor detected - a tumor that forms in the pituitary gland"
}

@dataclass
class PredictionResult:
    class_name: str
    class_description: str
    probabilities: Dict[str, float]
    processing_time: float
    confidence_score: float

# Setup device with fallback options
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

device = get_device()

# Quantum circuit definition with error handling
try:
    DEV = qml.device("default.qubit", wires=N_QUBITS)
    @qml.qnode(DEV, interface="torch", diff_method="backprop")
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        for i in range(NUM_Q_LAYERS):
            qml.BasicEntanglerLayers(weights[i], wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]
except Exception as e:
    logger.error(f"Error initializing quantum circuit: {str(e)}")
    raise

class QuantumHybridResNet(nn.Module):
    def __init__(self, num_classes_arg: int):
        super().__init__()
        try:
            self.classical = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs_resnet = self.classical.fc.in_features
            self.classical.fc = nn.Identity()
            self.fc_to_quantum = nn.Linear(num_ftrs_resnet, N_QUBITS)
            self.qlayer = qml.qnn.TorchLayer(
                quantum_circuit,
                weight_shapes={"weights": (NUM_Q_LAYERS, 1, N_QUBITS)},
                init_method={"weights": lambda tensor: nn.init.uniform_(tensor, 0, 2 * torch.pi)}
            )
            self.classifier = nn.Sequential(
                nn.Linear(num_ftrs_resnet + N_QUBITS, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes_arg)
            )
        except Exception as e:
            logger.error(f"Error initializing model architecture: {str(e)}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            classical_features = self.classical(x)
            q_in_features = self.fc_to_quantum(classical_features)
            q_in_normalized = torch.sigmoid(q_in_features) * torch.pi
            quantum_features = self.qlayer(q_in_normalized)
            combined_features = torch.cat([classical_features, quantum_features], dim=1)
            return self.classifier(combined_features)
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

# Image preprocessing with validation
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def validate_image(image: Image.Image) -> None:
    """Validate image dimensions and format."""
    if image.mode != 'RGB':
        raise ValueError("Image must be in RGB format")
    if image.size[0] < 32 or image.size[1] < 32:
        raise ValueError("Image dimensions too small")
    if image.size[0] > 4096 or image.size[1] > 4096:
        raise ValueError("Image dimensions too large")

def load_model() -> nn.Module:
    """Load the model with proper error handling and device placement."""
    try:
        logger.info("Loading model...")
        model = QuantumHybridResNet(num_classes_arg=len(CLASS_NAMES))
        
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")
            
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
        # Remove 'module.' prefix if present (for DataParallel models)
        if any(k.startswith('module.') for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load model once at startup
MODEL = load_model()

def predict(image: Image.Image) -> Dict[str, Any]:
    """
    Perform prediction on the input image with comprehensive error handling and logging.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing prediction results and metadata
    """
    start_time = time.time()
    
    try:
        logger.info("Starting prediction process...")
        
        # Validate image
        validate_image(image)
        logger.info("Image validation successful")
        
        # Preprocess image
        input_tensor = preprocess(image).unsqueeze(0).to(device, non_blocking=True)
        logger.info("Image preprocessing completed")
        
        # Perform prediction
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence_score = float(probabilities[predicted_class_idx])
            class_name = CLASS_NAMES[predicted_class_idx]
            
        processing_time = time.time() - start_time
        logger.info(f"Prediction completed in {processing_time:.2f} seconds")
        
        # Prepare result
        result = PredictionResult(
            class_name=class_name,
            class_description=CLASS_DESCRIPTIONS[class_name],
            probabilities={CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)},
            processing_time=processing_time,
            confidence_score=confidence_score
        )
        
        logger.info(f"Prediction result: {result}")
        return {
            "class_name": result.class_name,
            "class_description": result.class_description,
            "probabilities": result.probabilities,
            "processing_time": result.processing_time,
            "confidence_score": result.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
