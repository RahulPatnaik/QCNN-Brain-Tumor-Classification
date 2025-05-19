import torch
import pennylane as qml
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quantum circuit and model architecture (identical to training)
N_QUBITS = 8
NUM_Q_LAYERS = 5
IMG_SIZE = 224
MODEL_SAVE_PATH = 'best_quantum_hybrid_resnet_model.pth'

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Quantum circuit definition
DEV = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(DEV, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    for i in range(NUM_Q_LAYERS):
        qml.BasicEntanglerLayers(weights[i], wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

# Model class
class QuantumHybridResNet(nn.Module):
    def __init__(self, num_classes_arg):
        super().__init__()
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

    def forward(self, x):
        classical_features = self.classical(x)
        q_in_features = self.fc_to_quantum(classical_features)
        q_in_normalized = torch.sigmoid(q_in_features) * torch.pi
        quantum_features = self.qlayer(q_in_normalized)
        combined_features = torch.cat([classical_features, quantum_features], dim=1)
        out = self.classifier(combined_features)
        return out

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hardcoded class names as per classification report/model output order
CLASS_NAMES = ["Normal", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
NUM_CLASSES = len(CLASS_NAMES)

# Model loading function
def load_model():
    try:
        logger.info("Loading model...")
        model = QuantumHybridResNet(num_classes_arg=NUM_CLASSES)
        
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")
            
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
        # Remove 'module.' prefix if present
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

MODEL = load_model()

# Inference function
def predict(image: Image.Image):
    try:
        logger.info("Starting prediction...")
        input_tensor = preprocess(image).unsqueeze(0).to(device, non_blocking=True)
        
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            class_name = CLASS_NAMES[class_idx]
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        result = {
            "class_index": class_idx,
            "class_name": class_name,
            "probabilities": {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}
        }
        logger.info(f"Prediction completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
