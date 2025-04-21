import torch
from torchvision import transforms, models
from PIL import Image
import os
from torchvision.models import densenet121


CLASS_LABELS =  ['Ajanta Caves', 'Ellora Caves', 'Fatehpur Sikri',
       'Gateway of India', 'Hawa mahal', 'Khajuraho', 'Sun Temple Konark',
       'alai_darwaza', 'alai_minar', 'basilica_of_bom_jesus', 'charminar',
       'golden temple', 'iron_pillar', 'jamali_kamali_tomb',
       'lotus_temple', 'mysore_palace', 'qutub_minar', 'tajmahal',
       'tanjavur temple', 'victoria memorial']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(model_path):
    # Load pre-trained DenseNet121 model
    model = models.densenet121(pretrained=False)
    
    # Modify the classifier layer to match your number of classes
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, len(CLASS_LABELS))  # Adjust classifier for your classes

    # Try loading the model state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # If keys mismatch, load model ignoring the mismatch
    model.load_state_dict(state_dict, strict=False)
    
    model.eval() 
    return model

def predict_image(model, image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return CLASS_LABELS[predicted.item()]