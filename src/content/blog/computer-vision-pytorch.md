---
title: "Building Computer Vision Models with PyTorch: From CNNs to Vision Transformers"
description: "A comprehensive guide to modern computer vision techniques using PyTorch. Learn to build and train CNNs, implement data augmentation, and explore Vision Transformers."
author: alex-chen
publishDate: 2024-03-20
heroImage: https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=800&h=400&fit=crop
category: "Machine Learning"
tags: ["computer-vision", "pytorch", "cnn", "vision-transformer", "deep-learning"]
featured: false
draft: false
readingTime: 15
---

## Introduction

Computer vision has evolved dramatically from simple edge detection to sophisticated models that surpass human performance. This guide walks through building modern computer vision models with PyTorch, from traditional CNNs to cutting-edge Vision Transformers.

## Setting Up the Environment

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## Building a Modern CNN Architecture

Let's implement a ResNet-inspired architecture with modern improvements:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add shortcut
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ModernCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ModernCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, 2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
```

## Advanced Data Augmentation

Modern computer vision relies heavily on data augmentation:

```python
class AutoAugment:
    """Simplified AutoAugment policy for CIFAR-10"""
    def __init__(self):
        self.policies = [
            [('Rotate', 0.4, 30), ('TranslateX', 0.3, 0.2)],
            [('Color', 0.6, 0.5), ('Brightness', 0.3, 0.3)],
            [('Sharpness', 0.5, 0.8), ('Contrast', 0.3, 0.5)],
            [('Cutout', 0.7, 16), ('Equalize', 0.3, None)],
        ]
        
    def __call__(self, img):
        policy = np.random.choice(self.policies)
        for name, prob, magnitude in policy:
            if np.random.random() < prob:
                img = self._apply_augmentation(img, name, magnitude)
        return img
    
    def _apply_augmentation(self, img, name, magnitude):
        # Implementation of various augmentations
        # This is simplified - in practice, use torchvision.transforms
        return img

# Advanced augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MixUp augmentation
class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, data, targets):
        batch_size = data.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random shuffle for mixing
        index = torch.randperm(batch_size).to(data.device)
        
        # Mix data and targets
        mixed_data = lam * data + (1 - lam) * data[index]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_data, targets_a, targets_b, lam
```

## Vision Transformer Implementation

Let's implement a Vision Transformer (ViT) from scratch:

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12, 
                 mlp_ratio=4., dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Position embeddings and class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=depth
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x[:, 0])  # Use class token
        x = self.head(x)
        
        return x
```

## Training with Modern Techniques

```python
class VisionModelTrainer:
    def __init__(self, model, device, learning_rate=3e-4):
        self.model = model.to(device)
        self.device = device
        
        # AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.05
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # MixUp augmentation
        self.mixup = MixUp(alpha=1.0)
        
        # Gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Apply MixUp
            if np.random.random() > 0.5:
                data, targets_a, targets_b, lam = self.mixup(data, targets)
                
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = self.model(data)
                
                if 'targets_a' in locals():
                    # MixUp loss
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if 'targets_a' in locals():
                correct += (lam * predicted.eq(targets_a).sum().item() + 
                           (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                correct += predicted.eq(targets).sum().item()
                
        return total_loss / len(train_loader), 100. * correct / total
```

## Advanced Techniques

### 1. Attention Visualization

```python
def visualize_attention(model, image, device):
    """Visualize attention maps from Vision Transformer"""
    model.eval()
    
    with torch.no_grad():
        # Get attention weights
        x = image.unsqueeze(0).to(device)
        
        # Hook to capture attention weights
        attention_weights = []
        def hook_fn(module, input, output):
            attention_weights.append(output[1])
            
        # Register hook
        handle = model.transformer.layers[-1].self_attn.register_forward_hook(hook_fn)
        
        # Forward pass
        _ = model(x)
        handle.remove()
        
        # Process attention weights
        attn = attention_weights[0].squeeze(0)
        
        # Average over heads
        attn = attn.mean(dim=0)
        
        # Get attention for CLS token
        cls_attn = attn[0, 1:]  # Skip CLS token itself
        
        # Reshape to image dimensions
        H = W = int(np.sqrt(cls_attn.size(0)))
        cls_attn = cls_attn.reshape(H, W)
        
        return cls_attn.cpu().numpy()
```

### 2. Grad-CAM for CNNs

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_image, target_class):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Generate heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()
```

### 3. Self-Supervised Pre-training

```python
class SimCLR(nn.Module):
    """Simplified SimCLR for self-supervised learning"""
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.encoder = base_encoder
        
        # Remove the classification head
        self.encoder.fc = nn.Identity()
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
    def forward(self, x1, x2):
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        
        return z1, z2
    
    def contrastive_loss(self, z1, z2, temperature=0.5):
        batch_size = z1.shape[0]
        
        # Normalize
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # Compute similarity matrix
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create positive mask
        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z1.device)
        mask[:batch_size, batch_size:].fill_diagonal_(True)
        mask[batch_size:, :batch_size].fill_diagonal_(True)
        
        # Compute loss
        positives = similarity_matrix[mask].view(batch_size * 2, -1)
        negatives = similarity_matrix[~mask].view(batch_size * 2, -1)
        
        logits = torch.cat([positives, negatives], dim=1) / temperature
        labels = torch.zeros(batch_size * 2, dtype=torch.long).to(z1.device)
        
        return nn.functional.cross_entropy(logits, labels)
```

## Model Deployment

```python
# Export to ONNX
def export_to_onnx(model, input_shape=(1, 3, 224, 224), filename="model.onnx"):
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )

# TorchScript for production
def convert_to_torchscript(model, example_input):
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("model_traced.pt")
    return traced_model

# Quantization for edge deployment
def quantize_model(model, calibration_loader):
    model.eval()
    
    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with representative data
    with torch.no_grad():
        for data, _ in calibration_loader:
            model(data)
            
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    return model
```

## Best Practices and Tips

1. **Data Loading Optimization**
   - Use multiple workers in DataLoader
   - Pin memory for faster GPU transfer
   - Prefetch data to GPU

2. **Training Stability**
   - Use gradient accumulation for larger effective batch sizes
   - Monitor gradient norms
   - Implement early stopping

3. **Model Selection**
   - CNNs: Still excellent for smaller datasets and edge deployment
   - Vision Transformers: Superior for large-scale datasets
   - Hybrid approaches: Combine CNN and transformer strengths

## Conclusion

Modern computer vision has evolved far beyond simple convolutions. Whether you're building efficient CNNs for edge devices or scaling Vision Transformers for massive datasets, PyTorch provides the flexibility to implement state-of-the-art techniques. The key is understanding when to use each approach and how to optimize for your specific use case.

Start with proven architectures, experiment with data augmentation, and don't forget the importance of proper training techniques. Computer vision continues to advance rapidly, but these fundamentals will serve as your foundation for tackling any vision challenge.