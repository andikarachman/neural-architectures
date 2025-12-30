"""
YOLOv1: You Only Look Once - Unified, Real-Time Object Detection

YOLOv1 was introduced in "You Only Look Once: Unified, Real-Time Object Detection"
by Redmon et al. (2015) - revolutionized object detection with single-stage approach.

THE PARADIGM SHIFT: FROM TWO-STAGE TO SINGLE-STAGE
===================================================
Problem with traditional detectors (R-CNN, Fast R-CNN, Faster R-CNN):
    - Two-stage approach: Region proposals → Classification
    - Slow: Multiple passes over image, complex pipeline
    - Hard to optimize end-to-end

YOLO's Innovation: Frame detection as regression problem
    - Single neural network pass
    - Predicts bounding boxes AND class probabilities simultaneously
    - Extremely fast: 45 FPS, real-time capable

CORE CONCEPT: DIVIDE AND CONQUER
=================================
1. DIVIDE image into S×S grid (default: 7×7 = 49 cells)
2. Each grid cell predicts:
   - B bounding boxes (default: 2 boxes per cell)
   - Confidence score for each box
   - C class probabilities (shared across boxes in cell)

3. CONQUER: Post-process with Non-Maximum Suppression (NMS)

WHY THIS WORKS:
===============
1. GLOBAL CONTEXT: Unlike sliding windows, YOLO sees entire image
   - Can reason about objects and their context
   - Fewer background mistakes

2. UNIFIED ARCHITECTURE: Single network, end-to-end trainable
   - Simple pipeline: Image → CNN → Predictions
   - No separate region proposal network

3. EXTREMELY FAST: Single forward pass
   - Real-time detection on GPU
   - 1000× faster than R-CNN, 100× faster than Fast R-CNN

GRID CELL RESPONSIBILITY:
==========================
Each grid cell is "responsible" for detecting an object if:
    - The object's center falls within that cell

Example: Image with dog and cat
    - Dog's center at (120, 200) → Grid cell (2, 3) responsible
    - Cat's center at (350, 150) → Grid cell (5, 2) responsible

Each cell predicts B boxes, but only ONE box is responsible:
    - The box with highest IoU with ground truth during training
    - Other boxes learn to predict different objects or nothing

BOUNDING BOX REPRESENTATION:
============================
Each bounding box predicts 5 values: (x, y, w, h, confidence)

1. (x, y): Box center coordinates RELATIVE to grid cell
   - Range: [0, 1] within the cell
   - Example: x=0.5, y=0.5 means center of cell
   
2. (w, h): Box width and height RELATIVE to entire image
   - Range: [0, 1] as fraction of image dimensions
   - Example: w=0.3, h=0.4 means 30% width, 40% height
   
3. confidence: Pr(Object) × IoU^truth_pred
   - If no object: confidence should be 0
   - If object exists: confidence should equal IoU with ground truth

CLASS PROBABILITIES:
====================
Each grid cell predicts C class probabilities: Pr(Class_i | Object)
- Conditional probability: Given object exists, which class?
- Shared across all B boxes in the cell
- Only predicted once per cell (not per box)

FINAL DETECTION:
================
At test time, multiply class probabilities by box confidence:
    Score = Pr(Class_i | Object) × Pr(Object) × IoU
          = Pr(Class_i) × IoU

This gives class-specific confidence scores for each box.

CONCRETE EXAMPLE:
=================
Let's say we're looking at grid cell (3, 4) and the network outputs these 30 values:

Box 1: [x=0.3, y=0.7, w=0.4, h=0.5, confidence=0.8]
Box 2: [x=0.5, y=0.2, w=0.2, h=0.3, confidence=0.2]
Classes (20 values): [0.1, 0.05, 0.7, 0.05, 0.02, ..., 0.0] ← car=0.7 is highest

How to get final detections for this cell:

FOR BOX 1:
----------
Step 1: Get box confidence = 0.8
Step 2: Get class probabilities (SHARED for entire cell):
        [dog=0.1, cat=0.05, car=0.7, person=0.05, ...]
Step 3: Multiply to get class-specific scores:
        dog_score   = 0.8 × 0.1  = 0.08
        cat_score   = 0.8 × 0.05 = 0.04
        car_score   = 0.8 × 0.7  = 0.56  ← HIGHEST!
        person_score = 0.8 × 0.05 = 0.04
        ...
Step 4: Best class = "car" with confidence 0.56
Step 5: Convert to absolute coordinates:
        x_abs = (3 + 0.3) / 7 = 0.471  (47.1% from left)
        y_abs = (4 + 0.7) / 7 = 0.671  (67.1% from top)
        w_abs = 0.4  (40% of image width)
        h_abs = 0.5  (50% of image height)

Detection 1: "car" at [0.271, 0.421, 0.671, 0.921] with confidence 0.56

FOR BOX 2:
----------
Step 1: Get box confidence = 0.2
Step 2: Get SAME class probabilities (SHARED!):
        [dog=0.1, cat=0.05, car=0.7, person=0.05, ...]
Step 3: Multiply to get class-specific scores:
        dog_score = 0.2 × 0.1  = 0.02
        cat_score = 0.2 × 0.05 = 0.01
        car_score = 0.2 × 0.7  = 0.14  ← HIGHEST!
        ...
Step 4: Best class = "car" with confidence 0.14 (but might be too low)
Step 5: If confidence < threshold (e.g., 0.1), this detection is filtered out

KEY INSIGHT: Both boxes in the same cell predict the SAME class ("car")
because they share the same class probabilities. The boxes differ only in:
  - Location (x, y)
  - Size (w, h)  
  - Confidence (how sure there's an object there)

This is why YOLOv1 struggles with multiple objects of DIFFERENT classes 
in the same cell - the class probabilities are shared!

OUTPUT TENSOR SHAPE:
====================
For S=7, B=2, C=20 (Pascal VOC):
    Output: 7 × 7 × 30

Breakdown per grid cell (30 values):
    - Box 1: x, y, w, h, confidence     (5 values)
    - Box 2: x, y, w, h, confidence     (5 values)
    - Class probs: C1, C2, ..., C20     (20 values)
    Total: 5×2 + 20 = 30 values per cell

NETWORK ARCHITECTURE:
=====================
Based on GoogLeNet, customized for detection:

Layer Type          | Output Size  | Filters/Params       | Notes
--------------------|--------------|----------------------|------------------
Input               | 448×448×3    | -                    | 
Conv1 7×7/2         | 224×224×64   | 64 filters, stride=2 | 
MaxPool 2×2/2       | 112×112×64   | -                    |
Conv2 3×3           | 112×112×192  | 192 filters          |
MaxPool 2×2/2       | 56×56×192    | -                    |
Conv3 1×1           | 56×56×128    | 128 filters          |
Conv4 3×3           | 56×56×256    | 256 filters          |
Conv5 1×1           | 56×56×256    | 256 filters          |
Conv6 3×3           | 56×56×512    | 512 filters          |
MaxPool 2×2/2       | 28×28×512    | -                    |

[4× Inception-like blocks with 1×1 and 3×3 convs]

Conv15 3×3          | 14×14×1024   | 1024 filters         |
MaxPool 2×2/2       | 7×7×1024     | -                    |

[2× Conv blocks]

Conv23 3×3          | 7×7×1024     | 1024 filters         |
Conv24 3×3/2        | 7×7×1024     | 1024 filters         | stride=2? (paper unclear)

FC1                 | 4096         | Flatten + Dense      |
FC2                 | S×S×(B×5+C)  | Output reshape       | 7×7×30

Total: 24 conv layers + 2 FC layers = 26 weight layers

LOSS FUNCTION (Multi-Part):
============================
YOLO uses a custom loss that balances multiple objectives:

1. LOCALIZATION LOSS (bounding box coordinates):
   λ_coord × Σ Σ 𝟙^obj_ij [(x_i - x̂_i)² + (y_i - ŷ_i)²]
   λ_coord × Σ Σ 𝟙^obj_ij [(√w_i - √ŵ_i)² + (√h_i - √ĥ_i)²]
   
   - Only penalize if object exists in cell (𝟙^obj_ij)
   - Use sqrt for w, h to penalize errors in small boxes more
   - λ_coord = 5 (increase importance of localization)

2. CONFIDENCE LOSS (object exists):
   Σ Σ 𝟙^obj_ij (C_i - Ĉ_i)²
   
   - Penalize when object exists but confidence is low
   - C_i = IoU between predicted and ground truth box

3. CONFIDENCE LOSS (no object):
   λ_noobj × Σ Σ 𝟙^noobj_ij (C_i - Ĉ_i)²
   
   - Penalize when no object but confidence is high
   - λ_noobj = 0.5 (reduce importance, most cells have no object)

4. CLASSIFICATION LOSS:
   Σ 𝟙^obj_i Σ (p_i(c) - p̂_i(c))²
   
   - Only penalize if object exists in cell
   - Sum squared error over all classes

KEY HYPERPARAMETERS:
====================
- λ_coord = 5.0    : Weight for localization loss
- λ_noobj = 0.5    : Weight for no-object confidence loss
- S = 7            : Grid size (7×7 = 49 cells)
- B = 2            : Boxes per cell
- C = 20           : Number of classes (Pascal VOC)
- IoU threshold = 0.5 : For NMS

LIMITATIONS OF YOLOv1:
======================
1. STRUGGLES WITH SMALL OBJECTS: Each cell predicts only 2 boxes
   - Flock of birds: Multiple birds in one cell → only detects 2
   
2. STRUGGLES WITH NEW ASPECT RATIOS: Trained on specific shapes
   - Unusual box shapes may not be detected well
   
3. LOCALIZATION ERRORS: Main source of errors
   - Coarse grid (7×7) limits precision
   - Improved in YOLOv2 with finer grids

4. GROUP DETECTION ISSUES: Max B objects per cell
   - Objects close together in same cell compete

PERFORMANCE (Pascal VOC 2007):
==============================
Model          | mAP   | FPS  | Notes
---------------|-------|------|----------------------------------
R-CNN          | 66.0% | 0.05 | Very slow, but accurate
Fast R-CNN     | 66.9% | 0.5  | Faster, similar accuracy
Faster R-CNN   | 73.2% | 7    | State-of-art accuracy, slower
YOLOv1         | 63.4% | 45   | Real-time, slightly less accurate
Fast YOLO      | 52.7% | 155  | Even faster, lower accuracy

Trade-off: YOLO sacrifices some accuracy for massive speed improvement

IMPROVEMENTS IN LATER VERSIONS:
================================
YOLOv2 (YOLO9000, 2016):
    - Batch normalization
    - Higher resolution (416×416)
    - Anchor boxes (inspired by Faster R-CNN)
    - Multi-scale training
    - mAP: 76.8%, FPS: 67

YOLOv3 (2018):
    - Multi-scale predictions (3 scales)
    - Better backbone (Darknet-53)
    - Binary cross-entropy for classes (multi-label)
    - mAP: 57.9% @ 51 FPS (COCO)

YOLOv4-v8: Continued improvements in accuracy and speed

Reference:
    "You Only Look Once: Unified, Real-Time Object Detection"
    Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
    CVPR 2016
    https://arxiv.org/abs/1506.02640
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class YOLOv1(nn.Module):
    """
    YOLOv1: Single-stage object detector.
    
    Divides image into S×S grid, each cell predicts B bounding boxes
    and C class probabilities. Processes entire image in single pass.
    
    Architecture:
        - 24 convolutional layers (feature extraction)
        - 2 fully connected layers (detection head)
        - Inspired by GoogLeNet but simpler
    
    Output:
        Tensor of shape (batch, S, S, B*5 + C) where:
        - S×S: Grid size (7×7 cells)
        - B*5: Each box has (x, y, w, h, confidence)
        - C: Class probabilities per cell
    
    Args:
        num_classes (int): Number of object classes (default: 20 for Pascal VOC)
        grid_size (int): Grid size S (default: 7)
        num_boxes (int): Boxes per cell B (default: 2)
        dropout (float): Dropout probability (default: 0.5)
    
    Shape:
        - Input: (N, 3, 448, 448)
        - Output: (N, S, S, B*5+C) = (N, 7, 7, 30) for default settings
    
    Example:
        >>> model = YOLOv1(num_classes=20)
        >>> x = torch.randn(1, 3, 448, 448)
        >>> output = model(x)  # (1, 7, 7, 30)
    """
    
    def __init__(
        self,
        num_classes: int = 20,
        grid_size: int = 7,
        num_boxes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.dropout = dropout
        
        # Output channels per cell: B boxes × 5 values + C classes
        # Example: 2 × 5 + 20 = 30
        self.output_size = num_boxes * 5 + num_classes
        
        # Feature extraction: Conv layers (inspired by GoogLeNet)
        # Input: 448×448×3 → Output: 7×7×1024
        self.features = self._build_feature_extractor()
        
        # Detection head: FC layers
        # Input: 7×7×1024 = 50176 → Output: 7×7×30 = 1470
        self.detection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, grid_size * grid_size * self.output_size),
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_feature_extractor(self) -> nn.Sequential:
        """
        Build the convolutional feature extractor (24 conv layers).
        
        Architecture follows the original paper with some simplifications.
        Uses Leaky ReLU with slope 0.1 throughout.
        
        Key stages:
            1. Initial conv 7×7 with stride 2 → 224×224×64
            2. Series of conv blocks with max pooling
            3. Final output: 7×7×1024 feature map
        
        Returns:
            Sequential module containing feature extraction layers
        """
        layers = []
        in_channels = 3
        
        # Layer configuration: (out_channels, kernel_size, stride, padding)
        # 'M' indicates MaxPool2d(kernel_size=2, stride=2)
        architecture = [
            # Block 1: 448×448×3 → 224×224×64 → 112×112×64
            (64, 7, 2, 3),
            'M',
            
            # Block 2: 112×112×64 → 112×112×192 → 56×56×192
            (192, 3, 1, 1),
            'M',
            
            # Block 3: 56×56×192 → 56×56×512 → 28×28×512
            (128, 1, 1, 0),
            (256, 3, 1, 1),
            (256, 1, 1, 0),
            (512, 3, 1, 1),
            'M',
            
            # Block 4: 28×28×512 → 28×28×1024 → 14×14×1024
            # Multiple 1×1 and 3×3 conv (Inception-like)
            (256, 1, 1, 0),
            (512, 3, 1, 1),
            (256, 1, 1, 0),
            (512, 3, 1, 1),
            (256, 1, 1, 0),
            (512, 3, 1, 1),
            (256, 1, 1, 0),
            (512, 3, 1, 1),
            (512, 1, 1, 0),
            (1024, 3, 1, 1),
            'M',
            
            # Block 5: 14×14×1024 → 14×14×1024 → 7×7×1024
            (512, 1, 1, 0),
            (1024, 3, 1, 1),
            (512, 1, 1, 0),
            (1024, 3, 1, 1),
            (1024, 3, 1, 1),
            (1024, 3, 2, 1),  # stride=2 for spatial downsampling
            
            # Block 6: 7×7×1024 → 7×7×1024
            (1024, 3, 1, 1),
            (1024, 3, 1, 1),
        ]
        
        for layer_config in architecture:
            if layer_config == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.LeakyReLU(0.1, inplace=True))
                in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through YOLOv1.
        
        Args:
            x: Input images (N, 3, 448, 448)
            
        Returns:
            Predictions (N, S, S, B*5+C) where:
                - S: Grid size (7)
                - B: Number of boxes per cell (2)
                - For each box: (x, y, w, h, confidence)
                - C: Class probabilities (20)
                
        Example output shape for one cell (30 values):
            [box1_x, box1_y, box1_w, box1_h, box1_conf,
             box2_x, box2_y, box2_w, box2_h, box2_conf,
             class_prob_1, class_prob_2, ..., class_prob_20]
        """
        # Extract features: 448×448×3 → 7×7×1024
        features = self.features(x)
        
        # Detection head: 7×7×1024 → S×S×(B*5+C)
        output = self.detection_head(features)
        
        # Reshape to grid format
        # (N, S*S*30) → (N, S, S, 30)
        batch_size = x.size(0)
        output = output.view(
            batch_size, self.grid_size, self.grid_size, self.output_size
        )
        
        return output


class YOLOLoss(nn.Module):
    """
    YOLO multi-part loss function.
    
    Combines localization loss, confidence loss, and classification loss
    with different weights to balance their importance.
    
    Loss components:
        1. Bounding box coordinate loss (x, y)
        2. Bounding box size loss (w, h) - uses sqrt
        3. Confidence loss (object exists)
        4. Confidence loss (no object exists)
        5. Classification loss
    
    Args:
        lambda_coord (float): Weight for coordinate loss (default: 5.0)
        lambda_noobj (float): Weight for no-object confidence loss (default: 0.5)
        grid_size (int): Grid size S (default: 7)
        num_boxes (int): Boxes per cell B (default: 2)
        num_classes (int): Number of classes C (default: 20)
    
    Shape:
        - predictions: (N, S, S, B*5+C)
        - targets: (N, S, S, B*5+C) - same format as predictions
        
    Returns:
        Scalar loss value
    """
    
    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
    ) -> None:
        super().__init__()
        
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate YOLO loss.
        
        Args:
            predictions: Model output (N, S, S, B*5+C)
            targets: Ground truth (N, S, S, B*5+C)
        
        Returns:
            loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        # Reshape predictions for easier indexing
        # (N, S, S, 30) where 30 = [box1(5), box2(5), classes(20)]
        batch_size = predictions.size(0)
        
        # Extract predictions for each box and classes
        # Box 1: indices 0-4 (x, y, w, h, confidence)
        # Box 2: indices 5-9
        # Classes: indices 10-29
        box1_pred = predictions[..., :5]  # (N, S, S, 5)
        box2_pred = predictions[..., 5:10]  # (N, S, S, 5)
        class_pred = predictions[..., 10:]  # (N, S, S, C)
        
        # Extract targets
        box1_target = targets[..., :5]
        box2_target = targets[..., 5:10]
        class_target = targets[..., 10:]
        
        # Determine which box is responsible (higher IoU with ground truth)
        # In practice, during training, we select the box with highest IoU
        # For simplicity, we'll use confidence as indicator
        # obj_mask: (N, S, S, 1) - indicates if object exists in cell
        obj_mask1 = box1_target[..., 4:5] > 0  # confidence > 0
        obj_mask2 = box2_target[..., 4:5] > 0
        
        # Calculate IoU to determine responsible box
        # For simplicity, we'll use both boxes and weight by object mask
        # In full implementation, you'd calculate IoU and select best box
        
        # 1. COORDINATE LOSS (x, y) - only for responsible box
        coord_loss = 0
        for box_pred, box_target, obj_mask in [
            (box1_pred, box1_target, obj_mask1),
            (box2_pred, box2_target, obj_mask2),
        ]:
            xy_pred = box_pred[..., :2]  # (N, S, S, 2)
            xy_target = box_target[..., :2]
            
            # Only penalize if object exists in cell
            coord_loss += self.lambda_coord * self.mse(
                xy_pred * obj_mask,
                xy_target * obj_mask,
            )
        
        # 2. SIZE LOSS (w, h) - use sqrt to weight small boxes more
        size_loss = 0
        for box_pred, box_target, obj_mask in [
            (box1_pred, box1_target, obj_mask1),
            (box2_pred, box2_target, obj_mask2),
        ]:
            wh_pred = torch.sqrt(torch.abs(box_pred[..., 2:4]) + 1e-6)
            wh_target = torch.sqrt(torch.abs(box_target[..., 2:4]) + 1e-6)
            
            size_loss += self.lambda_coord * self.mse(
                wh_pred * obj_mask,
                wh_target * obj_mask,
            )
        
        # 3. CONFIDENCE LOSS (object exists)
        obj_conf_loss = 0
        for box_pred, box_target, obj_mask in [
            (box1_pred, box1_target, obj_mask1),
            (box2_pred, box2_target, obj_mask2),
        ]:
            conf_pred = box_pred[..., 4:5]
            conf_target = box_target[..., 4:5]
            
            obj_conf_loss += self.mse(
                conf_pred * obj_mask,
                conf_target * obj_mask,
            )
        
        # 4. CONFIDENCE LOSS (no object)
        noobj_conf_loss = 0
        for box_pred, box_target, obj_mask in [
            (box1_pred, box1_target, obj_mask1),
            (box2_pred, box2_target, obj_mask2),
        ]:
            conf_pred = box_pred[..., 4:5]
            noobj_mask = ~obj_mask  # Inverse of object mask
            
            noobj_conf_loss += self.lambda_noobj * self.mse(
                conf_pred * noobj_mask,
                torch.zeros_like(conf_pred) * noobj_mask,
            )
        
        # 5. CLASSIFICATION LOSS
        # Only penalize if object exists (use mask from either box)
        obj_mask_class = (obj_mask1 | obj_mask2).float()
        class_loss = self.mse(
            class_pred * obj_mask_class,
            class_target * obj_mask_class,
        )
        
        # Total loss
        total_loss = (
            coord_loss + size_loss + obj_conf_loss + noobj_conf_loss + class_loss
        )
        
        # Normalize by batch size
        total_loss = total_loss / batch_size
        
        # Return loss and components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'coord_loss': (coord_loss / batch_size).item(),
            'size_loss': (size_loss / batch_size).item(),
            'obj_conf_loss': (obj_conf_loss / batch_size).item(),
            'noobj_conf_loss': (noobj_conf_loss / batch_size).item(),
            'class_loss': (class_loss / batch_size).item(),
        }
        
        return total_loss, loss_dict


def decode_predictions(
    predictions: torch.Tensor,
    confidence_threshold: float = 0.1,
    iou_threshold: float = 0.5,
    grid_size: int = 7,
    num_boxes: int = 2,
    num_classes: int = 20,
) -> List[List[dict]]:
    """
    Decode YOLO predictions into bounding boxes.
    
    Converts raw network output to list of detected objects with
    coordinates, class, and confidence. Applies NMS to remove duplicates.
    
    Args:
        predictions: Model output (N, S, S, B*5+C)
        confidence_threshold: Minimum confidence to keep detection
        iou_threshold: IoU threshold for NMS
        grid_size: Grid size S
        num_boxes: Boxes per cell B
        num_classes: Number of classes C
    
    Returns:
        List of detections per image, each detection is a dict with:
            - 'bbox': [x1, y1, x2, y2] in absolute coordinates
            - 'class': Class index
            - 'confidence': Detection confidence
    """
    batch_size = predictions.size(0)
    detections = []
    
    for batch_idx in range(batch_size):
        pred = predictions[batch_idx]  # (S, S, B*5+C)
        image_detections = []
        
        # Iterate through each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                cell_pred = pred[i, j]  # (B*5+C,)
                
                # Extract boxes and class probabilities
                for b in range(num_boxes):
                    # Get box predictions
                    box_start = b * 5
                    x, y, w, h, conf = cell_pred[box_start:box_start+5]
                    
                    # Convert relative coordinates to absolute
                    # x, y are relative to cell, w, h relative to image
                    x_abs = (j + x) / grid_size  # Normalize to [0, 1]
                    y_abs = (i + y) / grid_size
                    
                    # Get class probabilities
                    class_probs = cell_pred[num_boxes*5:]  # (C,)
                    
                    # Calculate class-specific confidence scores
                    class_scores = conf * class_probs
                    
                    # Get best class
                    max_score, max_class = torch.max(class_scores, dim=0)
                    
                    # Filter by confidence threshold
                    if max_score > confidence_threshold:
                        # Convert to corner coordinates (x1, y1, x2, y2)
                        x1 = x_abs - w / 2
                        y1 = y_abs - h / 2
                        x2 = x_abs + w / 2
                        y2 = y_abs + h / 2
                        
                        image_detections.append({
                            'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                            'class': max_class.item(),
                            'confidence': max_score.item(),
                        })
        
        # Apply Non-Maximum Suppression (NMS)
        # This is a simplified version; full implementation would group by class
        image_detections = non_maximum_suppression(
            image_detections, iou_threshold
        )
        
        detections.append(image_detections)
    
    return detections


def non_maximum_suppression(
    detections: List[dict], iou_threshold: float = 0.5
) -> List[dict]:
    """
    Apply Non-Maximum Suppression to remove duplicate detections.
    
    Args:
        detections: List of detection dicts with 'bbox', 'class', 'confidence'
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while len(detections) > 0:
        # Keep highest confidence detection
        best = detections[0]
        keep.append(best)
        
        # Remove detections with high IoU overlap
        detections = [
            det for det in detections[1:]
            if calculate_iou(best['bbox'], det['bbox']) < iou_threshold
        ]
    
    return keep


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1, box2: Boxes in format [x1, y1, x2, y2]
    
    Returns:
        IoU value in [0, 1]
    """
    # Get intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def yolov1(num_classes: int = 20, pretrained: bool = False) -> YOLOv1:
    """
    Create YOLOv1 model for object detection.
    
    Args:
        num_classes (int): Number of object classes
        pretrained (bool): Load pretrained weights (not implemented)
    
    Returns:
        YOLOv1 model
    
    Example:
        >>> model = yolov1(num_classes=20)
        >>> x = torch.randn(1, 3, 448, 448)
        >>> output = model(x)  # (1, 7, 7, 30)
    """
    model = YOLOv1(num_classes=num_classes)
    
    if pretrained:
        raise NotImplementedError("Pretrained weights not available")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing YOLOv1...")
    model = yolov1(num_classes=20)
    x = torch.randn(2, 3, 448, 448)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test loss function
    print("\nTesting YOLOLoss...")
    criterion = YOLOLoss()
    targets = torch.randn_like(output)
    loss, loss_dict = criterion(output, targets)
    print(f"Loss: {loss.item():.4f}")
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Test decoding
    print("\nTesting prediction decoding...")
    detections = decode_predictions(output, confidence_threshold=0.1)
    print(f"Number of detections in batch: {[len(d) for d in detections]}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
