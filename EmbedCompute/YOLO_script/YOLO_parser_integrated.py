import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

global_label = None

# Define the parsing function
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_info = root.find('image')
    image_id = image_info.get('id')
    image_name = image_info.get('name')
    image_width = int(image_info.get('width'))
    image_height = int(image_info.get('height'))

    bounding_boxes = []
    keypoints_list = []
    for box in root.findall('box'):
        bbox_xmin = float(box.get('xtl'))
        bbox_ymin = float(box.get('ytl'))
        bbox_xmax = float(box.get('xbr'))
        bbox_ymax = float(box.get('ybr'))
        global_label = image_info.get('label')
        bounding_boxes.append((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))

        keypoints = []
        for points in box.findall('points'):
            points_str = points.get('points')
            points_list = [tuple(map(float, p.split(','))) for p in points_str.split(';')]
            keypoints.extend(points_list)
        keypoints_list.append(keypoints)

    return image_name, image_width, image_height, bounding_boxes, keypoints_list

# Convert the bounding box and keypoints to YOLO format
def convert_to_yolo_format(image_width, image_height, bbox, keypoints):
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    bbox_width = bbox_xmax - bbox_xmin
    bbox_height = bbox_ymax - bbox_ymin
    bbox_xcenter = bbox_xmin + bbox_width / 2
    bbox_ycenter = bbox_ymin + bbox_height / 2

    bbox_yolo = [bbox_xcenter / image_width, bbox_ycenter / image_height,
                 bbox_width / image_width, bbox_height / image_height]

    keypoints_yolo = []
    for x, y in keypoints:
        keypoints_yolo.append([x / image_width, y / image_height])

    return bbox_yolo, keypoints_yolo

# Define the YOLOv3 model
class YOLOv3(nn.Module):
    def __init__(self, num_classes, num_keypoints):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

        self.backbone = self._build_backbone()
        self.detector = self._build_detector()

    def _build_backbone(self):
        backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        return backbone

    def _build_detector(self):
        detector = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # Downsampling to 13x13
            nn.ReLU(),
            nn.Conv2d(1024, self.num_classes + 5 + 2 * self.num_keypoints, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        return detector

    def forward(self, x):
        x = self.backbone(x)
        detection_output = self.detector(x)
        return detection_output

# Initialize the model
#num_classes = 4  # For detecting nilgai, wild boar, macaque, tiger
#num_keypoints = 7  # Number of keypoints (left eye, right eye, nose, left ear, right ear, left mouth corner, right mouth corner, etc)

# Preprocess the test image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (416, 416))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    normalized_image = rgb_image / 255.0
    tensor_image = torch.tensor(normalized_image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return tensor_image

def run_inference(image_path, model):
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)
        
        # Process the output to get bbox_xcenter, bbox_ycenter, bbox_width, bbox_height, confidence, keypoints
        detections = []
        batch_size, _, grid_size, _ = output.shape
        output_view = output.view(batch_size, -1, grid_size * grid_size).permute(0, 2, 1)
        
        for det in output_view[0]:
            x, y, w, h, conf = det[:5]
            keypoints = None
            
            if len(det) > 5:
                keypoints_tensor = det[5:]
                
                if keypoints_tensor.numel() > 0:
                    # Ensure there are enough elements to reshape
                    if keypoints_tensor.numel() % 2 == 0:
                        keypoints = keypoints_tensor.reshape(-1, 2).tolist()
                    else:
                        # Handle odd number of elements if necessary
                        keypoints = keypoints_tensor[:-1].reshape(-1, 2).tolist()
            
            detections.append((x.item(), y.item(), w.item(), h.item(), conf.item(), keypoints))
        
    return detections, output

def process_yolo_output(detections_list, output, num_keypoints):
    detections = []
    num_classes = 7  # Assuming you have 7 classes

    grid_size = 13
    for batch_idx, batch_output in enumerate(output):
        batch_output = batch_output.to(torch.device('cpu'))  # Ensure it's on CPU if not already
        
        # Assuming batch_output is already processed and ready for interpretation
        for detection in batch_output:
            # Check the structure and access elements accordingly
            if detection.dim() == 2 and detection.shape[0] == grid_size * grid_size:
                bbox_xcenter = detection[0].item()
                bbox_ycenter = detection[1].item()
                bbox_width = detection[2].item()
                bbox_height = detection[3].item()
                confidence = detection[4].item()
                class_index = torch.argmax(detection[5:5 + num_classes]).item()

                # Extract keypoints if available
                keypoints = []
                if len(detection) > 5 + num_classes:  # Check if keypoints are present
                    keypoints_tensor = detection[5 + num_classes:]
                    if keypoints_tensor.numel() > 0:  # Ensure there are keypoints
                        keypoints = keypoints_tensor.reshape(-1, 2).tolist()

                detections.append((bbox_xcenter, bbox_ycenter, bbox_width, bbox_height, confidence, class_index, keypoints))
            else:
                # Handle cases where detection has unexpected dimensions or shapes
                print(f"Ignoring detection with unexpected shape: {detection.shape}")

    # Now you can use detections_list if needed, for example:
    for prev_detection in detections_list:
        # Compare or use previous detections as necessary
        pass
    
    return detections

def draw_detections(image_path, detections, save_path, class_name):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    print('Detections are ', detections)
    print('class_name is ', class_name)

    for detection in zip(detections):
        bbox_xcenter, bbox_ycenter, bbox_width, bbox_height, confidence, keypoints = detection
        print('Detection: bbox_xcenter: ', bbox_xcenter, 'bbox_ycenter: ', bbox_ycenter, 'bbox_width: ', bbox_width, 'bbox_height: ', bbox_height, 'confidence: ', confidence, 'class_index: ', class_index, 'keypoints: ', keypoints)
        xmin = int((bbox_xcenter - bbox_width / 2) * width)
        ymin = int((bbox_ycenter - bbox_height / 2) * height)
        xmax = int((bbox_xcenter + bbox_width / 2) * width)
        ymax = int((bbox_ycenter + bbox_height / 2) * height)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw keypoints
        for keypoint in keypoints:
            keypoint_x = int(keypoint[0] * width)
            keypoint_y = int(keypoint[1] * height)
            cv2.circle(image, (keypoint_x, keypoint_y), 3, (0, 0, 255), -1)

    cv2.imwrite(save_path, image)
    # OR
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Inferred Image')
    plt.show()


def process_images(image_dir, annotation_dir, save_dir, num_images=100):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpeg')])[:num_images]
    annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.xml')])[:num_images]

    model = YOLOv3(num_classes=7, num_keypoints=12)  # Example: 7 classes, 12 keypoints
    model.eval()

    for image_file, annotation_file in zip(image_files, annotation_files):
        image_path = os.path.join(image_dir, image_file)
        annotation_path = os.path.join(annotation_dir, annotation_file)
        save_path = os.path.join(save_dir, image_file)

        image_name, image_width, image_height, bounding_boxes, keypoints_list = parse_xml(annotation_path)
        bboxes_yolo = []
        keypoints_yolo = []

        for bbox, keypoints in zip(bounding_boxes, keypoints_list):
            bbox_yolo, keypoints_yolo_single = convert_to_yolo_format(image_width, image_height, bbox, keypoints)
            bboxes_yolo.append(bbox_yolo)
            keypoints_yolo.append(keypoints_yolo_single)

        output, output_shape = run_inference(image_path, model)
        detections = process_yolo_output(output, output_shape, num_keypoints=12)

        draw_detections(image_path, detections, save_path, global_label)


#Paths
image_dir = '/mnt/c/Users/Development/Desktop/LatestDocs/Project/Dataset_Prep/grayscale-images'
annotation_dir = '/mnt/c/Users/Development/Desktop/LatestDocs/Project/Dataset_Prep/YOLO_script/annotations'
save_dir = '/mnt/c/Users/Development/Desktop/LatestDocs/Project/Dataset_Prep/YOLO_script/output'

# Process images
process_images(image_dir, annotation_dir, save_dir)