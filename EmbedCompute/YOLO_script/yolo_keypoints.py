import xml.etree.ElementTree as ET
import cv2
import numpy as np

# Path to your XML file
xml_file = 'C:\\Users\\Development\\Desktop\\LatestDocs\\Project\\Dataset_Prep\\YOLO_script\\tiger_dataset\\annotations.xml'
# Path to your image file
image_file = 'C:\\Users\\Development\\Desktop\\LatestDocs\\Project\\Dataset_Prep\\YOLO_script\\download_tiger.jpg'

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract image information
    image_info = root.find('image')
    image_id = image_info.get('id')
    image_name = image_info.get('name')
    image_width = int(image_info.get('width'))
    image_height = int(image_info.get('height'))

    # Extract bounding box information
    bounding_box = root.find('box')
    bbox_xmin = float(bounding_box.get('xtl'))
    bbox_ymin = float(bounding_box.get('ytl'))
    bbox_xmax = float(bounding_box.get('xbr'))
    bbox_ymax = float(bounding_box.get('ybr'))

    # Extract keypoints information
    keypoints = []
    keypoints_labels = []
    for points in root.findall('points'):
        label = points.get('label')
        points_str = points.get('points')
        points_list = [tuple(map(float, p.split(','))) for p in points_str.split(';')]
        keypoints.extend(points_list)
        keypoints_labels.extend([label] * len(points_list))

    return image_name, image_width, image_height, \
           (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax), \
           keypoints, keypoints_labels

def convert_to_yolo_format(image_width, image_height, bbox, keypoints):
    # Convert bounding box to YOLO format (normalized)
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    bbox_width = bbox_xmax - bbox_xmin
    bbox_height = bbox_ymax - bbox_ymin
    bbox_xcenter = bbox_xmin + bbox_width / 2
    bbox_ycenter = bbox_ymin + bbox_height / 2

    bbox_yolo = [bbox_xcenter / image_width, bbox_ycenter / image_height,
                 bbox_width / image_width, bbox_height / image_height]

    # Convert keypoints to YOLO format (normalized)
    keypoints_yolo = []
    for x, y in keypoints:
        keypoints_yolo.append([x / image_width, y / image_height])

    return bbox_yolo, keypoints_yolo

def visualize_keypoints(image_file, bbox, keypoints):
    image = cv2.imread(image_file)
    image_height, image_width, _ = image.shape

    # Draw bounding box
    bbox_xmin = int(bbox[0] * image_width)
    bbox_ymin = int(bbox[1] * image_height)
    bbox_xmax = int((bbox[0] + bbox[2]) * image_width)
    bbox_ymax = int((bbox[1] + bbox[3]) * image_height)
    cv2.rectangle(image, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), (0, 255, 0), 2)

    # Draw keypoints
    for (x, y) in keypoints:
        keypoint_x = int(x * image_width)
        keypoint_y = int(y * image_height)
        cv2.circle(image, (keypoint_x, keypoint_y), 3, (255, 0, 0), -1)

    # Display image
    cv2.imshow('YOLO Keypoints Example', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse XML file
    image_name, image_width, image_height, bbox, keypoints, keypoints_labels = parse_xml(xml_file)

    # Convert to YOLO format
    bbox_yolo, keypoints_yolo = convert_to_yolo_format(image_width, image_height, bbox, keypoints)

    # Visualize keypoints and bounding box
    visualize_keypoints(image_file, bbox_yolo, keypoints_yolo)
