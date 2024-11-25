import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from numpy.lib.stride_tricks import as_strided

IMAGE_DIR = "No_entry"
ANNOTATIONS_FILE = "annotations.txt"
CASCADE_FILE = "NoEntrycascade/cascade.xml"

@dataclass
class DetectionResult:
    image_name: str
    tpr: float
    f1: float
    num_detections: int
    num_ground_truth: int
    false_positives: int
    false_negatives: int

class DetectionEvaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def calculate_iou(box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        # Calculate the 4 intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        # Calculate the areas of both boxes
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]

        # Compute intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate the union area i.e sum of both areas minus intersection
        union = area1 + area2 - intersection
        
        # Return the IoU score while handling edge case of zero union
        return intersection / union if union > 0 else 0
    
    def evaluate(self, detections: List[Tuple[int, int, int, int]], 
                ground_truth: List[Tuple[int, int, int, int]]) -> DetectionResult:
        """ 
        For each ground truth box, we find the detection with highest IoU score, 
        and then if the score exceeds our IoU threshold, we count it as a true 
        positive.
        """
        # Count true positives by matching ground truth to detections
        true_positives = 0
        for gt_box in ground_truth:
            # Find detection with highest IoU for this ground truth box
            max_iou = max((self.calculate_iou(det_box, gt_box) 
                          for det_box in detections), default=0)
            if max_iou >= self.iou_threshold:
                true_positives += 1
        
        # Handle the edge case of no ground truth boxes
        if len(ground_truth) == 0:
            tpr = 1.0 if len(detections) == 0 else 0.0
        else:
            tpr = true_positives / len(ground_truth)
        
        # Calculate the precision and recall metrics
        precision = true_positives / len(detections) if len(detections) > 0 else 0
        recall = tpr
        
        # Calculate the F1 score, and handle division by zero
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate the false positives and negatives
        false_positives = max(0, len(detections) - true_positives)
        false_negatives = max(0, len(ground_truth) - true_positives)
        
        return DetectionResult(
            image_name="",
            tpr=tpr,
            f1=f1,
            num_detections=len(detections),
            num_ground_truth=len(ground_truth),
            false_positives=false_positives,
            false_negatives=false_negatives
        )
 
class GroundTruthLoader:
    def __init__(self, annotation_file: str):
        self.annotation_file = annotation_file
        self.ground_truth_dict = self._parse_annotation_file()
    
    def _parse_annotation_file(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        ground_truth_dict = {}

        try:
            # Read all the lines from the inputed annotation file
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                # Skip if invalid format (no coordinates found)
                if len(parts) < 2:
                    continue
                
                image_name = Path(parts[0]).name  # Get the raw filename
                num_boxes = int(parts[1])
                boxes = []
                
                # Parse each box's coordinates
                for i in range(num_boxes):
                    # Start at index 2 (after image_name and num_boxes)
                    # and multiply loop counter i by 4 to skip previous 
                    # boxes' coordinates
                    idx = 2 + (i * 4)
                    # Skip if not enough values remain for a complete 
                    # box (4 coordinates)
                    if idx + 3 >= len(parts):
                        break
                    
                    try:
                        # Convert a slice of 4 consecutive coordinates to 
                        # integers using map():
                        x, y, w, h = map(int, parts[idx:idx+4])
                        # If the box is valid (positive width and height) add it
                        if w > 0 and h > 0:
                            boxes.append((x, y, w, h))
                    except ValueError:
                        continue
                
                ground_truth_dict[image_name] = boxes
                
        except FileNotFoundError:
            print(f"Warning: Annotation file {self.annotation_file} not found")
            
        return ground_truth_dict
    
    def get_ground_truth(self, image_name: str) -> List[Tuple[int, int, int, int]]:
        return self.ground_truth_dict.get(image_name, [])

class ViolaJonesDetector:
    def __init__(self, cascade_path: str, config: Optional[Dict] = None):
         # Load our trained cascade classifier 
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise ValueError(f"Error: Could not load cascade classifier from {cascade_path}")
        
        self.config = {
            'SCALE_FACTOR': 1.05,
            'MIN_NEIGHBORS': 1,
            'MIN_SIZE': (15, 15),
            'MAX_SIZE': None
        }
    
    def detect(self, image: np.ndarray, return_debug: bool = False) -> Tuple[List[Tuple[int, int, int, int]], Optional[Dict]]:
        # Convert to grayscale and apply histogram equalization
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Returns a numpy array of detections, where each detection is an 
        # array of [x, y, width, height]
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.config['SCALE_FACTOR'],
            minNeighbors=self.config['MIN_NEIGHBORS'],
            minSize=self.config['MIN_SIZE'],
            maxSize=self.config['MAX_SIZE']
        )
        
        # Convert each numpy array detection to a tuple format to make it immutable
        boxes = [tuple(d) for d in detections] if len(detections) > 0 else []
        
        # Return detection results with no debug information
        return boxes, {}
    
class CircleDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'MIN_RADIUS': 15,
            'MAX_RADIUS': 65,
            'HOUGH_THRESHOLD_RATIO': 0.4,
            'MIN_CIRCLE_DISTANCE': 15,
            'GRADIENT_THRESHOLD_PERCENTAGE': 55,
            'BATCH_SIZE': 1000,
            'MAX_DETECTIONS': 10
        }
    
    def detect(self, image: np.ndarray, return_debug: bool = False) -> Tuple[List[Tuple[int, int, int, int]], Optional[Dict]]:
        """
        Detect circles in the input image and return their bounding boxes.
        """
        min_radius = self.config["MIN_RADIUS"]  
        max_radius = self.config["MAX_RADIUS"]

        grad_x, grad_y, grad_magnitude, grad_orientation = self._sobel_edge_detection(image)
        threshold_percentage = self.config['GRADIENT_THRESHOLD_PERCENTAGE']
        edges = self._threshold_gradient(grad_magnitude, threshold_percentage)
        
        # Apply circular Hough transform to detect potential circles
        accumulator = self._compute_circular_hough_accumulator(edges, grad_orientation, min_radius, max_radius)
        
        # Project accumulator from 3D to 2D
        hough_space_2d = np.sum(accumulator, axis=2)
        
        # Threshhold to find strong circle candidates
        threshold_ratio = self.config["HOUGH_THRESHOLD_RATIO"]    
        threshold_value = np.max(hough_space_2d) * threshold_ratio
        thresholded_hough = np.where(hough_space_2d > threshold_value, 255, 0).astype(np.uint8)
        
        # Find the circle parameters from the accumulator peaks
        circles = self._find_circles(accumulator, min_radius, threshold_ratio, min_distance=20)
        
        # Convert circles to the bounding boxes
        boxes = []
        for x, y, r in circles:
            # Ensure the boxes stay within the image boundaries
            box_x = max(0, x - r)
            box_y = max(0, y - r)
            box_w = min(2*r, image.shape[1] - box_x)
            box_h = min(2*r, image.shape[0] - box_y)
            boxes.append((box_x, box_y, box_w, box_h))

        # Create the debug images
        if return_debug:
            debug_info = self._create_debug_visualization(
                image=image,
                boxes=boxes,
                circles=circles,
                edges=edges,
                thresholded_hough=thresholded_hough
            )
            return boxes, debug_info
        
        return boxes, None

    def _create_debug_visualization(
        self, 
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        circles: List[Tuple[int, int, int]],
        edges: np.ndarray,
        thresholded_hough: np.ndarray
    ) -> Dict:
        # Create copies of the input image for visualization overlays
        circle_viz = image.copy()  # For drawing detected circles
        box_viz = image.copy()     # For drawing bounding boxes
        
        # Draw our detected circles in green
        for x, y, r in circles:
            cv2.circle(circle_viz, (x, y), r, (0, 255, 0), 2)
        
        # Draw our ground truth boxes in red
        for box_x, box_y, box_w, box_h in boxes:
            cv2.rectangle(
                box_viz,
                (box_x, box_y),                    
                (box_x + box_w, box_y + box_h),   
                (0, 0, 255),                      
                2                     
            )
    
        return {
            'edges': edges,                         
            'thresholded_hough': thresholded_hough,
            'circle_detections': circle_viz,    
        }

    def _sobel_edge_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Convert image to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (7, 7), 1.5)
        
        # Define the Sobel kernels for gradient computation
        kernel_x = np.array([[-1, 0, 1],  
                            [-2, 0, 2],    
                            [-1, 0, 1]])   
        
        kernel_y = np.array([[-1, -2, -1],  
                            [0, 0, 0],      
                            [1, 2, 1]])    
        
        # Compute the vertical and horizontal gradient
        grad_x = self._fast_convolve2d(image, kernel_x)
        grad_y = self._fast_convolve2d(image, kernel_y)

        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        grad_direction = np.arctan2(grad_y, grad_x)
        
        return grad_x, grad_y, grad_magnitude, grad_direction

    def _compute_circular_hough_accumulator(self, edge_image: np.ndarray, grad_orientation: np.ndarray,
                        min_radius: int, max_radius: int) -> np.ndarray:
        # Initialize 3D voting space
        height, width = edge_image.shape
        num_radii = max_radius - min_radius + 1
        hough_space = np.zeros((height, width, num_radii), dtype=np.uint32)
        
        # Get coordinates of edge pixels and create array of radii to check
        y_coords, x_coords = np.nonzero(edge_image)  # Find all edge pixel coordinates
        radii = np.arange(min_radius, max_radius + 1)  # All possible radii
        
        # Process edge points in batches
        for start_idx in range(0, len(y_coords), self.config['BATCH_SIZE']):
            # Get current batch of edge points
            end_idx = min(start_idx + self.config['BATCH_SIZE'], len(y_coords))
            
            # Extract coordinates and gradient angles for current batch
            y_batch = y_coords[start_idx:end_idx]
            x_batch = x_coords[start_idx:end_idx]
            angles = grad_orientation[y_batch, x_batch]
            
            # For each radius, compute potential circle centers
            for r_idx, radius in enumerate(radii):
                # Calculate potential circle center coordinates
                # Use gradient direction to vote in the direction perpendicular to edge
                x_centers = (x_batch + radius * np.cos(angles)).astype(int)
                y_centers = (y_batch + radius * np.sin(angles)).astype(int)
                
                # Filter out centers that would create circles outside image bounds
                valid_points = (x_centers >= 0) & (x_centers < width) & \
                            (y_centers >= 0) & (y_centers < height)
                
                # Keep only valid circle centers
                x_valid = x_centers[valid_points]
                y_valid = y_centers[valid_points]
                
                # Increment votes for valid circle centers at current radius
                # np.add.at handles multiple votes at same coordinates
                np.add.at(hough_space[:, :, r_idx], (y_valid, x_valid), 1)
        
        return hough_space

    def _find_circles(self, accumulator: np.ndarray, min_radius: int, 
                     threshold_ratio: float, min_distance: int = 30) -> List[Tuple[int, int, int]]:
        height, width, _ = accumulator.shape
        
        # Project accumulator from 3D to 2D
        votes_2d = np.sum(accumulator, axis=2)
        
        # Threshhold to find strong circle candidates
        threshold = np.max(votes_2d) * threshold_ratio
        peak_candidates = votes_2d > threshold
        
        # Perform non-maximum suppression to find a maximum number of 
        # distinct circle centers   
        circles = []
        for y in range(height):
            for x in range(width):
                if peak_candidates[y, x]:
                    # Extract local neighborhood around potentiall circle centers
                    neighborhoo_y_start = max(0, y - min_distance//2)
                    neighborhoo_y_end = min(height, y + min_distance//2 + 1)
                    neighborhoo_x_start = max(0, x - min_distance//2)
                    neighborhoo_x_end = min(width, x + min_distance//2 + 1)
                    
                    local_neighborhoo = votes_2d[neighborhoo_y_start:neighborhoo_y_end,
                                            neighborhoo_x_start:neighborhoo_x_end]
                    
                    # Check if current circle centre is the local maximum in its neighborhood
                    if votes_2d[y, x] == np.max(local_neighborhoo):
                        # Find the best radius for this circle centre
                        r_idx = np.argmax(accumulator[y, x, :])
                        radius = r_idx + min_radius
                        strength = votes_2d[y, x]
                        circles.append((x, y, radius, strength))
        
        # Sort circles by detection strength and limit to a maximum number
        circles.sort(key=lambda x: x[3], reverse=True)  # Sort by strength (4th element)
        circles = circles[:self.config['MAX_DETECTIONS']]
   
        # Return only the circle parameters (x, y, r) without strength
        return [(x, y, r) for x, y, r, _ in circles]

    def _threshold_gradient(self, grad_magnitude: np.ndarray, threshold_percentage: float) -> np.ndarray:
        """
        Convert gradient magnitude image to binary edge map
        """
        # np.percentile finds the value below which threshold_percentage of pixels fall
        # E.g. If threshold_percentage is 55, then we keep the top 45% of strongest gradients
        threshold = np.percentile(grad_magnitude, threshold_percentage)
         
        # Create and return the binary edge map
        return np.where(grad_magnitude > threshold, 255, 0).astype(np.uint8)
    
    def _fast_convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs fast 2D convolution using stride tricks and einsum for better performance.
        """
        # Flip the kernel for convolution
        kernel = np.flipud(np.fliplr(kernel))
        
        # Adds padding around the image edges to handle boundary pixels
        padding = kernel.shape[0] // 2
        padded = np.pad(image, padding, mode='constant')  # Zero padding
        
        # Create a 4D view of the image where each pixel has its own neighborhood
        # Avoids explicit loops by using numpy's stride tricks
        shape = (image.shape[0], image.shape[1], kernel.shape[0], kernel.shape[1])
        strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
        
        # Create sliding windows view of padded image
        # This uses numpy's stride tricks to avoid copying data
        windows = as_strided(padded, shape=shape, strides=strides)
        
        # Compute convolution using einstein summation
        output = np.einsum('ijkl,kl->ij', windows, kernel)
        
        return output

class RedDetector:
    def __init__(self):
        self.thresholds = {
            'hsv_lower1': np.array([0, 70, 50]),
            'hsv_upper1': np.array([10, 255, 255]),
            'hsv_lower2': np.array([160, 70, 50]),
            'hsv_upper2': np.array([180, 255, 255]),
            'lab_threshold': 150,
            'red_ratio': 0.5,
            'min_area': 200,
            'min_area_ratio': 0.001,
            'max_area_ratio': 0.2
        }
        self.kernel = np.ones((3,3), np.uint8)
        
    def detect(self, image: np.ndarray, return_debug: bool = False) -> Tuple[List[Tuple[int, int, int, int]], Optional[Dict]]:
        # Generate a binary mask identifying red regions
        combined_mask = self._get_red_mask(image)

        # Clean up the mask using morphological operations
        combined_mask = self._apply_morphology(combined_mask)

        # Convert the mask regions into bounding boxes
        boxes = self._find_boxes(combined_mask, image.shape)
        
        if return_debug:
            debug_info = self._create_debug_visualization(image, boxes, combined_mask)
            return boxes, debug_info
        
        return boxes, None

    def _get_red_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask highlighting red regions by combining multiple color space filters.
        """
        # Get a mask based on HSV color space thresholds
        hsv_mask = self._get_hsv_mask(image)
        
        # Get a mask based on LAB color space
        lab_mask = self._get_lab_mask(image)
        
        # Get a mask based on RGB channel ratios
        ratio_mask = self._get_ratio_mask(image)
        
        # Combine the masks using logical AND operations
        combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
        combined_mask = cv2.bitwise_and(combined_mask, ratio_mask)
        
        return combined_mask
    
    def _get_hsv_mask(self, image: np.ndarray) -> np.ndarray:
        # Convert the BGR image to an HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Mask for the lower red hues
        red_mask_l = cv2.inRange(hsv, self.thresholds['hsv_lower1'], 
                                    self.thresholds['hsv_upper1'])
        
        # Mask for the upper red hues
        red_mask_u = cv2.inRange(hsv, self.thresholds['hsv_lower2'], 
                                    self.thresholds['hsv_upper2'])
        
        # Combine the two masks using a logical OR operations
        return cv2.bitwise_or(red_mask_l, red_mask_u)

    def _get_lab_mask(self, image: np.ndarray) -> np.ndarray:
        # Convert the BGR image to a LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract just the 'a' channel which represents red-green
        _, a_channel, _ = cv2.split(lab)
        
        # Create a binary mask in which the pixels above our 
        # threshold are considered red
        _, red_mask_lab = cv2.threshold(a_channel, self.thresholds['lab_threshold'], 255, 
                                        cv2.THRESH_BINARY)
        return red_mask_lab  # Add this return statement

    def _get_ratio_mask(self, image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image)
        
        # Prevents divison by zero
        small_num = 1e-6
        
        red_ratio = r.astype(float) / (b.astype(float) + g.astype(float) + small_num) 
        
        # Create a binary mask where the pixel is set to 255 if the red
        # ratio exceeds our threshold
        return np.where(red_ratio > self.thresholds['red_ratio'], 255, 0).astype(np.uint8)

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Cleans up the binary mask using morphological operations.
        Uses opening to remove the small noise and closing to fill the small holes,
        """
        # Apply opening (erosion followed by dilation)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # Apply morphological closing (dilation followed by erosion)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        return mask

    def _find_boxes(self, mask: np.ndarray, image_shape: Tuple[int, ...]) -> List[Tuple[int, int, int, int]]:
        # Find the continuous regions (contours) in our binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        
        for contour in contours:
            # Calculate the absolute area of the contour in the pixels
            area = cv2.contourArea(contour)
            # Skip if region is too small (likely noise)
            if area < self.thresholds['min_area']:
                continue
                
            # Get the bounding rectangle coordinates
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate the region's area as fraction of the total image area
            area_ratio = (w * h) / (image_shape[0] * image_shape[1])
            
            # Skip if region is too small or too large relative to image
            if area_ratio < self.thresholds['min_area_ratio'] or area_ratio > self.thresholds['max_area_ratio']:
                continue
                
            boxes.append((x, y, w, h))
        return boxes

    def _create_debug_visualization(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        combined_mask: np.ndarray
    ) -> Dict:
        detection_viz = image.copy()
        box_viz = image.copy()
        overlay_viz = image.copy()
        
        # Generate the individual color space masks
        hsv_mask = self._get_hsv_mask(image)
        lab_mask = self._get_lab_mask(image)
        
        # Convert the grayscale masks to BGR for visualization
        hsv_mask_viz = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
        lab_mask_viz = cv2.cvtColor(lab_mask, cv2.COLOR_GRAY2BGR)
        
        # Draw detection boxes
        for box_x, box_y, box_w, box_h in boxes:
            # Draw a green rectangle on the detection visualization
            cv2.rectangle(detection_viz, (box_x, box_y), 
                        (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
            
            # Draw a red rectangle on the ground truth visualization
            cv2.rectangle(box_viz, (box_x, box_y), 
                        (box_x + box_w, box_y + box_h), (0, 0, 255), 2)
        
        return {
            'hsv_mask': hsv_mask_viz,         
            'lab_mask': lab_mask_viz,         
            'detections': detection_viz,     
        }

class DetectionMerger:
    def __init__(self, distance_threshold: float = 0.5, iou_threshold: float = 0.4):
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def _calculate_box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        # Add half the width to x and half the height to y to get the center
        # point of the box
        return (box[0] + box[2]//2, box[1] + box[3]//2)
    
    @staticmethod
    def _calculate_center_distance(center1: Tuple[float, float], 
                                 center2: Tuple[float, float]) -> float:
        # Uses Pythagorean theorem to get the distance between two points
        return ((center1[0] - center2[0])**2 + 
                (center1[1] - center2[1])**2)**0.5
    
    def _merge_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """
        Merge multiple boxes into a single box by averaging their coordinates and dimensions.
        """
        x = int(np.mean([box[0] for box in boxes]))
        y = int(np.mean([box[1] for box in boxes]))
        w = int(np.mean([box[2] for box in boxes]))
        h = int(np.mean([box[3] for box in boxes]))
        return (x, y, w, h)
    
    def merge_detections(self, detection_lists: List[List[Tuple[int, int, int, int]]]) -> List[Tuple[int, int, int, int]]:
        """
        Merge multiple lists of detections into a single list, combining overlapping boxes.
        """
        # Return empty list if no detections provided
        if not detection_lists or not any(detection_lists):
            return []
        
        final_detections = []  # Will store our merged results
        # Track which of the detections we've already used in merging
        used_detections = [set() for _ in detection_lists]
        
        # Use the first detection list as the primary detections and try to match others to it
        for primary_idx, primary_box in enumerate(detection_lists[0]):
            # Calculate the center of the primary detections box for distance comparisons
            primary_center = self._calculate_box_center(primary_box)
            
            best_matches = []
            best_ious = []   
            best_indices = []
            
            # Look through all our other detection lists
            for list_idx in range(1, len(detection_lists)):
                best_match = None
                best_iou = 0
                best_idx = None
                
                # Compare each of the box's in the current list to the primary box
                for box_idx, box in enumerate(detection_lists[list_idx]):
                    # Skip if we've already used this detection
                    if box_idx in used_detections[list_idx]:
                        continue
                    
                    # Calculate the distance and overlap between the boxes
                    box_center = self._calculate_box_center(box)
                    dist = self._calculate_center_distance(primary_center, box_center)
                    iou = DetectionEvaluator.calculate_iou(primary_box, box)
                    
                    # If the boxes are close enough and have a better overlap than the previous best
                    # update the stored comparison variables
                    if (dist < max(primary_box[2], box[2]) * self.distance_threshold and 
                        iou > best_iou):
                        best_match = box
                        best_iou = iou
                        best_idx = box_idx
                
                best_matches.append(best_match)
                best_ious.append(best_iou)
                best_indices.append(best_idx)
            
            # Find which of the matches have enough overlap to be considered the same object
            valid_matches = [i for i, iou in enumerate(best_ious) 
                            if iou > self.iou_threshold]
            
            # If we have found any valid matches, merge them
            if valid_matches:
                # Mark the matched detections as used detections
                for list_idx, box_idx in enumerate(best_indices):
                    if box_idx is not None and list_idx in valid_matches:
                        used_detections[list_idx + 1].add(box_idx)
                
                # Combine the primary box with all the other valid matches
                boxes_to_merge = [primary_box] + [best_matches[i] 
                                                for i in valid_matches]
                merged_box = self._merge_boxes(boxes_to_merge)
                final_detections.append(merged_box)
        
        return final_detections
    
class CombinedDetector:
    """
    A wrapper class that combines multiple detectors and merges their results.
    """
    def __init__(self, detectors: List[Any], merger: DetectionMerger):
        self.detectors = detectors # List of the individual detectors
        self.merger = merger       # For merging overlapping detections
    
    def detect(self, image: np.ndarray, return_debug: bool = False):
        # Lists to store the results from each of the detectors
        all_boxes = []
        all_debug = {}
        
        # Run each detector on the image
        for detector in self.detectors:
            # Get the boxes and debug info from the current detector
            boxes, debug = detector.detect(image, return_debug=True)
            all_boxes.append(boxes)
            # Add debug images to the combined dict if any were returned
            if debug:
                all_debug.update(debug)
        
        # Merge the overlapping detections from the different detectors
        merged_boxes = self.merger.merge_detections(all_boxes)
        
        if return_debug:
            return merged_boxes, all_debug
            
        return merged_boxes, None
        
class DetectionSystem:
    def __init__(self, 
                 method: str,
                 image_dir: str,
                 output_dir: str,
                 debug_dir: str):
        self.method = method
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.debug_dir = Path(debug_dir)
        self.evaluator = DetectionEvaluator()
        
        self.output_dir.mkdir(exist_ok=True)
        self.debug_dir.mkdir(exist_ok=True)
        
        for i in range(16):
            (self.debug_dir / f'NoEntry{i}').mkdir(exist_ok=True)
    
    @staticmethod
    def create_detector(method: str, detector_types: Optional[List[str]] = None):
        """
        Factory method to create different types of No-Entry sign detectors.
        Creates either individual detectors or combinations based on the method specified.
        """
         # Just Viola-Jones detector
        if method == 'vj':
            return ViolaJonesDetector(CASCADE_FILE)
        
        # Viola-Jones + circle detection
        elif method == 'vj_circle':
            detectors = [
                ViolaJonesDetector(CASCADE_FILE),
                CircleDetector()
            ]
            merger = DetectionMerger()
            return CombinedDetector(detectors=detectors, merger=merger)
        
        # Viola-Jones + circle + red colour detection
        elif method == 'vj_circle_red':
            detectors = [
                ViolaJonesDetector(CASCADE_FILE),
                CircleDetector(),
                RedDetector()
            ]
            merger = DetectionMerger()
            return CombinedDetector(detectors=detectors, merger=merger)

    def save_detection_visualization(self, 
                                  image: np.ndarray,
                                  detections: List[Tuple[int, int, int, int]],
                                  ground_truth: List[Tuple[int, int, int, int]],
                                  image_name: str):
        """
        Save an image showing both the detected and ground truth bounding boxes.
        Draws green boxes for detections and red boxes for ground truth
        """
        result_image = image.copy()
        
        # Draw the detections as green boxes
        for box in detections:
            cv2.rectangle(result_image, 
                         (box[0], box[1]), 
                         (box[0] + box[2], box[1] + box[3]),
                         (0, 255, 0), 2)
        
        # Draw the ground truth boxes in red
        for box in ground_truth:
            cv2.rectangle(result_image,
                         (box[0], box[1]),
                         (box[0] + box[2], box[1] + box[3]),
                         (0, 0, 255), 2)
        
        # Save the visualization to the output directory
        output_path = self.output_dir / image_name
        cv2.imwrite(str(output_path), result_image)
    
    def save_debug_data(self, debug_info: Dict, image_name: str):
        """
        Save the debug images to separate files for analysis.
        """
        # Get the base filename
        image_base = Path(image_name).stem
        # Create directory path for this image's debug images
        debug_dir = self.debug_dir / image_base
        
        # Loop through all debug items
        for key, data in debug_info.items():
            # Save any numpy arrays as images
            if isinstance(data, np.ndarray):
                cv2.imwrite(str(debug_dir / f"{key}.png"), data)
    
    def run_detection(self, 
                      detector: ViolaJonesDetector,
                      ground_truth_loader: GroundTruthLoader) -> List[DetectionResult]:
        """
        Processes each of the images, saves the debug info and visualizations, and 
        then computes the detection accuracy metrics.
        """
        results = []
        
        # Process each of the images in the ground truth dataset
        for image_name, ground_truth in ground_truth_loader.ground_truth_dict.items():
            image_path = self.image_dir / image_name

            # Skip if the image file doesn't exist
            if not image_path.exists():
                print(f"Warning: Image not found at {image_path}")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not read image {image_path}")
                continue
            
            # Run the detector and get its boxes and debug info
            detections, debug_info = detector.detect(image, return_debug=True)
            

            # Save the intermediate processing steps images for debugging
            self.save_debug_data(debug_info, image_name)
            
            # Save the image showing detections vs ground truth
            self.save_detection_visualization(image, detections, ground_truth, image_name)
            
            # Evaluate the detection accuracy against the ground truth
            result = self.evaluator.evaluate(detections, ground_truth)
            result.image_name = image_name
            results.append(result)
        
        return results
    
    @staticmethod
    def print_results(results: List[DetectionResult], method: str):
        print(f"\nResults for {method} method:")
        print("Image\t\tTPR\tF1\tDetections\tGround Truth")
        print("-" * 60)
        
        total_tpr = total_f1 = total_fp = total_fn = 0
        
        for result in results:
            print(f"{result.image_name:<15}\t{result.tpr:.3f}\t{result.f1:.3f}\t"
                  f"{result.num_detections}\t\t{result.num_ground_truth}")
            
            total_tpr += result.tpr
            total_f1 += result.f1
            total_fp += result.false_positives
            total_fn += result.false_negatives
        
        if results:
            num_images = len(results)
            print(f"\nSummary Statistics:")
            print(f"Average TPR: {total_tpr/num_images:.3f}")
            print(f"Average F1:  {total_f1/num_images:.3f}")
            print(f"Total False Positives: {total_fp}")
            print(f"Total False Negatives: {total_fn}")
            print(f"Images Processed: {num_images}")

class NoEntryDetector:
    """
    A single detector for "No Entry" signs using our most enhanced version that Combines 
    shape detection (Viola-Jones and circle) with red color detection.
    """
    def __init__(self):
        self.detector = DetectionSystem.create_detector('vj_circle_red')
    
    def detect(self, image_path: str) -> None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run the detection without getting any debug images
        detections = self.detector.detect(image, return_debug=False)[0]
        
        # Create the visualization of the detections
        result_image = image.copy()
        # Draws green boxes around each of the detected signs
        for box in detections:
            cv2.rectangle(result_image,
                         (box[0], box[1]),
                         (box[0] + box[2], box[1] + box[3]),
                         (0, 255, 0), 2)
        
         # Save the resulting image with the drawn boxes
        cv2.imwrite('detected.jpg', result_image)

def main():
    """
    Main entry point for No Entry Sign detection. Handles both the single image detection and 
    evaluation modes based on inputed command line arguments.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='No Entry Sign Detector')

    # Create mutually exclusive group - must choose either detection or evaluation
    group = parser.add_mutually_exclusive_group(required=True)

    # Option for running detection on a single image
    group.add_argument('--image', type=str, help='Path to input image for detection mode')

    # Option for running evaluation with different detection methods
    group.add_argument('--method', type=str, choices=['vj', 'vj_circle', 'vj_circle_red'],
                        help='Evaluation method')
    args = parser.parse_args()
    
    # Evaluation mode - testing of detectors against ground truth
    if args.method:
        # Load the ground truths 
        ground_truth_loader = GroundTruthLoader(ANNOTATIONS_FILE)
        
        # Create a detection system with appropriate directories
        detection_system = DetectionSystem(
            method=args.method,
            image_dir=IMAGE_DIR,             # Where to find input images
            output_dir=f"out_{args.method}", # Where to save detection visualizations
            debug_dir=f"debug_{args.method}" # Where to save debug information
        )
        
        # Create a detector of the specified type
        detector = detection_system.create_detector(args.method)
        
        # Run detection on all the images and evaluate results
        results = detection_system.run_detection(detector, ground_truth_loader)
        
        # Print the evaluation metrics
        detection_system.print_results(results, args.method)
    
    # Detection mode - processes a single image
    else:
        # Create our most enhanced detector and process a specified image
        detector = NoEntryDetector()
        detector.detect(args.image)

if __name__ == "__main__":
    main()