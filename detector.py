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
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def evaluate(self, detections: List[Tuple[int, int, int, int]], 
                ground_truth: List[Tuple[int, int, int, int]]) -> DetectionResult:
        true_positives = 0
        for gt_box in ground_truth:
            max_iou = max((self.calculate_iou(det_box, gt_box) 
                          for det_box in detections), default=0)
            if max_iou >= self.iou_threshold:
                true_positives += 1
        
        if len(ground_truth) == 0:
            tpr = 1.0 if len(detections) == 0 else 0.0
        else:
            tpr = true_positives / len(ground_truth)
        
        precision = true_positives / len(detections) if len(detections) > 0 else 0
        recall = tpr
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
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
        self.ground_truth_dict = self._load_annotations()
    
    def _load_annotations(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        ground_truth_dict = {}
        
        try:
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                image_name = Path(parts[0]).name
                num_boxes = int(parts[1])
                boxes = []
                
                for i in range(num_boxes):
                    idx = 2 + (i * 4)
                    if idx + 3 >= len(parts):
                        break
                    
                    try:
                        x, y, w, h = map(int, parts[idx:idx+4])
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
        gray = self._preprocess_image(image)
        
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.config['SCALE_FACTOR'],
            minNeighbors=self.config['MIN_NEIGHBORS'],
            minSize=self.config['MIN_SIZE'],
            maxSize=self.config['MAX_SIZE']
        )
        
        boxes = [tuple(d) for d in detections] if len(detections) > 0 else []
        
        return boxes, {}
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = cv2.equalizeHist(gray)
        return gray
    
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
        min_radius = self.config["MIN_RADIUS"]  
        max_radius = self.config["MAX_RADIUS"]

        grad_x, grad_y, grad_magnitude, grad_orientation = self._sobel(image)
        threshold_percentage = self.config['GRADIENT_THRESHOLD_PERCENTAGE']
        edges = self._threshold_gradient(grad_magnitude, threshold_percentage)
        
        accumulator = self._compute_circular_hough_accumulator(edges, grad_orientation, min_radius, max_radius)
        
        hough_space_2d = np.sum(accumulator, axis=2)
        
        threshold_ratio = self.config["HOUGH_THRESHOLD_RATIO"]    
        threshold_value = np.max(hough_space_2d) * threshold_ratio
        thresholded_hough = np.where(hough_space_2d > threshold_value, 255, 0).astype(np.uint8)
        
        circles = self._find_circles(accumulator, min_radius, threshold_ratio, min_distance=20)
        
        boxes = []
        for x, y, r in circles:
            box_x = max(0, x - r)
            box_y = max(0, y - r)
            box_w = min(2*r, image.shape[1] - box_x)
            box_h = min(2*r, image.shape[0] - box_y)
            boxes.append((box_x, box_y, box_w, box_h))

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
        circle_viz = image.copy()
        box_viz = image.copy()
        
        for x, y, r in circles:
            cv2.circle(circle_viz, (x, y), r, (0, 255, 0), 2)
        
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

    def _sobel(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = cv2.GaussianBlur(image, (7, 7), 1.5)
        
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        grad_x = self._fast_convolve2d(image, kernel_x)
        grad_y = self._fast_convolve2d(image, kernel_y)
        
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        return grad_x, grad_y, grad_magnitude, grad_direction

    def _compute_circular_hough_accumulator(self, edge_image: np.ndarray, grad_orientation: np.ndarray,
                        min_radius: int, max_radius: int) -> np.ndarray:
        height, width = edge_image.shape
        num_radii = max_radius - min_radius + 1
        accumulator = np.zeros((height, width, num_radii), dtype=np.uint32)
        
        y_coords, x_coords = np.nonzero(edge_image)
        radii = np.arange(min_radius, max_radius + 1)
        
        for start_idx in range(0, len(y_coords), self.config['BATCH_SIZE']):
            end_idx = min(start_idx + self.config['BATCH_SIZE'], len(y_coords))
            
            y_batch = y_coords[start_idx:end_idx]
            x_batch = x_coords[start_idx:end_idx]
            angles = grad_orientation[y_batch, x_batch]
            
            for r_idx, radius in enumerate(radii):
                x_centers = (x_batch + radius * np.cos(angles)).astype(int)
                y_centers = (y_batch + radius * np.sin(angles)).astype(int)
                
                valid_points = (x_centers >= 0) & (x_centers < width) & \
                             (y_centers >= 0) & (y_centers < height)
                
                x_valid = x_centers[valid_points]
                y_valid = y_centers[valid_points]
                
                np.add.at(accumulator[:, :, r_idx], (y_valid, x_valid), 1)
        
        return accumulator

    def _find_circles(self, accumulator: np.ndarray, min_radius: int, 
                     threshold_ratio: float, min_distance: int = 30) -> List[Tuple[int, int, int]]:
        height, width, _ = accumulator.shape
        acc_2d = np.sum(accumulator, axis=2)
        
        threshold = np.max(acc_2d) * threshold_ratio
        peaks = acc_2d > threshold
        
        circles = []
        for y in range(height):
            for x in range(width):
                if peaks[y, x]:
                    window = acc_2d[max(0, y-min_distance//2):min(height, y+min_distance//2+1),
                                  max(0, x-min_distance//2):min(width, x+min_distance//2+1)]
                    if acc_2d[y, x] == np.max(window):
                        r_idx = np.argmax(accumulator[y, x, :])
                        radius = r_idx + min_radius
                        strength = acc_2d[y, x]
                        circles.append((x, y, radius, strength))
        
        circles.sort(key=lambda x: x[3], reverse=True)
        circles = circles[:self.config['MAX_DETECTIONS']]
        
        return [(x, y, r) for x, y, r, _ in circles]

    def _threshold_gradient(self, grad_magnitude: np.ndarray, threshold_percentage: float) -> np.ndarray:
        threshold = np.percentile(grad_magnitude, threshold_percentage)
        return np.where(grad_magnitude > threshold, 255, 0).astype(np.uint8)
    
    def _fast_convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kernel = np.flipud(np.fliplr(kernel))
        padding = kernel.shape[0] // 2
        padded = np.pad(image, padding, mode='constant')
        
        shape = (image.shape[0], image.shape[1], kernel.shape[0], kernel.shape[1])
        strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
        
        windows = as_strided(padded, shape=shape, strides=strides)
        
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
        combined_mask = self._get_red_mask(image)
        combined_mask = self._apply_morphology(combined_mask)
        boxes = self._find_boxes(combined_mask, image.shape)
        
        if return_debug:
            debug_info = self._create_debug_visualization(image, boxes, combined_mask)
            return boxes, debug_info
        
        return boxes, None

    def _get_red_mask(self, image: np.ndarray) -> np.ndarray:
        hsv_mask = self._get_hsv_mask(image)
        lab_mask = self._get_lab_mask(image)
        ratio_mask = self._get_ratio_mask(image)
        
        combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
        combined_mask = cv2.bitwise_and(combined_mask, ratio_mask)
        return combined_mask
    
    def _get_hsv_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask_hsv1 = cv2.inRange(hsv, self.thresholds['hsv_lower1'], 
                                   self.thresholds['hsv_upper1'])
        red_mask_hsv2 = cv2.inRange(hsv, self.thresholds['hsv_lower2'], 
                                   self.thresholds['hsv_upper2'])
        return cv2.bitwise_or(red_mask_hsv1, red_mask_hsv2)

    def _get_lab_mask(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a_channel, _ = cv2.split(lab)
        _, red_mask_lab = cv2.threshold(a_channel, self.thresholds['lab_threshold'], 255, 
                                      cv2.THRESH_BINARY)
        return red_mask_lab

    def _get_ratio_mask(self, image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image)
        epsilon = 1e-6
        red_ratio = r.astype(float) / (b.astype(float) + g.astype(float) + epsilon)
        return np.where(red_ratio > self.thresholds['red_ratio'], 255, 0).astype(np.uint8)

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _find_boxes(self, mask: np.ndarray, image_shape: Tuple[int, ...]) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.thresholds['min_area']:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = (w * h) / (image_shape[0] * image_shape[1])
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
        
        for box_x, box_y, box_w, box_h in boxes:
            cv2.rectangle(detection_viz, (box_x, box_y), 
                        (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
            cv2.rectangle(box_viz, (box_x, box_y), 
                        (box_x + box_w, box_y + box_h), (0, 0, 255), 2)
            
            # Add semi-transparent overlay
            roi = overlay_viz[box_y:box_y+box_h, box_x:box_x+box_w]
            overlay = np.zeros_like(roi)
            overlay[:, :] = [0, 0, 255]
            cv2.addWeighted(overlay, 0.3, roi, 0.7, 0, roi)
        
        return {
            'red_mask': cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR),
            'detections': detection_viz,
            'boxes': box_viz,
            'overlay': overlay_viz
        }

class DetectionMerger:
    def __init__(self, distance_threshold: float = 0.5, iou_threshold: float = 0.4):
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def _calculate_box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return (box[0] + box[2]//2, box[1] + box[3]//2)
    
    @staticmethod
    def _calculate_center_distance(center1: Tuple[float, float], 
                                 center2: Tuple[float, float]) -> float:
        return ((center1[0] - center2[0])**2 + 
                (center1[1] - center2[1])**2)**0.5
    
    def _merge_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        x = int(np.mean([box[0] for box in boxes]))
        y = int(np.mean([box[1] for box in boxes]))
        w = int(np.mean([box[2] for box in boxes]))
        h = int(np.mean([box[3] for box in boxes]))
        return (x, y, w, h)
    
    def merge_detections(self, detection_lists: List[List[Tuple[int, int, int, int]]]) -> List[Tuple[int, int, int, int]]:
        if not detection_lists or not any(detection_lists):
            return []
        
        final_detections = []
        used_detections = [set() for _ in detection_lists]
        
        for primary_idx, primary_box in enumerate(detection_lists[0]):
            primary_center = self._calculate_box_center(primary_box)
            
            best_matches = []
            best_ious = []
            best_indices = []
            
            for list_idx in range(1, len(detection_lists)):
                best_match = None
                best_iou = 0
                best_idx = None
                
                for box_idx, box in enumerate(detection_lists[list_idx]):
                    if box_idx in used_detections[list_idx]:
                        continue
                    
                    box_center = self._calculate_box_center(box)
                    dist = self._calculate_center_distance(primary_center, box_center)
                    iou = DetectionEvaluator.calculate_iou(primary_box, box)
                    
                    if (dist < max(primary_box[2], box[2]) * self.distance_threshold and 
                        iou > best_iou):
                        best_match = box
                        best_iou = iou
                        best_idx = box_idx
                
                best_matches.append(best_match)
                best_ious.append(best_iou)
                best_indices.append(best_idx)
            
            valid_matches = [i for i, iou in enumerate(best_ious) 
                           if iou > self.iou_threshold]
            
            if valid_matches:
                for list_idx, box_idx in enumerate(best_indices):
                    if box_idx is not None and list_idx in valid_matches:
                        used_detections[list_idx + 1].add(box_idx)
                
                boxes_to_merge = [primary_box] + [best_matches[i] 
                                                for i in valid_matches]
                merged_box = self._merge_boxes(boxes_to_merge)
                final_detections.append(merged_box)
        
        return final_detections
    
class CombinedDetector:
    def __init__(self, detectors: List[Any], merger: DetectionMerger):
        self.detectors = detectors
        self.merger = merger
    
    def detect(self, image: np.ndarray, return_debug: bool = False):
        all_boxes = []
        all_debug = {}
        
        for detector in self.detectors:
            boxes, debug = detector.detect(image, return_debug=True)
            all_boxes.append(boxes)
            if debug:
                all_debug.update(debug)
        
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
        if method == 'vj':
            return ViolaJonesDetector(CASCADE_FILE)
        elif method == 'vj_circle':
            detectors = [
                ViolaJonesDetector(CASCADE_FILE),
                CircleDetector()
            ]
            merger = DetectionMerger()
            return CombinedDetector(detectors=detectors, merger=merger)
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
        result_image = image.copy()
        
        for box in detections:
            cv2.rectangle(result_image, 
                         (box[0], box[1]), 
                         (box[0] + box[2], box[1] + box[3]),
                         (0, 255, 0), 2)
        
        for box in ground_truth:
            cv2.rectangle(result_image,
                         (box[0], box[1]),
                         (box[0] + box[2], box[1] + box[3]),
                         (0, 0, 255), 2)
        
        output_path = self.output_dir / image_name
        cv2.imwrite(str(output_path), result_image)
    
    def save_debug_data(self, debug_info: Dict, image_name: str):
        image_base = Path(image_name).stem
        debug_dir = self.debug_dir / image_base
        
        for key, data in debug_info.items():
            if isinstance(data, np.ndarray):
                cv2.imwrite(str(debug_dir / f"{key}.png"), data)
    
    def run_detection(self, 
                      detector: ViolaJonesDetector,
                      ground_truth_loader: GroundTruthLoader) -> List[DetectionResult]:
        results = []
        
        for image_name, ground_truth in ground_truth_loader.ground_truth_dict.items():
            image_path = self.image_dir / image_name
            if not image_path.exists():
                print(f"Warning: Image not found at {image_path}")
                continue
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not read image {image_path}")
                continue
            
            detections, debug_info = detector.detect(image, return_debug=True)
            
            self.save_debug_data(debug_info, image_name)
            
            self.save_detection_visualization(image, detections, ground_truth, image_name)
            
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
    def __init__(self):
        self.detector = DetectionSystem.create_detector('vj_circle_red')  # Use most sophisticated method
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def detect(self, image_path: str) -> None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        detections = self.detector.detect(image, return_debug=False)[0]  # Get first element of tuple
        
        result_image = image.copy()
        for box in detections:
            cv2.rectangle(result_image,
                         (box[0], box[1]),
                         (box[0] + box[2], box[1] + box[3]),
                         (0, 255, 0), 2)
        
        cv2.imwrite('detected.jpg', result_image)

def main():
    parser = argparse.ArgumentParser(description='No Entry Sign Detector')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to input image for detection mode')
    group.add_argument('--method', type=str, choices=['vj', 'vj_circle', 'vj_circle_red'],
                      help='Evaluation method')
    args = parser.parse_args()
    
    if args.method:
        ground_truth_loader = GroundTruthLoader(ANNOTATIONS_FILE)
        detection_system = DetectionSystem(
            method=args.method,
            image_dir=IMAGE_DIR,
            output_dir=f"out_{args.method}",
            debug_dir=f"debug_{args.method}"
        )
        detector = detection_system.create_detector(args.method)
        results = detection_system.run_detection(detector, ground_truth_loader)
        detection_system.print_results(results, args.method)
    else:
        detector = NoEntryDetector()
        detector.detect(args.image)

if __name__ == "__main__":
    main()