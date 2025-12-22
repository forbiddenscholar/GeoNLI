#!/usr/bin/env python3
"""
Spatial Grounding with Small LLM

Handles:
- "find ships on the left"
- "identify the bigger car"
- "2nd smallest vessel"
- "largest ship near the bottom"
- "tanks closest to the center"
"""

import torch
import json
import re
from dataclasses import dataclass, field
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BBox:
    class_name: str
    score: float
    x1: float  # normalized 0-1
    y1: float
    x2: float
    y2: float
    
    # Computed properties
    idx: int = 0
    
    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def region(self) -> str:
        h = "left" if self.cx < 0.33 else ("right" if self.cx > 0.66 else "center")
        v = "top" if self.cy < 0.33 else ("bottom" if self.cy > 0.66 else "middle")
        return f"{v}-{h}"
    
    def distance_to_center(self) -> float:
        return ((self.cx - 0.5) ** 2 + (self.cy - 0.5) ** 2) ** 0.5
    
    def to_info(self) -> dict:
        return {
            "id": self.idx,
            "class": self.class_name,
            "center_x": round(self.cx, 3),
            "center_y": round(self.cy, 3),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "area": round(self.area, 6),
            "region": self.region,
            "distance_to_center": round(self.distance_to_center(), 3)
        }


def parse_detections(det_text: str) -> List[BBox]:
    """Parse detection file."""
    bboxes = []
    idx = 0
    for line in det_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 6:
            bbox = BBox(
                class_name=parts[0],
                score=float(parts[1]),
                x1=float(parts[2]),
                y1=float(parts[3]),
                x2=float(parts[4]),
                y2=float(parts[5])
            )
            bbox.idx = idx
            bboxes.append(bbox)
            idx += 1
    return bboxes


def compute_rankings(bboxes: List[BBox]) -> List[dict]:
    """
    Compute size rankings and other properties for LLM.
    Groups by class and ranks within each class.
    """
    # Group by class
    by_class = {}
    for b in bboxes:
        if b.class_name not in by_class:
            by_class[b.class_name] = []
        by_class[b.class_name].append(b)
    
    # Compute rankings within each class
    info_list = []
    
    for cls_name, class_bboxes in by_class.items():
        # Sort by area (largest first)
        sorted_by_size = sorted(class_bboxes, key=lambda x: x.area, reverse=True)
        
        # Sort by position (left to right)
        sorted_left_right = sorted(class_bboxes, key=lambda x: x.cx)
        
        # Sort by position (top to bottom)
        sorted_top_bottom = sorted(class_bboxes, key=lambda x: x.cy)
        
        for b in class_bboxes:
            info = b.to_info()
            
            # Size ranking (1 = largest)
            info["size_rank"] = sorted_by_size.index(b) + 1
            info["size_rank_desc"] = f"{info['size_rank']} of {len(sorted_by_size)}"
            
            if len(sorted_by_size) > 1:
                if info["size_rank"] == 1:
                    info["size_label"] = "largest"
                elif info["size_rank"] == len(sorted_by_size):
                    info["size_label"] = "smallest"
                else:
                    info["size_label"] = f"{info['size_rank']}th largest"
            else:
                info["size_label"] = "only one"
            
            # Position ranking (left-right)
            