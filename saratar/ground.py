import os
import cv2
import numpy as np
import argparse
import torch
import yaml
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

try:
    from saratrx_hivit import hivit  # noqa: F401
except KeyError:
    pass


def ground(image_path: str, query: str, config_path: str, output_path: str = None):
    from mmdet.apis import init_detector, inference_detector

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = config.get("paths", {}).get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # =============================================
    # STEP 1: DETECT
    # =============================================
    print("=" * 50)
    print("STEP 1: Detection")
    print("=" * 50)
    
    print("Loading detector...")
    detector = init_detector(
        config["detector"]["config"],
        config["detector"]["checkpoint"],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    class_names = list(detector.CLASSES)
    threshold = config["detector"].get("threshold", 0.3)

    print(f"Available classes: {class_names}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    img_h, img_w = image.shape[:2]
    print(f"Image size: {img_w} x {img_h}")

    print("Running detection...")
    result = inference_detector(detector, image_path)

    detections = []
    for cls_idx, bboxes in enumerate(result):
        for bbox in bboxes:
            if bbox[4] >= threshold:
                x1_norm = bbox[0] / img_w
                y1_norm = bbox[1] / img_h
                x2_norm = bbox[2] / img_w
                y2_norm = bbox[3] / img_h
                
                cx = (x1_norm + x2_norm) / 2
                cy = (y1_norm + y2_norm) / 2
                area = (x2_norm - x1_norm) * (y2_norm - y1_norm)

                detections.append({
                    "idx": len(detections),
                    "class": class_names[cls_idx],
                    "score": float(bbox[4]),
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "bbox_pixel": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "bbox_norm": [x1_norm, y1_norm, x2_norm, y2_norm]
                })

    print(f"Found {len(detections)} objects")
    
    for d in detections:
        print(f"  [{d['idx']}] {d['class']}: pos=({d['cx']:.2f},{d['cy']:.2f}), size={d['area']:.4f}")

    if not detections:
        print("No objects detected.")
        if output_path is None:
            name = os.path.basename(image_path).rsplit(".", 1)[0]
            output_path = os.path.join(output_dir, f"{name}_no_detection.jpg")
        cv2.imwrite(output_path, image)
        return

    del detector
    torch.cuda.empty_cache()

    # =============================================
    # STEP 2: LLM SELECTION
    # =============================================
    print("")
    print("=" * 50)
    print("STEP 2: LLM Selection")
    print("=" * 50)

    model_path = config.get("llm", {}).get("model", "./models/flan-t5-large")
    
    print(f"Loading LLM: {model_path}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Query: {query}")

    # Parse query to understand what user wants
    selected_ids = select_objects(query, detections, model, tokenizer, device)

    print(f"Selected IDs: {list(selected_ids)}")

    del model, tokenizer
    torch.cuda.empty_cache()

    # =============================================
    # STEP 3: DRAW BOXES
    # =============================================
    print("")
    print("=" * 50)
    print("STEP 3: Drawing")
    print("=" * 50)

    for d in detections:
        if d["idx"] in selected_ids:
            x1, y1, x2, y2 = d["bbox_pixel"]

            np.random.seed(hash(d["class"]) % (2**32))
            color = tuple(map(int, np.random.randint(50, 255, 3)))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{d['class']}: {d['score']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if output_path is None:
        name = os.path.basename(image_path).rsplit(".", 1)[0]
        query_slug = re.sub(r'[^\w]', '_', query.lower())[:20]
        output_path = os.path.join(output_dir, f"{name}_{query_slug}.jpg")

    cv2.imwrite(output_path, image)
    print(f"Saved image: {output_path}")

    txt_path = output_path.rsplit(".", 1)[0] + ".txt"
    with open(txt_path, 'w') as f:
        f.write(f"# query: {query}\n")
        f.write(f"# image_size: {img_w} {img_h}\n")
        for d in detections:
            if d["idx"] in selected_ids:
                x1, y1, x2, y2 = d["bbox_norm"]
                f.write(f"{d['class']} {d['score']:.4f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")
    
    print(f"Saved boxes: {txt_path}")

    print("")
    print("=" * 50)
    print(f"Done! Selected {len(selected_ids)}/{len(detections)} objects")
    print("=" * 50)


def select_objects(query: str, detections: list, model, tokenizer, device: str) -> set:
    
    query_lower = query.lower()
    selected = set()
    
    # ===== STEP A: Determine target class =====
    class_prompt = f"""What object type is this query asking for?

Query: "{query}"

Options: ship, aircraft, car, tank, bridge, harbor, all

Note: boat/vessel means ship, plane means aircraft, vehicle means car, similarly understand the synonyms for the given options

Answer with one word:"""

    inputs = tokenizer(class_prompt, return_tensors="pt", max_length=256, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    target_class = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    print(f"  Target class: {target_class}")
    
    # Map synonyms
    class_map = {
        "boat": "ship", "vessel": "ship", "boats": "ship", "vessels": "ship", "ships": "ship",
        "plane": "aircraft", "airplane": "aircraft", "planes": "aircraft", "jet": "aircraft",
        "vehicle": "car", "vehicles": "car", "cars": "car", "truck": "car",
        "tanks": "tank", "bridges": "bridge", "harbors": "harbor"
    }
    target_class = class_map.get(target_class, target_class)
    
    # Filter by class
    if target_class == "all":
        class_filtered = detections
    else:
        class_filtered = [d for d in detections if d["class"] == target_class]
    
    if not class_filtered:
        print(f"  No {target_class} found in detections")
        return selected
    
    print(f"  Found {len(class_filtered)} {target_class}(s)")
    
    # ===== STEP B: Check for size filter =====
    size_keywords = ["largest", "biggest", "largest", "smallest", "bigger", "smaller", "big", "small"]
    has_size = any(kw in query_lower for kw in size_keywords)
    
    if has_size:
        size_prompt = f"""Does this query want the largest or smallest object?

Query: "{query}"

Answer "largest", "smallest", or "none":"""

        inputs = tokenizer(size_prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        size_filter = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        print(f"  Size filter: {size_filter}")
        
        if "largest" in size_filter or "biggest" in size_filter:
            sorted_by_size = sorted(class_filtered, key=lambda x: x["area"], reverse=True)
            selected.add(sorted_by_size[0]["idx"])
            return selected
        elif "smallest" in size_filter:
            sorted_by_size = sorted(class_filtered, key=lambda x: x["area"])
            selected.add(sorted_by_size[0]["idx"])
            return selected
    
    # ===== STEP C: Check for position filter =====
    pos_keywords = ["left", "right", "top", "bottom", "center", "middle"]
    has_position = any(kw in query_lower for kw in pos_keywords)
    
    if has_position:
        pos_prompt = f"""What position does this query specify?

Query: "{query}"

Answer one of: left, right, top, bottom, center, or none:"""

        inputs = tokenizer(pos_prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        position = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        print(f"  Position filter: {position}")
        
        for d in class_filtered:
            include = False
            if "left" in position and d["cx"] < 0.4:
                include = True
            elif "right" in position and d["cx"] > 0.6:
                include = True
            elif "top" in position and d["cy"] < 0.4:
                include = True
            elif "bottom" in position and d["cy"] > 0.6:
                include = True
            elif "center" in position or "middle" in position:
                if 0.3 < d["cx"] < 0.7 and 0.3 < d["cy"] < 0.7:
                    include = True
            
            if include:
                selected.add(d["idx"])
        
        if selected:
            return selected
    
    # ===== STEP D: No specific filter - return all of target class =====
    for d in class_filtered:
        selected.add(d["idx"])
    
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()

    ground(args.image, args.query, args.config, args.output)