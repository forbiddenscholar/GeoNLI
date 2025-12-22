import os
import cv2
import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM
import re

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAM_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

MD_MODEL_ID = "vikhyatk/moondream2"


# ---------------- IMAGE LOADING ---------------- #

def load_image_from_url(url):
    r = requests.get(url)
    r.raise_for_status()
    img_bytes = r.content

    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    return pil_img, cv_img


def save_original_image(pil_img, save_path="original_rgb.jpg"):
    img = np.array(pil_img)
    cv2.imwrite(save_path, img)


# ---------------- MOONDREAM ---------------- #

def run_moondream(image_url, text_prompt):
    pil_img, _ = load_image_from_url(image_url)
    W, H = pil_img.size

    md_dtype = torch.float32 if DEVICE == "cpu" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        MD_MODEL_ID,
        trust_remote_code=True,
        dtype=md_dtype,
        device_map={"": DEVICE},
    ).to(DEVICE)
    model.eval()

    result = model.detect(pil_img, text_prompt)

    if not isinstance(result, dict) or "objects" not in result:
        return pil_img, []

    boxes = []
    for obj in result["objects"]:
        try:
            x_min = float(obj.get("x_min", obj.get("xmin", obj.get("left"))))
            y_min = float(obj.get("y_min", obj.get("ymin", obj.get("top"))))
            x_max = float(obj.get("x_max", obj.get("xmax", obj.get("right"))))
            y_max = float(obj.get("y_max", obj.get("ymax", obj.get("bottom"))))
        except:
            continue

        x1 = max(0, min(x_min * W, W - 1))
        y1 = max(0, min(y_min * H, H - 1))
        x2 = max(0, min(x_max * W, W - 1))
        y2 = max(0, min(y_max * H, H - 1))

        if x2 > x1 + 5 and y2 > y1 + 5:
            boxes.append([x1, y1, x2, y2])

    return pil_img, boxes


def save_moondream_boxes(pil_img, boxes, save_path="moondream_hbb.jpg"):
    img = np.array(pil_img)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(save_path, img)


# ---------------- SAM 2.1 ---------------- #

def run_sam2(pil_img, boxes):
    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError("Missing SAM checkpoint")

    sam_model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam_model)

    predictor.set_image(np.array(pil_img))

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(boxes, dtype=np.float32),
        multimask_output=False
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return masks


def save_masked_image(pil_img, masks, save_path="sam_mask.jpg"):
    img = np.array(pil_img)
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for mask in masks:
        combined_mask = np.maximum(combined_mask, (mask * 255).astype(np.uint8))

    cv2.imwrite("binary_mask.png", combined_mask)

    color_mask = np.zeros_like(img)
    color_mask[:, :, 1] = combined_mask

    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    cv2.imwrite(save_path, overlay)


# ---------------- MIN AREA RECT ---------------- #

def get_rotated_rectangles(image_url, masks, save_path="minarearect_obb.jpg"):
    _, img = load_image_from_url(image_url)
    rotated_rects = []

    for mask in masks:
        m = (mask * 255).astype("uint8")

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 10:
            continue

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        rotated_rects.append(box.flatten().tolist())
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    cv2.imwrite(save_path, img)
    return rotated_rects


# ---------------- QUERY REFINEMENT ---------------- #

def refine_query(user_query: str) -> str:
    try:
        q = user_query.strip().lower()

        q = re.sub(
            r'locate\s+and\s+return\s+(?:the\s+)?(?:oriented\s+)?bounding\s+boxes(?:\s+of)?',
            '',
            q
        )

        q = re.sub(r'\bin\s+the\s+image\b', '', q)
        q = re.sub(r'\bfor\b', ' ', q)
        q = re.sub(r'\bof\b', ' ', q)

        q = re.sub(r'\s+', ' ', q).strip(" .,\n\t")
        return q
    except:
        return user_query.strip().lower()


# ---------------- NORMALIZATION ---------------- #

def normalize_obbs(obbs, img):
    W, H = img.size
    norm = []

    for box in obbs:
        norm_box = []
        for i in range(0, 8, 2):
            x = box[i] / W
            y = box[i+1] / H
            norm_box.extend([x, y])
        norm.append(norm_box)

    return norm


# ---------------- MAIN PIPELINE ---------------- #

def detect_objects(image_url, text_prompt):
    final_query = refine_query(text_prompt)

    pil_img, boxes = run_moondream(image_url, final_query)
    if len(boxes) == 0:
        return []

    save_original_image(pil_img, "original_rgb.jpg")
    save_moondream_boxes(pil_img, boxes, "moondream_hbb.jpg")

    masks = run_sam2(pil_img, boxes)
    save_masked_image(pil_img, masks, "sam_mask.jpg")

    rotated_rects = get_rotated_rectangles(image_url, masks, "minarearect_obb.jpg")
    final_rotated_rects = normalize_obbs(rotated_rects, pil_img)

    return final_rotated_rects


# # ---------------- RUN ---------------- #

# if __name__ == "__main__":
#     image_url = "PUT_YOUR_IMAGE_URL_HERE"
#     query = "locate and return oriented bounding boxes for all the airplanes in the image"

#     result = detect_objects(image_url, query)
#     print("Final normalized OBBs:", result)
