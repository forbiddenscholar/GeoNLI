import os
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import requests
from io import BytesIO

def load_image(image_path_or_url):
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        # URL handling
        try:
            response = requests.get(image_path_or_url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image from URL: {e}")

def load_model():
    MODEL_DIR = "qwen3vl_final_ft_cap"    
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  

    print("[LOAD] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[LOAD] Loading fine-tuned Qwen3-VL VQA model...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()
    
    return processor, model

def answer_vqa(image_path, question, processor, model):
    image = load_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question + " Answer in one or two words."}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]

    answer = processor.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return answer



if __name__ == "__main__":
    
    image = load_image("https://img.sanishtech.com/u/e7c7ec16c171bd4066e6ff6dd4dc04c5.png")
    processor, model = load_model()
    query = ""
    ans = answer_vqa(image, query, processor, model)
    print(ans)