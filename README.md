# 🌍 GeoSpatial NLI  
**A Natural Language Interface for Multimodal Satellite Image Understanding**

> *“Is a picture really worth a thousand words?”*  
---

## 📌 Overview

**GeoSpatial NLI** is an end-to-end vision–language system that enables **non-expert users** to analyze satellite imagery using **natural language queries**.

Given a single satellite image, the system can:
- 📝 **Generate detailed captions**
- ❓ **Answer natural language questions (VQA)**
- 📍 **Localize objects via oriented bounding boxes (OBB grounding)**

The pipeline is designed to work across **RGB, SAR, IR, and False Color Composite (FCC)** imagery and supports **high-resolution inputs up to 2k×2k**, operating robustly across **0.5–10 m/pixel** spatial scales.

---

## 🧠 Key Contributions

- Unified **natural language interface** for satellite imagery  
- Multi-modal handling of **RGB, SAR, IR, and FCC** images  
- **Scale-robust inference** across diverse spatial resolutions  
- **Oriented object grounding** suitable for overhead viewpoints  
- **SAR grounding without SAR captions**, using detector + LLM reasoning  
- Fully deployable **web-based system**

---

## Acknowledgements

We thank the authors of SARATR-X, VRSBench, Qwen-VL, Moondream, and SAM
for open-sourcing their work, which made this project possible.
