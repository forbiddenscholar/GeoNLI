# User Guide

## 1. Introduction
The application provides a complete pipeline for natural-language interpretation of satellite imagery, supporting captioning, grounding, and visual question answering.  
This guide explains how to set up the deployment package and operate the interface to run inference.  
The setup is lightweight and requires only minimal environment preparation using the files provided.

---

## 2. System Requirements
No special system requirements are needed beyond a working Python installation (Python 3.x).

---

## 3. Installation

### 3.1 Unzip the Deployment Package
Extract the provided ZIP file.  
It contains the full project folder along with all required checkpoints.

### 3.2 Create a Virtual Environment
Run the following commands:
```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3.3 Install Dependencies    

Install all required Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3.4 Download SAM 2.1 Source Code

Download the official SAM 2.1 source repository:
```bash
wget https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip -O sam2_source.zip
```

### 3.5 Unzip SAM 2.1

Extract the downloaded SAM 2.1 source archive:
```bash
unzip sam2_source.zip
```

### 3.6 Install SAM 2.1 in Editable Mode

Navigate to the extracted directory and install SAM 2.1:
```bash
cd sam2-main
pip install -e .
```

### 3.7 Return to the Main Project Directory
```bash
cd ..
```

## 4. Running the Application

### 4.1 Start the Flask Server

Launch the application server in debug mode:
```bash
flask --app VLMHosting run --debug
```
