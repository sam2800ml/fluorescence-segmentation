# Fluorescence Segmentation Pipeline 🧬

## Overview

This project implements a full end-to-end pipeline to download, preprocess, segment, and train a deep model on fluorescence microscopy images (nucleus, cytoplasm, protein) from the Human Protein Atlas (HPA). The goal is to produce high-quality segmentation masks and a trained model (UNet / Attention-UNet) that can be used for downstream analyses.

Key features:

- Automatic retrieval of HPA images given a list of ENSG IDs  
- Preprocessing and normalization of raw microscopy data  
- Classical image processing (thresholding + watershed) to build initial masks  
- Deep segmentation model training using PyTorch + MONAI  
- Experiment tracking and model logging via MLflow  
- Workflow orchestration using Apache Airflow + Docker for reproducibility and scalability  
- Configuration via a YAML file for paths and pipelines  

---
```bash
## 📁 Project Structure
├── donfig
├── dags/ # Airflow DAG definitions
│ ├── tasks/ 
│ │ ├── data_acquisition/     
│ │ ├── segmentation/
│ │ ├── train/
│ ├── utils/ 
│ ├── test.py
├── notebooks
├── ends_id.txt # List of ENSG IDs to process
├── Dockerfile
├── Dockerfile.mlflow
├── Dockerfile.streamlit
├── docker-compose.yaml
├── pyproject.toml
├── README.md 
```


---

## 🧰 Dependencies & Tech Stack

| Component                | Purpose |
|--------------------------|---------|
| **Apache Airflow**       | Orchestrates the entire workflow — from data acquisition to model training |
| **Docker & Docker Compose** | Containerized, reproducible environment for everyone to run the pipeline |
| **Python 3.11**          | Main programming language |
| **PyTorch + MONAI**      | Deep learning framework for segmentation, model training and loss functions |
| **Albumentations / OpenCV / scikit-image** | Image preprocessing, augmentations and manipulation |
| **pandas / yaml / CSV**  | Dataset and configuration file management |
| **MLflow**               | Tracks experiments, logs parameters, metrics, and artifact / model versions |
| **BeautifulSoup / requests** | Fetches and parses HPA metadata (XML) to get image URLs |

---

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose installed  
- ~8–16 GB RAM or access to a GPU (recommended for segmentation tasks)  
- A Unix-like OS (Linux / macOS / WSL) — though Docker works on Windows too  

### Quick Start — Run the full pipeline

```bash
git clone https://github.com/<your-username>/fluorescence-segmentation-pipeline.git
cd fluorescence-segmentation-pipeline

# Build Docker image and start containers
docker compose build
docker compose up -d

```
Once the containers are running:

Go to Airflow UI at http://localhost:8080

Trigger the DAG called fluorescence-segmentation

The pipeline will:

Load config from YAML

Download HPA images for the ENSG IDs in ends_id.txt

Generate masks (nucleus, cytoplasm, protein)

Build dataset and train segmentation model

Log metrics and save model via MLflow

⏱️ Current Limitations & Future Work
Running training inside Airflow + Docker may still trigger memory issues depending on dataset size and model — consider using a more powerful environment or GPU and multicontainers.

Augmentations and preprocessing are basic — more advanced data cleaning / augmentation pipelines may improve performance.

Mask generation currently uses classical image-processing; future work could integrate human-annotated masks or semi-supervised refinement.
