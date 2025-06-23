# VIIS: Visible and Infrared Information Synthesis for Severe Low-light Image Enhancement

This repository contains the official implementation of the paper  
**"VIIS: Visible and Infrared Information Synthesis for Severe Low-light Image Enhancement"**.

---

## 🚀 Getting Started

Follow the steps below to set up the environment and prepare all necessary resources.

### 1. Install the Environment
Set up the environment according to the instructions provided in the  
[CompVis latent-diffusion repository](https://github.com/CompVis/latent-diffusion).

```bash
git clone https://github.com/CompVis/latent-diffusion.git
cd latent-diffusion
conda env create -f environment.yaml
conda activate ldm
```

Return to the VIIS directory after the environment is set up.

---

### 2. Install Deformable Attention Module

Install the attention mechanism discrepancy from  
[Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR):

```bash
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR
python setup.py build install
```

---

### 3. Download Checkpoint

Download the pre-trained VIIS model checkpoint from [Google Drive](https://drive.google.com/file/d/1ur9uv_eUWYbvJVdKZbrsu50RVWhpX4x5/view?usp=sharing)  
and place it in the `./checkpoint/` directory.

---

### 4. Download VQ-F4 First-Stage Model

Download the VQ-F4 model from the [latent-diffusion releases](https://github.com/CompVis/latent-diffusion)  
and place it in the `./models/first_stage_models/` folder.

---

### 5. Download MSRS Dataset

Download the MSRS dataset from the [PIAFusion repository](https://github.com/Linfeng-Tang/PIAFusion)  
and place it in the `./data/` directory.
