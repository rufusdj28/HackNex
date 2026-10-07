<div align="center">

# ◈ ReScene

### **Situation-Aware AI Scene Reconstruction**

**Reconstruct what is missing. Preserve what isn't.**

<br>

[![HackNex 2026](https://img.shields.io/badge/HackNex%202026-PS06-111827?style=for-the-badge)](https://github.com/rufusdj28/HackNex)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-FF7C00?style=for-the-badge)](https://www.gradio.app/)

<br>

> **ReScene doesn't blindly inpaint. It analyzes the situation first.**

</div>

---

## ✦ Overview

**ReScene** is a situation-aware generative computer vision system designed
to reconstruct damaged, missing, or unwanted regions of an image while
preserving the original scene outside the requested reconstruction area.

Instead of applying the same inpainting strategy to every image, ReScene
first analyzes the **damage geometry and surrounding visual context**.

It extracts measurable characteristics such as:

- Damage geometry
- Edge density
- Line structure
- Dominant orientation
- Texture characteristics
- Local complexity
- Surrounding context

These features are converted into a **situation profile**, which is then
used by the reconstruction planner to select an appropriate reconstruction
strategy.

### Core Principle

> **Understand the situation → choose the strategy → reconstruct only what is missing → verify what was preserved.**

---

# ✦ The Problem

Image restoration and object removal become challenging when the missing
region contains:

- Architectural structures
- Straight lines and perspective geometry
- Repetitive textures
- Large missing areas
- Thin wires, scratches, or linear damage
- Complex boundaries
- High-frequency textures
- Mixed structural and textured regions

A reconstruction model that performs well for one situation may not be
optimal for another.

ReScene addresses this by placing a **situation-aware analysis and planning
layer** between the user's mask and the reconstruction models.

### Conventional Pipeline

```text
Image
  ↓
Mask
  ↓
One Inpainting Model
  ↓
Output
```

### ReScene Pipeline

```text
Image + Mask
     ↓
Situation Analysis
     ↓
Damage Characterization
     ↓
Reconstruction Planning
     ↓
Model Selection
     ↓
Reconstruction
     ↓
Preservation Verification
     ↓
Final Output
```

---

# ✦ What Makes ReScene Different?

| Conventional Inpainting | ReScene |
|---|---|
| One reconstruction strategy | Situation-aware reconstruction planning |
| Fixed preprocessing | Adaptive context extraction |
| Generation-focused | Generation + preservation |
| Output accepted directly | Preservation verification |
| Model-first pipeline | Analysis → Planning → Reconstruction |
| Entire image can be regenerated | Reconstruction restricted to the mask |

ReScene is therefore designed as a **complete reconstruction system**, rather
than simply a wrapper around a single inpainting model.

---

# ✦ System Architecture

```text
                         ┌─────────────────────┐
                         │      INPUT IMAGE    │
                         │        + MASK       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ ADAPTIVE CONTEXT    │
                         │ EXTRACTION          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   SCENE ANALYZER    │
                         │                     │
                         │ Geometry            │
                         │ Edges               │
                         │ Lines               │
                         │ Texture             │
                         │ Orientation         │
                         │ Complexity          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ DAMAGE              │
                         │ CHARACTERIZATION    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ RECONSTRUCTION      │
                         │ PLANNER             │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
             ┌─────────┐       ┌─────────┐       ┌─────────┐
             │  LaMa   │       │  ZITS   │       │   MAT   │
             └────┬────┘       └────┬────┘       └────┬────┘
                  │                 │                 │
                  └─────────────────┼─────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ SPATIAL             │
                         │ RESTORATION         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ MASK-ONLY           │
                         │ COMPOSITING         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ PRESERVATION GUARD  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ FINAL RECONSTRUCTED │
                         │       IMAGE         │
                         └─────────────────────┘
```

---

# ✦ How ReScene Understands the Situation

ReScene currently uses **computer-vision feature extraction and rule-based
reasoning** to characterize the reconstruction problem.

It does not claim complete semantic understanding of the scene.

## Damage Geometry

The reconstruction mask provides:

- Area ratio
- Bounding box
- Width and height
- Aspect ratio
- Perimeter
- Compactness
- Boundary complexity
- Perimeter-to-area relationship

These measurements help distinguish situations such as:

```text
Thin / Linear Damage
Large Missing Region
Compact Object Removal
Complex Damage Boundary
```

---

## Scene Structure

The surrounding image is analyzed using several visual signals.

### Edge Analysis

ReScene uses edge information to estimate:

- Edge density
- Edge count
- Gradient strength

### Line Analysis

Line structure is estimated using line detection techniques to measure:

- Line density
- Dominant orientation
- Horizontal structure
- Vertical structure
- Diagonal structure
- Orientation entropy

### Texture Analysis

Texture characteristics include:

- Grayscale variance
- Laplacian variance
- Local complexity
- Color statistics

These measurements are combined to produce a scene-structure profile.

---

# ✦ Damage Characterization

The extracted features are converted into normalized situation scores.

The current characterization considers:

```text
Thin Damage
Large Damage
Structural Complexity
Texture Complexity
Scene Complexity
Object Removal
```

The system prioritizes the dominant characteristics before selecting a
reconstruction strategy.

### Example

```text
High perimeter / area
        +
High elongation
        ↓
Thin / Linear Damage
        ↓
LaMa
```

Another example:

```text
Large damaged area
        +
High context requirement
        ↓
Large Missing Region
        ↓
MAT + LaMa
```

---

# ✦ Intelligent Reconstruction Planning

ReScene maps the detected situation to an appropriate reconstruction
strategy.

| Detected Situation | Reconstruction Strategy |
|:---|:---|
| Simple background | **LaMa** |
| General object removal | **LaMa** |
| Thin / linear damage | **LaMa** |
| Textured background | **LaMa + MAT candidate** |
| Structural scene | **ZITS + LaMa** |
| Large missing region | **MAT + LaMa** |
| Complex mixed scene | **ZITS + LaMa + MAT** |

### Important

The reconstruction planner is a **system-level contribution** built around
established pretrained models.

ReScene does **not** claim LaMa, ZITS, or MAT as models developed from
scratch by the team.

---

# ✦ Adaptive Context-Aware Reconstruction

High-resolution images present a challenge for fixed-resolution inpainting
models.

Directly resizing an entire image to the model's input resolution can cause
important local information to occupy very few pixels.

ReScene instead focuses processing around the damaged region.

```text
                 ORIGINAL IMAGE

┌───────────────────────────────────────────┐
│                                           │
│        ┌─────────────────────────┐        │
│        │                         │        │
│        │      DAMAGE REGION      │        │
│        │                         │        │
│        └─────────────────────────┘        │
│                                           │
└───────────────────────────────────────────┘
                     │
                     ▼
              Damage Bounding Box
                     │
                     ▼
              Context Expansion
                     │
                     ▼
              Aspect Balancing
                     │
                     ▼
             Model Resolution
                     │
                     ▼
              Reconstruction
                     │
                     ▼
             Inverse Transform
                     │
                     ▼
              Original Space
```

The context crop is expanded using real surrounding image content rather
than artificial padding.

---

# ✦ Preservation-First Reconstruction

One of ReScene's core principles is:

> **If the user did not ask us to change it, we should not change it.**

The reconstruction model generates the candidate reconstruction, but the
final image is created using the original image outside the reconstruction
mask.

```text
Outside Mask
     ↓
Original Pixels

Inside Mask
     ↓
Reconstructed Pixels
```

Conceptually:

```text
Final Image =
    Original Image × (1 - Mask)
    +
    Reconstruction × Mask
```

This makes the original image the **source of truth** for all pixels outside
the requested reconstruction area.

---

# ✦ Preservation Guard

After reconstruction, ReScene evaluates the region that was intended to
remain untouched.

The Preservation Guard can evaluate:

- Pixel-level differences
- Structural Similarity (SSIM)
- Edge consistency
- Color fidelity

```text
                  RECONSTRUCTED IMAGE
                          │
                          ▼
                 ┌─────────────────┐
                 │ PRESERVATION    │
                 │      GUARD      │
                 └────────┬────────┘
                          │
               ┌──────────┼──────────┐
               ▼          ▼          ▼
            Pixels       SSIM      Edges
               │          │          │
               └──────────┼──────────┘
                          ▼
                 Preservation Metrics
```

The reported metrics are calculated from the actual image data rather than
using hardcoded values.

---

# ✦ Reconstruction Confidence

ReScene also provides a contextual reconstruction-confidence measure.

The current heuristic considers factors including:

- Damage area
- Available surrounding context
- Boundary complexity
- Seam continuity
- Texture consistency

The confidence value is intended as a **contextual indicator**.

It is not presented as a calibrated probability of reconstruction correctness.

---

# ✦ User Interface

ReScene provides an interactive Gradio interface for reconstruction.

### Workflow

```text
Upload Image
     ↓
Define Reconstruction Mask
     ↓
Review Mask
     ↓
Analyze Scene
     ↓
Characterize Damage
     ↓
Generate Reconstruction Plan
     ↓
Run Reconstruction
     ↓
Verify Preservation
     ↓
Review Final Output
```

The interface provides access to:

- Original image
- Reconstruction mask
- Reconstructed image
- Scene analysis
- Damage characterization
- Reconstruction plan
- Preservation metrics
- Reconstruction confidence
- Spatial alignment information

---

# ✦ Mask Convention

ReScene uses a binary reconstruction mask.

```text
WHITE (255)
     ↓
Reconstruct / Inpaint

BLACK (0)
     ↓
Preserve / Leave Untouched
```

---

# ✦ Technology Stack

| Layer | Technology |
|:---|:---|
| Programming | Python 3.12 |
| Deep Learning | PyTorch |
| GPU Acceleration | CUDA |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Image Processing | Pillow |
| Image Metrics | Scikit-Image |
| Web Interface | Gradio |
| General Inpainting | LaMa |
| Structure-Aware Inpainting | ZITS |
| Large-Mask Reconstruction | MAT |

---

# ✦ Repository Structure

```text
ReScene/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── damage_analyzer.py
│   └── verification.py
│
├── core/
│   ├── __init__.py
│   ├── analysis.py
│   ├── compositing.py
│   ├── confidence.py
│   ├── damage_classifier.py
│   ├── preprocessing.py
│   ├── preservation.py
│   ├── reconstruction.py
│   ├── reconstruction_planner.py
│   └── scene_analyzer.py
│
├── examples/
│   ├── sample_image.png
│   └── sample_mask.png
│
├── scripts/
│   ├── create_sample.py
│   ├── run_quality_tests.py
│   ├── test_inpainting.py
│   ├── test_reconstruction_planner.py
│   └── test_scene_analyzer.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ✦ Installation

## Requirements

Recommended environment:

- Windows 10 / 11
- Python 3.12
- NVIDIA GPU recommended
- CUDA-enabled PyTorch
- Sufficient storage for pretrained checkpoints

The current prototype was developed and tested on an NVIDIA RTX 3050
Laptop GPU.

---

## 1. Clone the Repository

```powershell
git clone https://github.com/rufusdj28/HackNex.git
cd HackNex
```

---

## 2. Create or Activate the Virtual Environment

If the provided environment is already configured:

```powershell
.\.venv\Scripts\Activate.ps1
```

For a fresh environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# ✦ Model Setup

Large pretrained checkpoints are intentionally **not stored in GitHub**.

The repository excludes:

```text
models/
*.pt
*.pth
*.onnx
*.ckpt
*.safetensors
```

Expected local structure:

```text
models/
│
├── LaMa-ONNX/
│   └── lama_fp32.onnx
│
├── ZITS_inpainting/
│   └── <ZITS checkpoints>
│
├── MAT/
│   └── <MAT checkpoints>
│
└── ...
```

### LaMa

The current reconstruction pipeline uses the LaMa ONNX checkpoint.

Expected path:

```text
models/LaMa-ONNX/lama_fp32.onnx
```

### ZITS

ZITS checkpoints are used as reconstruction candidates for structure-heavy
scenes.

### MAT

MAT checkpoints are used as candidates for large missing regions and
complex texture reconstruction.

> Download model checkpoints from their respective official repositories and
> comply with their individual licenses and usage restrictions.

---

# ✦ Running ReScene

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Launch the application:

```powershell
python app/main.py
```

The Gradio interface will start locally.

### Basic Workflow

```text
1. Upload an image
2. Define the damaged region
3. Review the mask
4. Run reconstruction
5. Inspect scene analysis
6. Review the reconstruction
7. Inspect preservation metrics
8. Export the result
```

---

# ✦ Command-Line Inference

Generate sample data:

```powershell
python scripts/create_sample.py
```

Run the reconstruction test:

```powershell
python scripts/test_inpainting.py
```

---

# ✦ Quality Verification

ReScene contains an engineering verification suite designed to evaluate
different reconstruction conditions.

Run:

```powershell
python scripts/run_quality_tests.py
```

### Current Test Scenarios

```text
TEST 1  — Plain background
TEST 2  — Detailed high-frequency texture
TEST 3  — Building / structural lines
TEST 4  — Object removal
TEST 5  — Small mask on high-resolution image
TEST 6  — Large mask
TEST 7  — Border-touching mask
TEST 8  — Non-square panoramic image
TEST 9  — Anime / illustration line-art
```

The verification suite checks:

- Output dimensions
- Unmasked pixel preservation
- Preservation metrics
- Reconstruction confidence
- Reconstruction artifacts

> These are engineering validation scenarios and are not intended to
> represent a standardized benchmark.

---

# ✦ Current Implementation

### Implemented

- [x] Adaptive context-aware preprocessing
- [x] Context crop expansion
- [x] Aspect-ratio balancing
- [x] Damage geometry analysis
- [x] Edge analysis
- [x] Line / orientation analysis
- [x] Texture analysis
- [x] Scene complexity analysis
- [x] Damage characterization
- [x] Situation-aware reconstruction planning
- [x] LaMa reconstruction pipeline
- [x] Mask-only compositing
- [x] Preservation Guard
- [x] Reconstruction confidence
- [x] Spatial alignment diagnostics
- [x] Gradio interface
- [x] Engineering verification suite

---

# ✦ Roadmap

## Intelligent Masking

- [ ] Smart Click object selection
- [ ] Smart Circle selection
- [ ] Smart Outline selection
- [ ] Automatic mask refinement
- [ ] Add / Remove mask operations
- [ ] Undo / Redo
- [ ] Mask preview and confirmation

## Reconstruction

- [ ] Automated candidate scoring
- [ ] More robust multi-model reconstruction
- [ ] Improved structural reconstruction
- [ ] Improved large-region reconstruction
- [ ] Automated reconstruction benchmarking

## Video Reconstruction

- [ ] Video input
- [ ] Frame-wise mask propagation
- [ ] Temporal reconstruction
- [ ] Temporal consistency verification
- [ ] Video backend such as E²FGVI-HQ
- [ ] Temporal Preservation Guard

> Video reconstruction is a planned extension and is not part of the current
> image reconstruction release.

---

# ✦ Current Limitations

ReScene is an active prototype.

### Reconstruction Ambiguity

Very large missing regions can be inherently ambiguous because the original
content is unavailable.

### Model Dependency

Reconstruction quality depends on the capabilities and limitations of the
underlying pretrained models.

### Semantic Understanding

The current situation-awareness layer primarily uses measurable
computer-vision features such as geometry, edges, lines, texture, and
orientation.

It does not claim complete semantic understanding of objects or scenes.

### Confidence

The reconstruction confidence value is a contextual heuristic and is not a
calibrated probability.

### Video

Temporal video reconstruction is currently planned and is not part of the
current image pipeline.

---

# ✦ External Models & Components

ReScene builds on established pretrained and open-source computer-vision
components.

## LaMa

**Large Mask Inpainting**

Used as the primary general-purpose reconstruction backend.

Official repository:

https://github.com/advimman/lama

---

## ZITS

**Incremental Structure-Preserving Transformer for Inpainting**

Used as a structure-aware reconstruction candidate.

Official repository:

https://github.com/DQiaole/ZITS_inpainting

---

## MAT

**Mask-Aware Transformer for Large Hole Image Inpainting**

Used as a candidate for large missing regions and complex textures.

Official repository:

https://github.com/fenglinglwb/MAT

---

## Gradio

Used to provide the interactive web interface.

Official repository:

https://github.com/gradio-app/gradio

---

## Licensing

ReScene's original system code is separate from the third-party pretrained
models and libraries used by the project.

Each external component remains subject to its respective:

- License
- Attribution requirements
- Model usage restrictions
- Redistribution requirements

Model weights are not redistributed through this repository.

---

# ✦ Reproducibility

To reproduce the current prototype:

```powershell
git clone https://github.com/rufusdj28/HackNex.git
cd HackNex

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python app/main.py
```

Before running reconstruction, ensure that the required pretrained model
checkpoints are available under the expected `models/` directories.

---

# ✦ HackNex 2026

<div align="center">

## PS06 — Generative Computer Vision & Scene Reconstruction

### ◈ ReScene

**Situation-Aware AI Scene Reconstruction**

<br>

```text
USER INTENT
     ↓
SITUATION AWARENESS
     ↓
RECONSTRUCTION PLANNING
     ↓
SPECIALIZED INPAINTING
     ↓
PRESERVATION VERIFICATION
```

<br>

### Understand the scene.
### Choose the strategy.
### Reconstruct only what is missing.

</div>

---

# ✦ Team

<div align="center">

### ReScene — HackNex 2026

**HNX26PSI06**

Generative Computer Vision & Scene Reconstruction

</div>

---

<div align="center">

### ◈ ReScene

**Reconstruct what is missing. Preserve what isn't.**

</div>
