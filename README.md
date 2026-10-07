<div align="center">

<a href="https://github.com/">
  <img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=700&size=34&duration=2800&pause=900&color=36BCF7&center=true&vCenter=true&width=850&lines=ReScene;Structure-Preserving+AI+Scene+Reconstruction;Context-aware+visual+repair+with+pixel+preservation" alt="ReScene animated title" />
</a>

<p><strong>Situation-aware image reconstruction for damaged, missing, or unwanted regions.</strong></p>

<p>
  <img src="https://img.shields.io/badge/HackNex-2026-7C3AED?style=for-the-badge" alt="HackNex 2026" />
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Gradio-5.49.1-FF7C00?style=for-the-badge" alt="Gradio" />
  <img src="https://img.shields.io/badge/LaMa-ONNX-00A67E?style=for-the-badge" alt="LaMa ONNX" />
</p>

<p><em>Repair what matters. Preserve what does not.</em></p>

</div>

<br />

## ✦ What is ReScene?

ReScene is a context-aware AI image reconstruction system built for the HackNex 2026 challenge. It reconstructs selected image regions while keeping every unselected pixel unchanged.

> **The reconstruction model may change only the user-selected mask. Everything outside the mask is preserved from the original image.**

## ✨ Highlights

| Capability | What it does |
| --- | --- |
| **Adaptive context cropping** | Expands the damaged region with useful surrounding visual context before inference. |
| **LaMa ONNX inference** | Runs the existing LaMa reconstruction backend through ONNX Runtime. |
| **Mask-only compositing** | Composites reconstructed pixels strictly inside the binary damage mask. |
| **Scene Analyzer** | Estimates structure, texture, geometry, context quality, and damage scale. |
| **Reconstruction Planner** | Produces a situation-aware reconstruction strategy. |
| **Preservation Guard** | Measures pixel, structural, edge, and color preservation outside the mask. |
| **Alignment inspector** | Exposes crop bounds, scale factors, padding, and transform telemetry. |
| **Interactive Gradio UI** | Upload, paint a mask, reconstruct, inspect, and download from one workspace. |

## 🧭 Reconstruction pipeline

```mermaid
flowchart LR
    A[Upload image] --> B[Paint binary mask]
    B --> C[Scene Analyzer]
    C --> D[Damage Classifier]
    D --> E[Reconstruction Planner]
    E --> F[Adaptive Context Crop]
    F --> G[LaMa ONNX]
    G --> H[Inverse Crop]
    H --> I[Mask-only Compositing]
    I --> J[Preservation Guard]
    J --> K[Final image + reports]

    classDef input fill:#172554,stroke:#60a5fa,color:#fff;
    classDef ai fill:#312e81,stroke:#a78bfa,color:#fff;
    classDef safe fill:#064e3b,stroke:#34d399,color:#fff;
    class A,B input;
    class C,D,E,F,G,H ai;
    class I,J,K safe;
```

## 🛡️ Preservation guarantee

Let `M` be the final binary mask, `I` the original image, and `R` the reconstructed image:

```text
output[p] = R[p]  when M[p] = 255
output[p] = I[p]  when M[p] = 0
```

The Preservation Guard verifies the unmasked area using exact unchanged-pixel comparison, pixel difference, SSIM, edge preservation, and color preservation.

## 📁 Project structure

```text
PS06/
├── app/
│   ├── main.py                    # Gradio application and reconstruction workflow
│   ├── damage_analyzer.py         # Application-level damage analysis helpers
│   ├── verification.py            # Output verification utilities
│   └── __init__.py
├── core/
│   ├── reconstruction.py          # LaMa ONNX inference and context-aware reconstruction
│   ├── preprocessing.py           # Mask preprocessing and input preparation
│   ├── tiled_inpainting.py        # Tiled / large-image inpainting helpers
│   ├── compositing.py             # Mask-only output compositing
│   ├── preservation.py            # Preservation Guard metrics
│   ├── confidence.py              # Reconstruction confidence estimation
│   ├── analysis.py                # Image and mask analysis
│   ├── scene_analyzer.py          # Scene structure and context analysis
│   ├── damage_classifier.py       # Damage characterization
│   ├── reconstruction_planner.py  # Situation-aware reconstruction planning
│   └── __init__.py
├── models/
│   ├── LaMa-ONNX/                 # Existing LaMa ONNX assets
│   ├── torch/                     # Existing PyTorch model assets
│   ├── IOPaint/                   # Existing inpainting resources
│   └── ZITS_inpainting/           # Existing reconstruction resources
├── data/output/                   # Generated reconstruction artifacts
├── examples/                      # Sample image and mask assets
├── scripts/                       # Smoke tests and quality checks
├── requirements.txt
└── README.md
```

## 🚀 Run locally on Windows

The project uses the existing isolated virtual environment.

```powershell
.\.venv\Scripts\Activate.ps1
python app\main.py
```

Then open the local Gradio URL shown in the terminal.

### Basic workflow

1. Upload an image.
2. Paint the region to reconstruct with the white brush.
3. Click **Reconstruct**.
4. Review the original, mask, reconstructed image, analysis, and preservation reports.
5. Download the generated image.

## 🧪 Verification

Run the existing quality checks:

```powershell
python scripts\run_quality_tests.py
```

The checks cover plain backgrounds, detailed textures, structural lines, object removal, small and large masks, border-touching masks, panoramic images, and illustration line art.

For the lightweight smoke test:

```powershell
python scripts\create_sample.py
python scripts\test_inpainting.py
```

## ⚙️ Environment

The current environment includes Python 3.12, PyTorch 2.5.1 + CUDA 12.4, ONNX Runtime GPU 1.30.0, OpenCV 4.11.0.86, scikit-image 0.24.0, Pillow 9.5.0, and Gradio 5.49.1.

Dependencies are pinned in [`requirements.txt`](requirements.txt). Existing model weights remain under `models/` and are not downloaded by the application at runtime.

## 🗺️ Roadmap

- [x] Context-aware image reconstruction
- [x] Mask-only compositing
- [x] Scene-aware planning
- [x] Preservation metrics and debug telemetry
- [x] Interactive Gradio prototype
- [ ] Advanced intent-aware selection
- [ ] Video reconstruction and frame propagation

## 📜 Project Context

Built for **HackNex 2026 — HNX26PSI06**. The project includes third-party model and research resources under their respective directories; consult those directories for their individual license terms.

<div align="center">

<br />

**ReScene** · *Context-aware repair with a preservation-first mindset.*

</div>
