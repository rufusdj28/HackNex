<div align="center">

# ◈ ReScene

### **Situation-Aware AI Scene Reconstruction**

**Reconstruct what is missing. Preserve what isn't.**

[![HackNex 2026](https://img.shields.io/badge/HackNex-2026-111827?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rufusdj28/HackNex)
[![Problem Statement](https://img.shields.io/badge/PS06-Scene%20Reconstruction-2563EB?style=for-the-badge)](#)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-See%20Components-6B7280?style=for-the-badge)](#external-components)

<br>

> **ReScene doesn't blindly inpaint. It analyzes the situation first.**

<br>

</div>

---

## ◇ The Idea

Traditional image inpainting asks one question:

> **"What should I generate here?"**

ReScene asks a more useful question:

> **"What kind of scene am I reconstructing?"**

Before reconstruction, ReScene analyzes the damaged region and its surrounding
visual context — geometry, edges, lines, texture, orientation, and complexity.

It then uses that information to determine an appropriate reconstruction
strategy.

```text
                    ┌──────────────────┐
                    │    INPUT IMAGE   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   DAMAGE MASK    │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │   SITUATION ANALYSIS  │
                  │                       │
                  │ Geometry • Edges      │
                  │ Lines • Texture       │
                  │ Orientation • Context │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  RECONSTRUCTION       │
                  │      PLANNER          │
                  └───────────┬───────────┘
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
              ┌──────┐    ┌──────┐    ┌──────┐
              │ LaMa │    │ ZITS │    │ MAT  │
              └───┬──┘    └───┬──┘    └───┬──┘
                  └────────────┼────────────┘
                               ▼
                    ┌──────────────────┐
                    │ MASK-ONLY        │
                    │ COMPOSITING      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ PRESERVATION     │
                    │     GUARD        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ FINAL RECONSTRUCTION │
                    └──────────────────┘
