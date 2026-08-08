# KDD'26 PyPOTS Hands-on Tutorial

## 📌 Event Details

- **Conference**: ACM SIGKDD 2026 (KDD '26 Hands-on Tutorial, System-Focused)
- **Date & Time**: 🗓️ August 9, 2026 (1:00 PM – 5:00 PM KST)
- **Location**: 📍 Halla A, ICC, Jeju, South Korea
- **Official Webpage**: [https://pypots.com/events/KDD2026Tutorial](https://pypots.com/events/KDD2026Tutorial)

---

## 📖 Overview

Partially-observed time series (POTS) are ubiquitous in real-world applications (healthcare vitals, IoT sensor networks, meteorology, and financial logs). However, conventional ML toolchains isolate missing-value imputation from downstream predictive modeling, causing error propagation, poor reusability, and fragmented codebases.

This **KDD 2026 Hands-on Tutorial** introduces **PyPOTS**, an open-source Python ecosystem for end-to-end data mining and machine learning on POTS. We present unified workflows covering missingness simulation, data preprocessing, neural modeling, and evaluation across core time-series tasks: **Imputation, Forecasting, Classification, Clustering, and Anomaly Detection**.

---

## 📂 Notebooks & Material Directory

This directory contains executable Jupyter Notebooks:

| File | Description | Target Audience |
| :--- | :--- | :--- |
| 📓 [`pypots_tutorial_part1_pots_pipeline.ipynb`](./pypots_tutorial_part1_pots_pipeline.ipynb) | **Part I: Apply PyPOTS to Time Series Analysis**<br>• POTS fundamentals & missingness mechanisms (MCAR, MAR, MNAR)<br>• Data preparation & missingness simulation with PyGrinder & BenchPOTS<br>• Unified model training across Imputation, Forecasting, Classification, Clustering, Anomaly Detection<br>• Metric selection, error analysis, and visualization | Practitioners, Data Scientists, & Software Engineers |
| 📓 [`pypots_tutorial_part2_extending_pypots.ipynb`](./pypots_tutorial_part2_extending_pypots.ipynb) | **Part II: Extend PyPOTS to Specialties**<br>• PyPOTS 3-Layer Architecture (Backbone `nn/modules`, Core `core.py`, Wrapper `model.py`) & 3 Integration Paths<br>• Multi-task standard neural network integration (Imputation, Forecasting, Classification)<br>• Complex multi-optimizer GAN models (`CustomGANImputer`) & Non-NN statistical algorithm wrapping (`CustomInterpolationImputer`)<br>• Domain constraints (`deltas`, feature-wise masks, weighted MSE)<br>• Standardized unit testing (`test_0_fit` -> `test_4_lazy_loading`) & HDF5 lazy loading<br>• Open-source contribution workflow (Black, NumPy docstrings, `pypots-cli dev`) & pre-PR checklist<br>• **Real-World Assignment**: Continuous 2D time series sliding-window slicing via `benchpots.utils.sliding_window` | Developers, Researchers, & Core Contributors |

---

## 👥 Tutors & Presenters

### 🛠️ Tutorial Organizers
* **Wenjie Du** — Founder & R&D Lead of PyPOTS Research
* **Yiyuan Yang** — PhD Student @ University of Oxford, Researcher @ PyPOTS Research
* **Tianxiang Zhan** — Researcher @ PyPOTS Research
* **Qingsong Wen** — Head of AI & Chief Scientist @ Squirrel Ai Learning, Advisor of PyPOTS Research

### 🎤 On-site Presenter
* **Yinghao Zhu** — PhD Student @ The University of Hong Kong (*On-site Speaker*)

---

## 🚀 Quickstart & Prerequisites

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.10
- PyPOTS == 1.5

### Installation
```bash
pip install pypots==1.5
```

---

## 📜 Citation

If you use PyPOTS or reference this tutorial in your research, please cite:

```bibtex
@inproceedings{du2026pypotstutorial,
  title     = {End-to-End Learning for Partially-Observed Time Series with PyPOTS},
  author    = {Du, Wenjie and Yang, Yiyuan and Zhan, Tianxiang and Wen, Qingsong},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026}
}

@article{du2023pypots,
  title   = {{PyPOTS: A Python Toolkit for Data Mining on Partially-Observed Time Series}},
  author  = {Wenjie Du},
  journal = {SIGKDD MiLeTS Workshop},
  year    = {2023}
}
```
