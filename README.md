# DMP-PCFC
Python training and test data, code, and results of a DMP-PCFC model for multi-energy loads forecasting in integrated energy systems.


```markdown
# DMP-PCFC: Dual resolution channel multi-period cross reconstruction parallel closed-form continuous-time network

[![Python 3.10](https://img.shields.io/badge/Python-3.10.13-blue?logo=python)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0.1%2Bcu118-red?logo=pytorch)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of DMP-PCFC for multi-step energy loads prediction, using meteorological data from the Integrated Energy System (lES) at Arizona State
University (ASU) Tempe Campus (http://cm.asu.edu/) and National Centers for Environmental Information (https://www.ncei.noaa.gov/).

![Prediction Visualization](assets/24-steps/)

## 🚀 Quick Start

### Prerequisites
- **NVIDIA GPU** with CUDA 11.8 support
- Python 3.10.13
- PyTorch 2.0.1
- numpy 1.26.0
- pandas 2.0.3

### 1. Clone Repository
```bash
git clone https://github.com/nuist-xf/DMP-PCFC.git
cd DMP-PCFC
```

### 2. Install Dependencies
```bash
conda create -n dmp-pcfc python=3.10.13
conda activate dmp-pcfc
pip install -r requirements.txt
```

### 3. Run Pipeline

# train（Automatic use of the GPU）
python train.py

# Generate the results of the prediction metrics
python Save_result_multipredict.py
```

## 🗂️ Repository Structure
```
.
├── assets/                 # Visualization
│   ├── 24-steps/          # 24-hours
│   ├── 48-steps/          # 48-hours
│   ├── 72-steps/          # 72-hours
│   └── 96-steps/          # 96-hours
│
├── data/                  # dataset
│   ├── dataset_input.csv  # Pre-processed sequence data
│
├── model/                 # The optimal weight of the model obtained by the experiment
│   ├── 24-steps/model.pt  # 24-hours
│   ├── 48-steps/model.pt  # 48-hours
│   ├── 72-steps/model.pt  # 72-hours
│   └── 96-steps/model.pt  # 96-hours
├── result/                # The optimal results obtained by the experiment
│   ├── 24-steps/          # 24-hours
│   ├── 48-steps/          # 48-hours
│   ├── 72-steps/          # 72-hours
│   └── 96-steps/          # 96-hours
│
├── metrics.py             # Evaluation indicators（RMSE/MAE/MAPE）
├── train.py               # Main training script
├── trainer.py             # Management of the training process (Optimizer, etc.)
├── torch_cfc.py           # DMP-PCFC model implementation
├── Save_result_multipredict.py # Compute the metric result for the entire test set and save
├── Save_result.py         # Compute the metric result for the first moment of each day of the test set
└── util.py                # Data Loader

```



## 📧 Contact
For technical support or data access requests:
- **Liguo Weng**  
  📧 002311@nuist.edu.cn 
  🏛 Nanjing University of Information Science & Technology

## ⚖️ License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
```

