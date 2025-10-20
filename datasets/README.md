# FPM-BioCell Dataset

A high-quality biomedical dataset for Fourier Ptychographic Microscopy (FPM) reconstruction, designed to support end-to-end supervised learning models. The dataset features diverse biological samples with multi-scale morphological characteristics and well-paired low-resolution (LR) and high-resolution (HR) image pairs.

> **Note:** The FPM-BioCell dataset is publicly released alongside the paper *"Fast High-Fidelity Fourier Ptychographic Microscopy via Wavelet Transform and Linear Attention"*.

## 📋 Dataset Overview
FPM-BioCell is constructed to enhance biomedical relevance and morphological complexity for FPM reconstruction research. It consists of **10 distinct biological samples** covering multi-scale structures from subcellular levels to tissue levels:
1. Ascaris eggs
2. Stratified squamous epithelium
3. Cross section of privet leaf
4. Lymph node
5. Testis section
6. Large intestine section
7. Locust testis meiosis
8. Fig fruit section
9. Fish gill cross section
10. Rat tail cross section

### Key Specifications
- **Original Image Sizes**:
  - LR images: 2048×2048 pixels
  - HR reference images: 12288×12288 pixels (reconstructed via WL-FPM)
- **Final Dataset Scale**: 586 valid paired patches (500 for training, 86 for testing)

## 📥 Dataset Download
To access the dataset, please use the following links: [**Kaggle**]()

## 🔗 Code Availability
This FPM-BioCell dataset is designed for the [**WM-FPM**](https://github.com/ww20250822/WM-FPM) model. The official code of WM-FPM, which supports loading and training on this dataset, is available in the corresponding repository.

## 📝 Citation
If you use this code for your research, please cite our paper.