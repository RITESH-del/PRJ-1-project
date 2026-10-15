## References

### Deep Learning & Foundation Models
- Training DinoV2 Model: https://huggingface.co/docs/transformers/model_doc/dinov2
  - Official documentation for fine-tuning DINOv2 vision transformers for downstream image classification.
- Torchvision DenseNet Architecture: https://pytorch.org/vision/main/models/densenet.html
  - Convolutional baseline architecture utilized for comparative evaluation on chest radiograph classification.
- DINOv2: Learning Robust Visual Features without Supervision: https://arxiv.org/abs/2304.07193
  - Self-supervised Vision Transformer foundation model; native $518 \times 518$ input resolution yields $37 \times 37 = 1,369$ visual tokens for dense pulmonary lesion localization.

### Preprocessing & Image Processing Algorithms
- Bilateral Filtering for Gray and Color Images: https://doi.org/10.1109/ICCV.1998.710815
  - C. Tomasi and R. Manduchi, IEEE ICCV 1998. Edge-preserving smoothing applied at full resolution before downsampling to suppress sensor noise while preserving 1–2 mm micro-nodules and thin cavitary margins.
- Contrast Limited Adaptive Histogram Equalization (CLAHE): https://doi.org/10.1016/B978-0-12-336156-1.50061-6
  - K. Zuiderveld, Graphics Gems IV, 1994. Dynamic tile grid scaling ($(16, 16)$ at $518 \times 518$) and contrast harmonization across heterogeneous digital detector cohorts.
- Otsu's Optimum Threshold Selection Method: https://doi.org/10.1109/TSMC.1979.4310076
  - N. Otsu, IEEE Transactions on Systems, Man, and Cybernetics, 1979. Non-parametric threshold selection adapted in `detect_thorax_roi` and `generate_lung_mask` for body contour and lung parenchyma segmentation.
- Unsharp Masking and High-Pass Edge Residuals in Digital Radiography: https://doi.org/10.1148/radiology.186.2.8421764
  - M. Prokop et al., Radiology 1993. High-contrast edge residual extraction utilized for Channel 2 in the 3-channel dual-representation stream to accentuate micro-calcifications and cavitary rims.
- Lanczos Filtering in Digital Image Resampling: https://doi.org/10.1175/1520-0450(1979)018<1016:LFIOAT>2.0.CO;2
  - C. E. Duchon, Journal of Applied Meteorology, 1979. High-order sinc-windowed interpolation used in `resize_with_padding` to avoid aliasing and moiré artifacts during high-resolution scaling.
- Chest X-ray Preprocessing Reference: https://www.kaggle.com/code/lakshmimahabaleshwar/chest-xray-preprocessing-kaggle
  - Exploratory Kaggle reference notebook adapted for initial ROI extraction, CLAHE equalization, and normalization.

### Clinical Guidance & Benchmark Radiograph Cohorts
- WHO Chest Radiography in Tuberculosis Detection - Operational Guidance: https://www.who.int/publications/i/item/9789241511506
  - World Health Organization, 2016. Establishes the clinical imperative for apical lung field preservation: post-primary adult pulmonary tuberculosis predominantly localizes in the lung apices and upper lobes.
- Shenzhen and Montgomery County Tuberculosis Datasets: https://doi.org/10.3978/j.issn.2223-4292.2014.11.20
  - S. Jaeger et al., Quantitative Imaging in Medicine and Surgery, 2014. Public chest radiograph benchmarks providing high-resolution digitized X-rays with verified clinical TB and Normal ground truth.
- NIH ChestX-ray14 Benchmark: https://arxiv.org/abs/1705.02315
  - X. Wang et al., IEEE CVPR 2017. Large-scale clinical center radiograph dataset utilized for training cohort representation.
  