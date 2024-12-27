# **Spectral Band Attention Networks for Efficient Multi-Feature Fusion in Hyperspectral and RGB Data with Ensemble Deep Learning Networks**

#### Overview  
This project presents a **dual-branch network** integrating hyperspectral imaging (HSI) and RGB data for accurate wheat seed classification. The network leverages the most informative spectral bands using a novel **Spectral Band Attention Network (SBAN)** and fine-tuned RGB models to achieve robust multi-modal feature fusion.

---

#### Key Components  

1. **Model Architecture**  
   - **Dual-Branch Network:**  
     - **Branch 1 (HSI Data):** Customized DenseNet model trained on selected spectral bands using SBAN.  
     - **Branch 2 (RGB Data):** Fine-tuned deep convolutional models (DenseNet121, ResNet34, GoogLeNet) for feature extraction from RGB images.  
   - **Classifier:** A Support Vector Machine (SVM) combines the outputs of both branches for final wheat seed class prediction.

    [pipeline_image](

2. **Feature Selection**  
   - **Spectral Bands:** Informative bands selected using SBAN.  
   - Comparison with traditional selection methods:  
     - **Successive Projections Algorithm (SPA)**  
     - **PCA-loading (Principal Component Analysis)**  

3. **Multi-Feature Fusion**  
   - Combines features extracted from hyperspectral and RGB data for enhanced performance.

---

#### Evaluation  

The performance of the dual-branch network was systematically compared by:  
- Varying the number of spectral bands selected.  
- Comparing SBAN-based selection with SPA and PCA-loading methods.  

---

#### Results  

This comprehensive approach improves wheat seed classification performance by:  
- Attending to the most critical spectral bands using SBAN.  
- Fine-tuning RGB models for optimized feature extraction.  
- Effectively integrating multi-modal features from hyperspectral and RGB imaging.  

---

#### Applications  

This methodology has significant potential for applications in:  
- Agricultural imaging and precision farming.  
- High-precision classification tasks requiring multi-modal data fusion.  

---

#### Citation  

If you use this work, please cite:  
> [Provide citation details here]
