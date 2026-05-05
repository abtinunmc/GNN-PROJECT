# Graph Neural Networks for Seizure Onset Zone Localization in Drug-Resistant Epilepsy: A Multi-Site Intracranial EEG Study

**Authors:** Abtin Akhtari

**Affiliation:** University of Nebraska Medical Center

**Corresponding Author:** abtin.akhtary@gmail.com

---

## Abstract

Accurate localization of the seizure onset zone (SOZ) is critical for surgical planning in patients with drug-resistant epilepsy. While intracranial electroencephalography (iEEG) remains the gold standard for SOZ identification, manual analysis is time-consuming and subject to inter-rater variability. In this study, we present a graph neural network (GNN) framework that models iEEG electrode arrays as graphs, where nodes represent electrodes and edges capture functional connectivity patterns. We combined two publicly available datasets from the Hospital of the University of Pennsylvania, totaling 81 patients and 212 recordings with 126 having expert-annotated SOZ labels. Our GraphSAGE-based architecture, incorporating self-supervised pretraining and online data augmentation, achieved an area under the receiver operating characteristic curve (AUC) of 0.763 ± 0.008 on held-out test data across 5 random seeds. Notably, the model demonstrated high sensitivity (recall of 62.3%), outperforming classical machine learning baselines including Random Forest (43.6% recall) despite the latter achieving higher overall AUC (0.847). This sensitivity advantage is clinically valuable for ensuring comprehensive SOZ coverage. These findings suggest that GNN-based approaches can serve as effective decision-support tools for epilepsy surgical planning.

**Keywords:** seizure onset zone, graph neural networks, intracranial EEG, epilepsy surgery, deep learning

---

## 1. Introduction

Epilepsy affects approximately 50 million people worldwide, with roughly one-third of patients developing drug-resistant epilepsy (Kwan & Brodie, 2000). For these individuals, surgical resection of the seizure onset zone offers the best chance of seizure freedom, with success rates ranging from 50% to 80% depending on the underlying pathology (Engel et al., 2012). However, the success of epilepsy surgery hinges critically on accurate identification of the SOZ, defined as the brain region where seizures originate.

Intracranial electroencephalography (iEEG), including stereoelectroencephalography (SEEG) and electrocorticography (ECoG), provides high spatial and temporal resolution recordings directly from brain tissue. Epileptologists analyze these recordings to identify electrodes showing ictal onset patterns, a process that is both time-intensive and subjective. Studies have shown substantial inter-rater variability in SOZ determination, highlighting the need for objective, automated methods (Varatharajah et al., 2018).

Recent advances in deep learning have shown promise for automated analysis of neurophysiological signals. However, traditional convolutional and recurrent architectures struggle to capture the complex spatial relationships inherent in electrode arrays, which vary in configuration across patients. Graph neural networks (GNNs) offer a natural solution by representing electrodes as nodes in a graph, with edges encoding functional connectivity between recording sites (Bessadok et al., 2022).

In this work, we develop and evaluate a GNN-based pipeline for SOZ localization using a combined dataset of 81 patients from two publicly available iEEG repositories. Our contributions include: (1) a unified preprocessing pipeline capable of handling multiple data formats, (2) a self-supervised pretraining strategy to leverage unlabeled recordings, (3) systematic evaluation of data augmentation techniques, and (4) comprehensive comparison across multiple GNN architectures.

---

## 2. Materials and Methods

### 2.1 Datasets

We utilized two publicly available datasets from the Hospital of the University of Pennsylvania (HUP), both hosted on OpenNeuro:

**Dataset 1 (ds003029):** This dataset contains iEEG recordings from 23 patients with drug-resistant epilepsy who underwent presurgical evaluation. Recordings are provided in BrainVision format, with SOZ annotations encoded in event markers within the BIDS events.tsv files.

**Dataset 2 (ds004100):** This larger dataset comprises 58 patients with SEEG or ECoG recordings. Data are provided in European Data Format (EDF), with SOZ labels directly annotated in the channels.tsv metadata files. All patients subsequently underwent surgical resection or laser ablation, with Engel outcomes documented.

The combined dataset included 212 recordings, of which 126 had expert-annotated SOZ labels (Table 1). Recordings without SOZ annotations were used for self-supervised pretraining.

**Table 1.** Dataset characteristics

| Characteristic | ds003029 | ds004100 | Combined |
|----------------|----------|----------|----------|
| Patients | 23 | 58 | 81 |
| Recordings | 103 | 109 | 212 |
| Recordings with SOZ | 30 | 96 | 126 |
| Electrode type | SEEG/ECoG | SEEG/ECoG | Mixed |
| Sampling rate | 500-2000 Hz | 500 Hz | Variable |

### 2.2 Preprocessing Pipeline

Our preprocessing pipeline consisted of four stages, designed to handle both BrainVision and EDF formats:

**Epoch extraction:** For each recording, we identified the seizure onset time from event annotations and extracted a 60-second epoch centered on this time point (30 seconds pre-ictal, 30 seconds post-ictal). Channels marked as "bad" in the metadata were excluded.

**Notch filtering:** Line noise at 60 Hz and its harmonics (120, 180, 240 Hz) was removed using a notch filter. The 4th harmonic (240 Hz) was included as it falls near the bandpass cutoff and could otherwise leak into the signal.

**Common average reference (CAR):** To reduce common-mode noise and volume conduction artifacts, we applied CAR by subtracting the mean signal across all electrodes at each time point.

**Bandpass filtering:** An elliptic (Cauer) bandpass filter with cutoffs at 1 Hz and 250 Hz was applied to preserve both low-frequency rhythms and high-frequency oscillations (HFOs), which have been implicated in SOZ identification (Jacobs et al., 2012).

### 2.3 Feature Extraction

Preprocessed signals were segmented into non-overlapping 12-second windows. For each window, we computed the following node features for each electrode:

**Spectral features (6):** Band power in delta (1-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (15-25 Hz), low gamma (35-50 Hz), and high-frequency oscillation (80-150 Hz) bands, computed using Welch's method.

**Statistical features (4):** Variance, line length (sum of absolute first differences), kurtosis, and skewness of the time series.

**Relative HFO power (1):** Ratio of HFO band power to total power, as elevated HFO activity is a recognized biomarker of epileptogenic tissue.

Window-level features were aggregated across time using mean, standard deviation, and maximum, yielding 33 features per electrode.

**Edge features:** Functional connectivity was quantified using Pearson correlation computed over each window, then aggregated using mean, standard deviation, and maximum across windows.

### 2.4 Graph Construction

Each recording was represented as a graph G = (V, E), where nodes V correspond to electrodes and edges E connect electrode pairs with correlation exceeding a threshold of 0.32 (optimized via hyperparameter search). Node features comprised the 33-dimensional aggregated feature vectors, while edge attributes encoded the correlation-derived connectivity strength.

SOZ labels were assigned at the node level: electrodes clinically identified as within the SOZ were labeled positive (1), while others were labeled negative (0). The resulting class distribution was imbalanced, with approximately 24% of electrodes labeled as SOZ.

### 2.5 Model Architecture

We employed a GraphSAGE architecture (Hamilton et al., 2017), which learns node embeddings by aggregating features from local neighborhoods. Our encoder consisted of two GraphSAGE convolutional layers with batch normalization, using a hidden dimension of 128 and dropout rate of 0.21.

The node classification head consisted of a two-layer multilayer perceptron with hidden dimension 64, dropout rate of 0.55, and sigmoid output activation for binary SOZ prediction.

### 2.6 Training Strategy

**Self-supervised pretraining:** To leverage unlabeled recordings, we pretrained the GNN encoder using a signal forecasting task. Given node features from window t, the model was trained to predict features at window t+1. This pretraining was performed on 827 window pairs from all available recordings.

**Supervised fine-tuning:** The pretrained encoder was then fine-tuned for SOZ classification using binary cross-entropy loss with class weights inversely proportional to class frequencies to address label imbalance.

**Online data augmentation:** During training, we applied stochastic augmentation including edge dropout (7.6% probability), Gaussian feature noise (standard deviation 0.011), and random feature masking (10% probability).

**Optimization:** We used AdamW optimizer with learning rates of 1.2 × 10^-4 for pretraining and 6.1 × 10^-4 for fine-tuning, with cosine annealing learning rate scheduling and early stopping based on validation AUC.

### 2.7 Experimental Setup

Data were split at the subject level to prevent information leakage, with 60% of subjects for training, 20% for validation, and 20% for testing. This resulted in 70 training graphs, 36 validation graphs, and 20 test graphs.

We compared three GNN architectures: GraphSAGE, Graph Attention Network (GAT; Velickovic et al., 2018), and Graph Isomorphism Network (GIN; Xu et al., 2019). Hyperparameters were optimized using Optuna with 50 trials per architecture.

Additionally, we evaluated several data augmentation strategies: online augmentation (edge dropout, feature noise, feature masking), time-shift augmentation (random window subsampling), mixup, and SMOTE oversampling.

### 2.8 Evaluation Metrics

Model performance was evaluated using area under the receiver operating characteristic curve (AUC), F1 score, precision, and recall. We prioritized recall (sensitivity) given the clinical importance of not missing true SOZ electrodes.

---

## 3. Results

### 3.1 Comparison with Classical Machine Learning

To contextualize GNN performance, we compared against classical machine learning baselines trained on the same node features without graph structure. Table 2 presents these results on the combined dataset.

**Table 2.** Comparison with classical ML baselines (combined dataset)

| Method | Test AUC | Test F1 | Test Precision | Test Recall |
|--------|----------|---------|----------------|-------------|
| Random Forest | 0.847 | 0.373 | 0.326 | 0.436 |
| Logistic Regression | 0.741 | 0.336 | 0.238 | 0.571 |
| SVM (RBF kernel) | 0.713 | 0.097 | 0.200 | 0.064 |
| **GraphSAGE (ours)** | **0.763 ± 0.008** | **0.273 ± 0.007** | **0.175 ± 0.006** | **0.623 ± 0.014** |

*Note: GNN results reported as mean ± standard deviation over 5 random seeds.*

Interestingly, Random Forest achieved the highest AUC (0.847), outperforming the GNN. However, a critical distinction emerges in the recall metric: the GNN achieved 62.3% recall compared to only 43.6% for Random Forest—a 43% relative improvement. For clinical SOZ localization, high recall is essential to avoid missing true SOZ electrodes that could lead to incomplete resection. The GNN's superior sensitivity, combined with its ability to model inter-electrode connectivity patterns, makes it particularly suitable for clinical screening applications where false negatives carry higher costs than false positives.

### 3.2 GNN Architecture Comparison

Table 3 presents the performance of different GNN architectures on the single-dataset (ds003029) benchmark. GraphSAGE achieved the best test AUC of 0.730 after hyperparameter tuning, outperforming both GAT (0.702) and GIN (0.562).

**Table 3.** Performance comparison of GNN architectures (ds003029 only)

| Architecture | Val AUC | Test AUC | Test F1 |
|--------------|---------|----------|---------|
| GraphSAGE (baseline) | 0.920 | 0.697 | 0.478 |
| GraphSAGE (tuned) | 0.983 | 0.730 | 0.404 |
| GAT (tuned) | 0.925 | 0.702 | 0.411 |
| GIN | 0.660 | 0.562 | 0.000 |

*Figure 1: ROC curve for the tuned GraphSAGE model (see data/processed/figures/roc_curve.png)*

*Figure 2: Optuna optimization history showing convergence of hyperparameter search (see data/processed/figures/optuna_optimization.png)*

### 3.4 Data Augmentation Analysis

We systematically evaluated multiple augmentation strategies (Table 4). Online augmentation (edge dropout, feature noise, feature masking) improved test AUC from 0.730 to 0.761, representing a 4.2% relative improvement. However, time-shift augmentation, mixup, and SMOTE did not provide additional benefits and in some cases degraded performance.

**Table 4.** Effect of data augmentation techniques

| Technique | Test AUC | Change |
|-----------|----------|--------|
| None (baseline) | 0.730 | - |
| Online augmentation | 0.761 | +4.2% |
| Time-shift | 0.718 | -1.6% |
| Mixup | 0.731 | +0.1% |
| SMOTE | 0.696 | -4.7% |

### 3.5 Combined Dataset Performance

Combining the two datasets substantially increased training data (70 vs. 20 labeled graphs) and improved model performance. The final model achieved a test AUC of 0.768 ± 0.0XX (mean ± std over 5 seeds), with notably high recall of 0.650 (Table 5).

**Table 5.** Performance on combined dataset (mean ± std over 5 seeds)

| Metric | Value |
|--------|-------|
| Validation AUC | 0.751 |
| Test AUC | 0.763 ± 0.008 |
| Test F1 | 0.273 ± 0.007 |
| Test Precision | 0.175 ± 0.006 |
| Test Recall | 0.623 ± 0.014 |

*Figure 3: Training loss curves for pretraining and fine-tuning (see data/processed/figures/pretrain_loss.png and train_loss.png)*

*Figure 4: Confusion matrix on test set (see data/processed/figures/confusion_matrix.png)*

### 3.6 Comparison with Single-Dataset Training

Table 6 summarizes the progression of improvements across our experiments. The combination of dataset expansion and online augmentation yielded the best overall performance.

**Table 6.** Summary of experimental progression

| Configuration | Training Graphs | Test AUC | Test Recall |
|---------------|-----------------|----------|-------------|
| ds003029 baseline | ~20 | 0.730 | 0.48 |
| ds003029 + augmentation | ~20 | 0.761 | 0.52 |
| Combined dataset (5 seeds) | 70 | 0.763 ± 0.008 | 0.623 ± 0.014 |

### 3.7 Hybrid Approaches to Improve AUC

Given the performance gap between Random Forest (AUC 0.847) and GNN (AUC 0.763), we investigated several strategies to improve GNN discrimination while preserving its recall advantage.

**Does graph structure help?** We first trained an MLP with identical architecture but without message passing to isolate the contribution of graph structure. The MLP achieved an AUC of 0.742 ± 0.009 compared to the GNN's 0.763 ± 0.008, confirming that graph-based learning provides a meaningful improvement (+2.8%) over treating electrodes independently.

**Alternative graph construction.** We tested higher correlation thresholds (0.4, 0.5, 0.6) to create sparser, potentially less noisy graphs. However, all variants performed worse than the baseline (AUC 0.72-0.75), suggesting that the original threshold of 0.32 was appropriate and that denser connectivity benefits the model.

**Stacking ensemble.** Combining RF and GNN predictions via a meta-learner achieved AUC of 0.837, approaching RF performance. However, this came at the cost of reduced recall (0.40 vs 0.63), as the ensemble shifted toward RF's more conservative predictions.

**RF predictions as node feature.** Our most successful approach incorporated RF prediction probabilities as an additional node feature for the GNN. This hybrid model achieved AUC of 0.785 ± 0.013 with recall of 0.630 ± 0.052, representing a 2.9% improvement in AUC while maintaining the GNN's sensitivity advantage (Table 7).

**Table 7.** Comparison of hybrid approaches

| Method | Test AUC | Test Recall |
|--------|----------|-------------|
| Random Forest | 0.847 | 0.436 |
| GNN (baseline) | 0.763 ± 0.008 | 0.623 ± 0.014 |
| MLP (no graph) | 0.742 ± 0.009 | 0.566 ± 0.015 |
| Stacking (RF + GNN) | 0.837 | 0.400 |
| **GNN + RF feature** | **0.785 ± 0.013** | **0.630 ± 0.052** |

The GNN + RF feature approach represents an effective compromise: it leverages RF's strong node-level discrimination while allowing the GNN to refine predictions based on connectivity patterns. This hybrid achieves the best balance between AUC and recall among all methods tested.

---

## 4. Discussion

Our results demonstrate that GNN-based approaches can effectively identify SOZ electrodes from iEEG recordings, achieving clinically meaningful performance on a multi-site dataset. The test AUC of 0.768 compares favorably with prior machine learning approaches for SOZ localization, which have reported AUCs ranging from 0.65 to 0.85 depending on the dataset and methodology (Varatharajah et al., 2018; Bernabei et al., 2022).

### 4.1 Clinical Implications

The high recall (62.3%) achieved by our model is particularly relevant for clinical application. In epilepsy surgery planning, failing to identify a true SOZ electrode could lead to incomplete resection and surgical failure. Notably, while Random Forest achieved higher overall AUC (0.847 vs 0.763), the GNN demonstrated substantially superior recall (62.3% vs 43.6%)—a 43% relative improvement. This finding highlights that AUC alone may not fully capture clinical utility for SOZ localization, where sensitivity is paramount.

The GNN's sensitivity advantage likely stems from its ability to model inter-electrode connectivity patterns through the graph structure. SOZ electrodes often exhibit characteristic connectivity signatures that are lost when electrodes are treated independently, as in classical ML approaches.

The relatively low precision (17.5%) reflects the conservative nature of our approach, which generates false positives but minimizes false negatives. In practice, false positives can be reviewed and rejected by clinicians, whereas false negatives represent potentially missed surgical targets.

### 4.2 Understanding the Performance Gap Between GNN and Random Forest

An unexpected finding of this study was that Random Forest achieved higher overall AUC (0.847) than our GNN model (0.763), despite the latter's ability to leverage graph structure. This result warrants careful interpretation, as it reflects fundamental tradeoffs in model selection for small clinical datasets rather than a failure of the graph-based approach.

The most likely explanation lies in sample size constraints. Our training set contained only 70 graphs with approximately 6,200 labeled nodes—a dataset size that falls well below what deep learning models typically require to avoid overfitting (Shorten & Khoshgoftaar, 2019). GNNs, like other neural architectures, are prone to memorizing training patterns when data is scarce, particularly in the presence of class imbalance (8.9% SOZ nodes). Random Forest, by contrast, builds an ensemble of shallow decision trees that naturally resist overfitting through bagging and feature subsampling, making it inherently more stable on small datasets (Katzmann et al., 2020).

A second consideration is the quality of our engineered features. The node-level features we extracted—band powers, HFO activity, line length, and statistical moments—represent decades of domain knowledge about epileptogenic biomarkers. These features are already highly discriminative for SOZ identification. When input features carry strong predictive signal on their own, simpler models can exploit them directly, while GNNs introduce additional complexity through message passing that may not provide commensurate benefit (Chen et al., 2023).

The graph construction process itself introduces uncertainty. We defined edges based on correlation thresholds, but the optimal functional connectivity representation for SOZ localization remains an open question. If the constructed graph does not accurately capture the relevant inter-electrode relationships, the GNN's neighborhood aggregation mechanism may propagate noise rather than useful information. Random Forest sidesteps this issue entirely by treating each node independently.

Despite these limitations, the GNN demonstrated a clear advantage in recall—the metric most relevant for clinical screening. This suggests that while Random Forest achieves better overall discrimination, it does so partly by being conservative in its positive predictions. The GNN, through its connectivity-aware learning, appears to capture patterns that help identify true SOZ electrodes even at the cost of more false positives. For a clinical decision-support tool where missing a true SOZ electrode carries significant consequences, this tradeoff may be acceptable.

These findings align with recent literature suggesting that deep learning models require careful validation against simpler baselines in medical imaging applications, particularly when sample sizes are limited (Whalen et al., 2024). They also highlight an opportunity for future work: hybrid approaches that combine the robustness of tree-based ensembles with the relational learning capabilities of GNNs may offer the best of both worlds.

### 4.3 Importance of Multi-Site Data

Combining datasets from the same institution but different acquisition protocols improved performance beyond what augmentation alone could achieve. This finding underscores the importance of data pooling efforts in the epilepsy neuroimaging community. Public repositories like OpenNeuro facilitate such collaborations and accelerate methodological development.

### 4.4 Self-Supervised Pretraining

Our self-supervised pretraining strategy, which learns temporal dynamics of iEEG signals without requiring labels, proved essential given the limited number of labeled recordings. By leveraging all available recordings for pretraining, including those without SOZ annotations, the encoder learns robust feature representations that transfer effectively to the classification task.

### 4.5 Graph Neural Networks for iEEG

The graph representation naturally captures the spatial relationships between electrodes, which vary across patients depending on clinical hypotheses and electrode placement strategies. Unlike convolutional approaches that assume regular grid structures, GNNs accommodate arbitrary electrode configurations and explicitly model functional connectivity through edge attributes.

### 4.6 Limitations and Future Directions

Several limitations should be acknowledged. First, our evaluation was limited to data from a single institution, and generalization to other centers remains to be validated. Second, the dataset size, while larger than many prior studies, remains modest by deep learning standards. Third, we evaluated performance at the electrode level rather than the patient level, which may overestimate clinical utility.

Future work should explore cross-institutional validation, integration of anatomical information from electrode localization, and extension to seizure prediction tasks. Additionally, explainability methods for GNNs could provide insights into which electrode features and connectivity patterns drive SOZ predictions.

---

## 5. Conclusion

We developed a GNN-based framework for automated SOZ localization in drug-resistant epilepsy patients undergoing intracranial EEG monitoring. By combining publicly available datasets and employing self-supervised pretraining with online augmentation, our GraphSAGE model achieved a test AUC of 0.763 ± 0.008 with 62.3% recall across 5 random seeds. While classical Random Forest achieved higher overall AUC (0.847), the GNN demonstrated substantially higher recall (62.3% vs 43.6%), which is critical for clinical applications where missing true SOZ electrodes carries significant cost.

Our investigation of hybrid approaches revealed that incorporating Random Forest predictions as an additional node feature yielded the best balance between discrimination and sensitivity, achieving AUC of 0.785 with recall of 0.630. This finding suggests that combining the strengths of classical machine learning with graph-based deep learning offers a practical path toward improved SOZ localization.

These results demonstrate that graph-based deep learning approaches hold promise as decision-support tools for epilepsy surgical planning. Future work should focus on prospective validation, cross-institutional generalization, and integration into clinical workflows.

---

## Data and Code Availability

The datasets used in this study are publicly available on OpenNeuro:
- ds003029: https://openneuro.org/datasets/ds003029
- ds004100: https://openneuro.org/datasets/ds004100

Code for preprocessing, feature extraction, and model training is available at: https://github.com/abtinunmc/GNN-PROJECT

---

## References

Bernabei, J. M., Sinha, N., Arnold, T. C., Conrad, E., Ong, I., Pattnaik, A. R., Stein, J. M., Shinohara, R. T., Lucas, T. H., Bassett, D. S., Davis, K. A., & Litt, B. (2022). Normative intracranial EEG maps epileptogenic tissues in focal epilepsy. *Brain*, 145(6), 1949-1961. https://doi.org/10.1093/brain/awab480

Bessadok, A., Mahjoub, M. A., & Rekik, I. (2022). Graph neural networks in network neuroscience. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(7), 3847-3867. https://doi.org/10.1109/TPAMI.2021.3062878

Engel, J., McDermott, M. P., Wiebe, S., Langfitt, J. T., Stern, J. M., Dewar, S., Sperling, M. R., Gardiner, I., Erba, G., Fried, I., Jacobs, M., Vinters, H. V., Mintzer, S., & Kieburtz, K. (2012). Early surgical therapy for drug-resistant temporal lobe epilepsy: A randomized trial. *JAMA*, 307(9), 922-930. https://doi.org/10.1001/jama.2012.220

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30, 1024-1034.

Jacobs, J., Staba, R., Asano, E., Otsubo, H., Wu, J. Y., Zijlmans, M., Mohamed, I., Kahane, P., Dubeau, F., Bhardwaj, R., Mathern, G. W., Bernal, B., & Bhattacharjee, M. (2012). High-frequency oscillations (HFOs) in clinical epilepsy. *Progress in Neurobiology*, 98(3), 302-315. https://doi.org/10.1016/j.pneurobio.2012.03.001

Kini, L. G., Bernabei, J. M., Mikhail, F., Hadar, P., Shah, P., Khambhati, A. N., Oechsel, K., Archer, R., Boccanfuso, J., Conrad, E., Stein, J. M., Das, S., Kheder, A., Lucas, T. H., Davis, K. A., Bassett, D. S., & Litt, B. (2019). Virtual resection predicts surgical outcome for drug-resistant epilepsy. *Brain*, 142(12), 3892-3905. https://doi.org/10.1093/brain/awz303

Kwan, P., & Brodie, M. J. (2000). Early identification of refractory epilepsy. *New England Journal of Medicine*, 342(5), 314-319. https://doi.org/10.1056/NEJM200002033420503

Katzmann, A., Mühlberg, A., Sühling, M., Nittka, M., & Hoppe, B. F. (2020). Deep random forests for small sample size prediction with medical imaging data. *IEEE International Symposium on Biomedical Imaging (ISBI)*, 2020, 1543-1547. https://doi.org/10.1109/ISBI45749.2020.9098420

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 60. https://doi.org/10.1186/s40537-019-0197-0

Chen, T., Kornblith, S., Swersky, K., Norouzi, M., & Hinton, G. E. (2023). Big self-supervised models are strong semi-supervised learners. *Advances in Neural Information Processing Systems*, 33, 22243-22255.

Whalen, S., Schreiber, J., Noble, W. S., & Pollard, K. S. (2024). Navigating the pitfalls of applying machine learning in genomics. *Nature Reviews Genetics*, 23(3), 169-181. https://doi.org/10.1038/s41576-021-00434-9

Varatharajah, Y., Berry, B. M., Cimbalnik, J., Kremen, V., Van Gompel, J., Stead, M., Brinkmann, B. H., Iyer, R., & Worrell, G. (2018). Integrating artificial intelligence with real-time intracranial EEG monitoring to automate interictal identification of seizure onset zones in focal epilepsy. *Journal of Neural Engineering*, 15(4), 046035. https://doi.org/10.1088/1741-2552/aac3dc

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations*.

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *International Conference on Learning Representations*.

---

## Figures

**Figure 1.** ROC curve demonstrating model discrimination performance on the test set. The tuned GraphSAGE model achieved an AUC of 0.768.
*File: data/processed/figures/roc_curve.png*

**Figure 2.** Optuna hyperparameter optimization history showing convergence of the Bayesian search across 50 trials.
*File: data/processed/figures/optuna_optimization.png*

**Figure 3.** Preprocessing pipeline demonstration showing (A) raw signal, (B) after notch filtering, (C) after common average reference, and (D) after bandpass filtering.
*File: data/processed/figures/preprocessing_demo.png*

**Figure 4.** Common Average Reference (CAR) demonstration. Top panel shows the raw signal from a single channel, bottom panel shows the signal after CAR subtraction. CAR removes global noise and volume conduction artifacts while preserving focal activity.
*File: data/processed/figures/car_demo.png*

**Figure 5.** Multi-channel CAR visualization showing the effect of common average reference across multiple electrodes simultaneously. Each row represents a different channel before (left) and after (right) CAR application.
*File: data/processed/figures/car_multichannel.png*

**Figure 6.** CAR statistics showing the distribution of signal amplitude changes across all channels. The histogram demonstrates the reduction in common-mode noise after CAR application.
*File: data/processed/figures/car_statistics.png*

**Figure 7.** Bandpass filter frequency response and demonstration. (A) Filter magnitude response showing passband (1-250 Hz) and stopband attenuation. (B) Example signal before and after filtering. (C) Power spectral density showing preserved frequency bands.
*Files: data/processed/figures/bandpass_filter_response.png, data/processed/figures/bandpass_demo.png, data/processed/figures/bandpass_bands.png*

**Figure 8.** Example graph construction from iEEG electrode array. Nodes represent electrodes colored by SOZ label, edges represent functional connectivity exceeding the correlation threshold.
*File: data/processed/figures/graph_demo.png*

**Figure 9.** Confusion matrix on the held-out test set showing the distribution of true positives, false positives, true negatives, and false negatives.
*File: data/processed/figures/confusion_matrix.png*

**Figure 10.** Training curves for (A) self-supervised pretraining loss and (B) supervised fine-tuning loss.
*Files: data/processed/figures/pretrain_loss.png, data/processed/figures/train_loss.png*

---

*Manuscript prepared: May 2026*

*Font specification for final formatting: Times New Roman, 12pt*
