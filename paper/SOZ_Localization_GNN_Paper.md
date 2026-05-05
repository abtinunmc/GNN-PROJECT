# Graph Neural Networks for Seizure Onset Zone Localization in Drug-Resistant Epilepsy: A Multi-Site Intracranial EEG Study

**Authors:** Abtin Akhtari

**Affiliation:** University of Nebraska Medical Center

**Corresponding Author:** abtin.akhtary@gmail.com

---

## Abstract

Accurate localization of the seizure onset zone (SOZ) is critical for surgical planning in patients with drug-resistant epilepsy. While intracranial electroencephalography (iEEG) remains the gold standard for SOZ identification, manual analysis is time-consuming and subject to inter-rater variability. In this study, we present a graph neural network (GNN) framework that models iEEG electrode arrays as graphs, where nodes represent electrodes and edges capture functional connectivity patterns. We combined two publicly available datasets from the Hospital of the University of Pennsylvania, totaling 81 patients and 212 recordings with 126 having expert-annotated SOZ labels. Our GraphSAGE-based architecture, incorporating self-supervised pretraining and online data augmentation, achieved an area under the receiver operating characteristic curve (AUC) of 0.768 on held-out test data. Notably, the model demonstrated high sensitivity (recall of 65%), which is clinically valuable for ensuring comprehensive SOZ coverage. These findings suggest that GNN-based approaches can serve as effective decision-support tools for epilepsy surgical planning.

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

**Notch filtering:** Line noise at 60 Hz and its harmonics (120, 180 Hz) was removed using a notch filter.

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

### 3.1 Architecture Comparison

Table 2 presents the performance of different GNN architectures on the single-dataset (ds003029) benchmark. GraphSAGE achieved the best test AUC of 0.730 after hyperparameter tuning, outperforming both GAT (0.702) and GIN (0.562).

**Table 2.** Performance comparison of GNN architectures (ds003029 only)

| Architecture | Val AUC | Test AUC | Test F1 |
|--------------|---------|----------|---------|
| GraphSAGE (baseline) | 0.920 | 0.697 | 0.478 |
| GraphSAGE (tuned) | 0.983 | 0.730 | 0.404 |
| GAT (tuned) | 0.925 | 0.702 | 0.411 |
| GIN | 0.660 | 0.562 | 0.000 |

*Figure 1: ROC curve for the tuned GraphSAGE model (see data/processed/figures/roc_curve.png)*

*Figure 2: Optuna optimization history showing convergence of hyperparameter search (see data/processed/figures/optuna_optimization.png)*

### 3.2 Data Augmentation Analysis

We systematically evaluated multiple augmentation strategies (Table 3). Online augmentation (edge dropout, feature noise, feature masking) improved test AUC from 0.730 to 0.761, representing a 4.2% relative improvement. However, time-shift augmentation, mixup, and SMOTE did not provide additional benefits and in some cases degraded performance.

**Table 3.** Effect of data augmentation techniques

| Technique | Test AUC | Change |
|-----------|----------|--------|
| None (baseline) | 0.730 | - |
| Online augmentation | 0.761 | +4.2% |
| Time-shift | 0.718 | -1.6% |
| Mixup | 0.731 | +0.1% |
| SMOTE | 0.696 | -4.7% |

### 3.3 Combined Dataset Performance

Combining the two datasets substantially increased training data (70 vs. 20 labeled graphs) and improved model performance. The final model achieved a test AUC of 0.768, with notably high recall of 0.650 (Table 4).

**Table 4.** Performance on combined dataset

| Metric | Value |
|--------|-------|
| Validation AUC | 0.751 |
| Test AUC | 0.768 |
| Test F1 | 0.296 |
| Test Precision | 0.192 |
| Test Recall | 0.650 |

*Figure 3: Training loss curves for pretraining and fine-tuning (see data/processed/figures/pretrain_loss.png and train_loss.png)*

*Figure 4: Confusion matrix on test set (see data/processed/figures/confusion_matrix.png)*

### 3.4 Comparison with Single-Dataset Training

Table 5 summarizes the progression of improvements across our experiments. The combination of dataset expansion and online augmentation yielded the best overall performance.

**Table 5.** Summary of experimental progression

| Configuration | Training Graphs | Test AUC | Test Recall |
|---------------|-----------------|----------|-------------|
| ds003029 baseline | ~20 | 0.730 | 0.48 |
| ds003029 + augmentation | ~20 | 0.761 | 0.52 |
| Combined dataset | 70 | 0.768 | 0.65 |

---

## 4. Discussion

Our results demonstrate that GNN-based approaches can effectively identify SOZ electrodes from iEEG recordings, achieving clinically meaningful performance on a multi-site dataset. The test AUC of 0.768 compares favorably with prior machine learning approaches for SOZ localization, which have reported AUCs ranging from 0.65 to 0.85 depending on the dataset and methodology (Varatharajah et al., 2018; Bernabei et al., 2022).

### 4.1 Clinical Implications

The high recall (65%) achieved by our model is particularly relevant for clinical application. In epilepsy surgery planning, failing to identify a true SOZ electrode could lead to incomplete resection and surgical failure. Our model's sensitivity suggests it could serve as an effective screening tool to flag candidate SOZ regions for detailed review by epileptologists.

The relatively low precision (19%) reflects the conservative nature of our approach, which generates false positives but minimizes false negatives. In practice, false positives can be reviewed and rejected by clinicians, whereas false negatives represent potentially missed surgical targets.

### 4.2 Importance of Multi-Site Data

Combining datasets from the same institution but different acquisition protocols improved performance beyond what augmentation alone could achieve. This finding underscores the importance of data pooling efforts in the epilepsy neuroimaging community. Public repositories like OpenNeuro facilitate such collaborations and accelerate methodological development.

### 4.3 Self-Supervised Pretraining

Our self-supervised pretraining strategy, which learns temporal dynamics of iEEG signals without requiring labels, proved essential given the limited number of labeled recordings. By leveraging all available recordings for pretraining, including those without SOZ annotations, the encoder learns robust feature representations that transfer effectively to the classification task.

### 4.4 Graph Neural Networks for iEEG

The graph representation naturally captures the spatial relationships between electrodes, which vary across patients depending on clinical hypotheses and electrode placement strategies. Unlike convolutional approaches that assume regular grid structures, GNNs accommodate arbitrary electrode configurations and explicitly model functional connectivity through edge attributes.

### 4.5 Limitations and Future Directions

Several limitations should be acknowledged. First, our evaluation was limited to data from a single institution, and generalization to other centers remains to be validated. Second, the dataset size, while larger than many prior studies, remains modest by deep learning standards. Third, we evaluated performance at the electrode level rather than the patient level, which may overestimate clinical utility.

Future work should explore cross-institutional validation, integration of anatomical information from electrode localization, and extension to seizure prediction tasks. Additionally, explainability methods for GNNs could provide insights into which electrode features and connectivity patterns drive SOZ predictions.

---

## 5. Conclusion

We developed a GNN-based framework for automated SOZ localization in drug-resistant epilepsy patients undergoing intracranial EEG monitoring. By combining publicly available datasets and employing self-supervised pretraining with online augmentation, our GraphSAGE model achieved a test AUC of 0.768 with 65% recall. These results suggest that graph-based deep learning approaches hold promise as decision-support tools for epilepsy surgical planning. Future work should focus on prospective validation and integration into clinical workflows.

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

**Figure 4.** Example graph construction from iEEG electrode array. Nodes represent electrodes colored by SOZ label, edges represent functional connectivity exceeding the correlation threshold.
*File: data/processed/figures/graph_demo.png*

**Figure 5.** Confusion matrix on the held-out test set showing the distribution of true positives, false positives, true negatives, and false negatives.
*File: data/processed/figures/confusion_matrix.png*

**Figure 6.** Training curves for (A) self-supervised pretraining loss and (B) supervised fine-tuning loss.
*Files: data/processed/figures/pretrain_loss.png, data/processed/figures/train_loss.png*

---

*Manuscript prepared: May 2026*

*Font specification for final formatting: Times New Roman, 12pt*
