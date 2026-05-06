# Graph Neural Networks for Seizure Onset Zone Localization in Drug-Resistant Epilepsy: A Multi-Site Intracranial EEG Study

**Running title:** GNN for SOZ Localization

---

**Abtin Akhtari**^1,*^

^1^ University of Nebraska Medical Center, Omaha, NE, USA

^\*^ **Corresponding author:** Abtin Akhtari (abtin.akhtary@gmail.com)

---

**Word count:** ~4,500 words (excluding references and figures)

**Figures:** 10 | **Tables:** 8

---

## Abstract

Finding the seizure onset zone (SOZ) accurately can make or break epilepsy surgery outcomes. Intracranial EEG remains our best tool for this job, but having clinicians manually review these recordings takes considerable time and different experts often disagree on their interpretations. We tackled this problem by building a graph neural network (GNN) that treats electrode arrays as interconnected networks—each electrode becomes a node, and the functional connections between them form the edges. We pooled data from two publicly available datasets out of the Hospital of the University of Pennsylvania: 81 patients total, 212 recordings, with 126 having confirmed SOZ labels from clinical experts. Our approach uses GraphSAGE with self-supervised pretraining and data augmentation during training. The base model reached an AUC of 0.768, which jumped to 0.785 ± 0.013 once we added Random Forest predictions as an extra feature for each node. What stood out most was the sensitivity: our model caught 63% of true SOZ electrodes, compared to just 44% for Random Forest alone—even though RF scored higher on overall AUC (0.847). This matters clinically because missing an SOZ electrode during surgery planning is far worse than flagging a few extra ones for review. These results point toward GNN-based tools, especially hybrid versions that blend classical ML with graph learning, as promising aids for surgical planning in epilepsy.

**Keywords:** seizure onset zone, graph neural networks, intracranial EEG, epilepsy surgery, deep learning

---

## 1. Introduction

Around 50 million people live with epilepsy globally. For roughly a third of them, medications simply don't control the seizures (Kwan & Brodie, 2000). Surgery offers these patients their best shot at becoming seizure-free—success rates hover between 50% and 80%, depending on what's causing the epilepsy (Engel et al., 2012). But here's the catch: surgery only works if we can pinpoint exactly where seizures start. Get the SOZ wrong, and the operation fails.

Intracranial EEG, whether through depth electrodes (SEEG) or surface grids (ECoG), gives us recordings straight from brain tissue with excellent spatial and temporal detail. Epileptologists spend hours poring over these signals, looking for the telltale patterns of seizure onset. It's painstaking work, and frankly, subjective. Ask three experts to mark the SOZ and you might get three different answers (Varatharajah et al., 2018). We clearly need more objective, automated approaches.

Deep learning has made inroads into neurophysiological signal analysis over the past few years. Standard convolutional and recurrent networks, though, weren't really designed for electrode arrays. These arrays have irregular spatial layouts that change from patient to patient based on where clinicians think the seizures might originate. Graph neural networks (GNNs) handle this naturally—they represent electrodes as nodes and encode the functional relationships between recording sites as edges (Bessadok et al., 2022). It's a framework that fits the problem well.

In this paper, we describe a GNN pipeline for SOZ localization that we built and tested on 81 patients from two open-access iEEG repositories. We make five main contributions: first, a preprocessing pipeline that works with multiple data formats; second, a self-supervised pretraining scheme that makes use of unlabeled recordings; third, a thorough comparison of different augmentation strategies; fourth, head-to-head testing of several GNN architectures; and fifth, an exploration of hybrid methods that pair traditional machine learning with graph-based deep learning.

---

## 2. Materials and Methods

### 2.1 Datasets

We worked with two datasets from the Hospital of the University of Pennsylvania (HUP), both freely available on OpenNeuro:

**Dataset 1 (ds003029):** Contains recordings from 23 drug-resistant epilepsy patients who went through presurgical workups. The files come in BrainVision format, and the SOZ annotations are tucked into the BIDS events.tsv files as event markers.

**Dataset 2 (ds004100):** A larger collection with 58 patients who had either SEEG or ECoG monitoring. These recordings use European Data Format (EDF), and the SOZ labels appear directly in the channels.tsv metadata. All these patients eventually had surgery or laser ablation, with Engel outcome scores on record.

Altogether, we had 212 recordings to work with. Of these, 126 came with expert-annotated SOZ labels (Table 1). The rest we used for pretraining.

**Table 1.** Dataset characteristics

| Characteristic | ds003029 | ds004100 | Combined |
|----------------|----------|----------|----------|
| Patients | 23 | 58 | 81 |
| Recordings | 103 | 109 | 212 |
| Recordings with SOZ | 30 | 96 | 126 |
| Electrode type | SEEG/ECoG | SEEG/ECoG | Mixed |
| Sampling rate | 500-2000 Hz | 500 Hz | Variable |

### 2.2 Preprocessing Pipeline

We built our preprocessing to handle both BrainVision and EDF files. Four steps:

**Extracting epochs:** We found the seizure onset timestamp in each recording and pulled out 60 seconds of data centered on that moment—30 seconds before onset, 30 seconds after. Any channels flagged as bad in the metadata got tossed.

**Notch filtering:** Power line noise at 60 Hz and its harmonics (120, 180, 240 Hz) had to go. We included the fourth harmonic because 240 Hz sits close to our bandpass cutoff and could sneak through otherwise.

**Common average referencing:** Volume conduction and common-mode noise muddy up iEEG signals. To clean this up, we subtracted the mean voltage across all electrodes at each time point.

**Bandpass filtering:** We applied an elliptic filter with corners at 1 Hz and 250 Hz. This keeps both the slow rhythms and the high-frequency oscillations (HFOs) that researchers have linked to epileptogenic tissue (Jacobs et al., 2012).

### 2.3 Feature Extraction

After preprocessing, we chopped each recording into 12-second windows with no overlap. From each window, we pulled out features for every electrode:

**Spectral features (6 total):** Power in the delta (1-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (15-25 Hz), low gamma (35-50 Hz), and HFO (80-150 Hz) bands. We used Welch's method for the power estimates.

**Statistical features (4 total):** Variance, line length (summing up the absolute differences between consecutive samples), kurtosis, and skewness.

**Relative HFO power (1):** The ratio of HFO power to total power. Elevated HFO activity tends to show up in epileptogenic regions.

We then aggregated across windows using the mean, standard deviation, and maximum—giving us 33 features per electrode.

**Edge features:** To capture functional connectivity, we computed Pearson correlations between electrode pairs within each window, then aggregated those the same way.

### 2.4 Graph Construction

Each recording became a graph. Electrodes are nodes, and we drew edges between any pair whose correlation topped 0.32 (a threshold we landed on through hyperparameter tuning). The 33 aggregated features went into each node, while edge weights encoded connectivity strength.

For labels, any electrode that clinicians had marked as SOZ got a positive label; the rest were negative. The class balance was pretty skewed—only about 8.9% of electrodes fell in the SOZ category across our combined dataset.

### 2.5 Model Architecture

We went with GraphSAGE (Hamilton et al., 2017), which builds node representations by pulling in information from neighboring nodes. Our encoder stacks two GraphSAGE layers with batch normalization, 128 hidden dimensions, and 21% dropout.

On top of that sits a classification head: a small MLP with 64 hidden units, 55% dropout, and a sigmoid output for the binary SOZ prediction.

### 2.6 Training Strategy

**Pretraining without labels:** Since we had plenty of unlabeled recordings, we pretrained the encoder on a forecasting task—predicting the features at time window t+1 from window t. This ran on 827 window pairs drawn from all available data.

**Fine-tuning with labels:** We then trained the full model for SOZ classification using weighted binary cross-entropy. The weights were set inversely to class frequencies to help with the imbalance problem.

**Augmentation on the fly:** During training, we randomly dropped 7.6% of edges, added Gaussian noise (σ = 0.011) to features, and masked out 10% of feature values.

**Optimization details:** AdamW optimizer, learning rates of 1.2×10⁻⁴ for pretraining and 6.1×10⁻⁴ for fine-tuning, cosine annealing schedule, and early stopping based on validation AUC.

### 2.7 Experimental Setup

We split the data by patient—no patient appeared in more than one split—with 60% for training, 20% for validation, and 20% for testing. That worked out to 70 training graphs, 36 validation graphs, and 20 test graphs.

We tried three GNN flavors: GraphSAGE, Graph Attention Network (GAT; Velickovic et al., 2018), and Graph Isomorphism Network (GIN; Xu et al., 2019). Optuna handled the hyperparameter search, running 50 trials for each architecture.

We also tested several augmentation approaches: the online augmentation described above, time-shift augmentation (randomly subsampling windows), mixup, and SMOTE oversampling.

### 2.8 Evaluation Metrics

We tracked AUC, F1 score, precision, and recall. Recall mattered most to us—in this clinical setting, missing a true SOZ electrode has serious consequences.

---

## 3. Results

### 3.1 Comparison with Classical Machine Learning

To put our GNN results in context, we trained several standard ML models on the same node features, just without the graph structure. Table 2 shows what happened.

**Table 2.** Comparison with classical ML baselines (combined dataset)

| Method | Test AUC | Test F1 | Test Precision | Test Recall |
|--------|----------|---------|----------------|-------------|
| Random Forest | 0.847 | 0.373 | 0.326 | 0.436 |
| Logistic Regression | 0.741 | 0.336 | 0.238 | 0.571 |
| Lasso (L1) | 0.738 | 0.338 | 0.240 | 0.571 |
| Ridge (L2) | 0.741 | 0.336 | 0.238 | 0.571 |
| LR (tuned, CV) | 0.742 | 0.340 | 0.242 | 0.571 |
| SVM (RBF kernel) | 0.713 | 0.097 | 0.200 | 0.064 |
| **GraphSAGE (ours)** | **0.763 ± 0.008** | **0.273 ± 0.007** | **0.175 ± 0.006** | **0.623 ± 0.014** |

*Note: GNN numbers are means ± standard deviations over 5 seeds [42, 123, 456, 789, 2024]. Classical models used balanced class weights.*

Here's what surprised us: Random Forest actually beat the GNN on AUC, hitting 0.847 versus our 0.763. But look at the recall numbers. The GNN caught 62.3% of SOZ electrodes while Random Forest only managed 43.6%—that's a 43% relative improvement. When the goal is to avoid missing any true SOZ contacts, that difference matters enormously. A few extra false positives that clinicians can dismiss beats missing electrodes that should have been resected.

### 3.2 GNN Architecture Comparison

We benchmarked different GNN architectures on just the ds003029 dataset first (Table 3). After tuning, GraphSAGE came out on top with a test AUC of 0.730, beating GAT (0.702) and GIN (0.562) by a fair margin.

**Table 3.** Performance comparison of GNN architectures (ds003029 only)

| Architecture | Val AUC | Test AUC | Test F1 |
|--------------|---------|----------|---------|
| GraphSAGE (baseline) | 0.920 | 0.697 | 0.478 |
| GraphSAGE (tuned) | 0.983 | 0.730 | 0.404 |
| GAT (tuned) | 0.925 | 0.702 | 0.411 |
| GIN | 0.660 | 0.562 | 0.000 |

*See Figure 1 for ROC curve and Figure 2 for Optuna optimization history.*

### 3.3 Data Augmentation Analysis

Not all augmentation helps. Table 4 breaks down what we found. Online augmentation—dropping edges, adding noise, masking features—pushed the test AUC from 0.730 up to 0.761, a solid 4.2% gain. Time-shift augmentation, mixup, and SMOTE either did nothing or actively hurt performance.

**Table 4.** Effect of data augmentation techniques

| Technique | Test AUC | Change |
|-----------|----------|--------|
| None (baseline) | 0.730 | - |
| Online augmentation | 0.761 | +4.2% |
| Time-shift | 0.718 | -1.6% |
| Mixup | 0.731 | +0.1% |
| SMOTE | 0.696 | -4.7% |

### 3.4 Combined Dataset Performance

Pooling both datasets gave us more training data—70 labeled graphs instead of roughly 20—and the numbers improved. A single training run hit 0.768 AUC with 65% recall. Across five random seeds, we averaged 0.763 ± 0.008 AUC and 62.3% ± 1.4% recall (Table 5).

**Table 5.** Performance on combined dataset

| Metric | Single Run | 5 Seeds (mean ± std) |
|--------|------------|----------------------|
| Validation AUC | 0.751 | - |
| Test AUC | 0.768 | 0.763 ± 0.008 |
| Test F1 | 0.296 | 0.273 ± 0.007 |
| Test Precision | 0.192 | 0.175 ± 0.006 |
| Test Recall | 0.650 | 0.623 ± 0.014 |

**Table 5a.** GNN per-seed results on combined dataset

| Seed | Test AUC | Test F1 | Test Recall |
|------|----------|---------|-------------|
| 42 | 0.753 | 0.260 | 0.621 |
| 123 | 0.776 | 0.278 | 0.643 |
| 456 | 0.762 | 0.281 | 0.629 |
| 789 | 0.756 | 0.270 | 0.600 |
| 2024 | 0.768 | 0.274 | 0.621 |

*See Figure 9 for confusion matrix and Figure 10 for training curves.*

### 3.5 Comparison with Single-Dataset Training

Table 6 traces our progress through the experiments. More data helped. Augmentation helped. But the biggest jump came from the hybrid approach.

**Table 6.** Summary of experimental progression

| Configuration | Training Graphs | Test AUC | Test Recall |
|---------------|-----------------|----------|-------------|
| ds003029 baseline | ~20 | 0.697 | 0.47 |
| ds003029 tuned | ~20 | 0.730 | 0.40 |
| ds003029 + augmentation | ~20 | 0.761 | 0.52 |
| Combined dataset (single) | 70 | 0.768 | 0.650 |
| Combined dataset (5 seeds) | 70 | 0.763 ± 0.008 | 0.623 ± 0.014 |
| **Combined + RF feature** | 70 | **0.785 ± 0.013** | **0.630 ± 0.052** |

### 3.6 Hybrid Approaches to Improve AUC

Random Forest's AUC lead (0.847 vs. 0.763) bugged us, so we tried a few ways to close the gap without giving up the GNN's recall advantage.

**Does graph structure actually help?** We trained an MLP with the same architecture but no message passing—just treating each electrode independently. It scored 0.742 ± 0.009 AUC, compared to 0.763 ± 0.008 for the GNN. So yes, the graph adds about 2.8% on top.

**What about different graph sparsity?** We tried correlation thresholds of 0.4, 0.5, and 0.6 to make sparser graphs. All of them performed worse (AUC between 0.72 and 0.75). The denser 0.32 threshold seems to work better—maybe more connectivity gives the model more to work with.

**Stacking the models?** Combining RF and GNN predictions through a meta-learner got us to 0.837 AUC, almost matching RF alone. But recall dropped to 40%. The ensemble basically learned to trust RF's conservative predictions.

**RF predictions as a feature?** This worked best. We ran Random Forest first, then fed its probability outputs as an extra node feature into the GNN. The result: 0.785 ± 0.013 AUC with 63% ± 5% recall. We got most of the AUC boost (2.9% improvement) while keeping the GNN's sensitivity edge (Table 7).

**Table 7.** Comparison of hybrid approaches

| Method | Test AUC | Test F1 | Test Precision | Test Recall |
|--------|----------|---------|----------------|-------------|
| Random Forest | 0.847 | 0.373 | 0.326 | 0.436 |
| GNN (baseline) | 0.763 ± 0.008 | 0.273 ± 0.007 | 0.175 ± 0.006 | 0.623 ± 0.014 |
| MLP (no graph) | 0.742 ± 0.009 | 0.286 ± 0.007 | 0.192 ± 0.007 | 0.566 ± 0.015 |
| Stacking (RF + GNN) | 0.837 | 0.336 | 0.290 | 0.400 |
| Stacking (full features) | 0.829 | 0.318 | 0.292 | 0.350 |
| **GNN + RF feature** | **0.785 ± 0.013** | **0.293 ± 0.012** | **0.192 ± 0.015** | **0.630 ± 0.052** |

**Table 7a.** Effect of correlation threshold on GNN performance

| Threshold | Test AUC | Test Recall |
|-----------|----------|-------------|
| 0.32 (baseline) | 0.763 ± 0.008 | 0.623 ± 0.014 |
| 0.40 | 0.747 ± 0.009 | 0.527 ± 0.158 |
| 0.50 | 0.729 ± 0.012 | 0.530 ± 0.148 |
| 0.60 | 0.717 ± 0.009 | 0.569 ± 0.023 |

The RF-feature hybrid strikes a nice balance. It taps into Random Forest's strong per-node discrimination while letting the GNN refine things based on how electrodes connect to each other.

---

## 4. Discussion

Our GNN pipeline can identify SOZ electrodes from iEEG with clinically useful accuracy. The 0.768 AUC we achieved sits comfortably within the 0.65–0.85 range that prior ML studies have reported for this task (Varatharajah et al., 2018; Bernabei et al., 2022).

### 4.1 Clinical Implications

What jumps out is the 62.3% recall. In surgical planning, you really don't want to miss SOZ electrodes—incomplete resection means the seizures come back. Random Forest beat us on overall discrimination (0.847 vs. 0.763 AUC), but it only caught 43.6% of the true SOZ contacts. That 43% relative improvement in sensitivity could translate to better surgical outcomes.

Why does the GNN do better at finding true positives? We think it comes down to connectivity. SOZ electrodes tend to have distinctive patterns in how they link to their neighbors—patterns that get lost when you analyze each electrode in isolation, the way Random Forest does.

The flip side is low precision (17.5%). Our model throws up a lot of false alarms. But in practice, that's manageable. Clinicians can review flagged electrodes and discard the false positives. What they can't do is go back and find electrodes the model missed entirely.

### 4.2 Understanding the Performance Gap Between GNN and Random Forest

We honestly didn't expect Random Forest to beat the GNN on AUC. Graphs are supposed to help, right? But after digging in, we think a few factors explain it.

First, sample size. We trained on just 70 graphs with about 6,200 labeled nodes total. That's not much by deep learning standards. Neural networks, GNNs included, tend to memorize training patterns when data is tight (Shorten & Khoshgoftaar, 2019). Add in the class imbalance (only 8.9% SOZ nodes) and overfitting becomes a real risk. Random Forest handles small datasets more gracefully—its bagging and feature subsampling naturally resist overfitting (Katzmann et al., 2020).

Second, our features are already pretty good. Band powers, HFO activity, line length, statistical moments—these biomarkers encode decades of clinical knowledge about what makes tissue epileptogenic. When the input features carry that much signal, a simple model can exploit them directly. The extra complexity of message passing may not buy you much on top of that (Chen et al., 2020).

Third, graph construction isn't a solved problem. We defined edges based on correlation thresholds, but who's to say that captures the connectivity patterns that matter for SOZ localization? If the graph structure is noisy or incomplete, the GNN might just be propagating garbage through its neighborhood aggregation. Random Forest sidesteps this by ignoring the graph entirely.

Still, the GNN's recall advantage is real and consistent. That tells us something: even if RF discriminates better overall, it does so by being conservative. The GNN, by incorporating connectivity, picks up on patterns that help identify true SOZ contacts—at the cost of more false positives. For a screening tool, that tradeoff makes sense.

These findings echo recent work showing that deep learning needs careful benchmarking against simpler baselines in medical settings, especially when data is scarce (Whalen et al., 2022). They also point to an opportunity: hybrid methods that combine RF's robustness with GNN's relational learning might give us the best of both.

### 4.3 Importance of Multi-Site Data

Pooling two datasets, even from the same institution with different protocols, helped more than any augmentation trick we tried. This underscores how much the epilepsy imaging community could gain from sharing data. Platforms like OpenNeuro make that possible and speed up methods development.

### 4.4 Self-Supervised Pretraining

With limited labeled recordings, pretraining mattered. By learning to predict the next time window from the current one, the encoder picks up useful representations of iEEG dynamics before ever seeing an SOZ label. The unlabeled recordings—which we'd otherwise have to ignore—become useful training data.

### 4.5 Graph Neural Networks for iEEG

Electrode arrays don't sit on neat grids. They're placed based on clinical hypotheses, so layouts vary wildly across patients. GNNs handle this naturally. Unlike CNNs that assume regular structure, GNNs work with arbitrary node configurations and explicitly represent inter-electrode relationships through edges.

### 4.6 Limitations and Future Directions

A few caveats. All our data came from one institution—we don't yet know how well this generalizes elsewhere. The dataset, while larger than many in this space, is still modest by deep learning standards. And we evaluated at the electrode level, not the patient level, which might paint an overly rosy picture of clinical utility.

Going forward, we'd like to test on data from other centers, bring in anatomical information from electrode localization, and extend to seizure prediction. Explainability tools for GNNs could also reveal which features and connectivity patterns drive the model's decisions.

---

## 5. Conclusion

We built a GNN framework for automated SOZ localization in patients undergoing intracranial EEG monitoring for drug-resistant epilepsy. Using two public datasets (81 patients, 126 labeled recordings), self-supervised pretraining, and online augmentation, our GraphSAGE model achieved 0.768 AUC (0.763 ± 0.008 across five seeds) with 62–65% recall. Random Forest scored higher on AUC (0.847), but the GNN caught far more true SOZ electrodes (62.3% vs. 43.6%)—a critical advantage when surgical success depends on not missing any.

Our experiments with hybrid methods turned up three key findings. First, graph structure genuinely helps—there's a 2.8% AUC gain over a comparable MLP that ignores connectivity. Second, stacking RF and GNN predictions boosts AUC but tanks recall. Third, and most promising, feeding RF's predictions as an extra node feature gives us both: **0.785 ± 0.013 AUC with 63% ± 5% recall**. This hybrid lets the GNN build on RF's per-node discrimination while still leveraging electrode connectivity.

The takeaway? Graph-based deep learning, especially when combined with classical ML, shows real promise as a decision-support tool for epilepsy surgery. The next steps are prospective validation, testing across institutions, and figuring out how to fit these models into clinical workflows.

---

## Data and Code Availability

Both datasets are publicly available on OpenNeuro:
- ds003029: https://openneuro.org/datasets/ds003029
- ds004100: https://openneuro.org/datasets/ds004100

Our preprocessing, feature extraction, and training code is at: https://github.com/abtinunmc/GNN-PROJECT

---

## References

Bernabei, J. M., Sinha, N., Arnold, T. C., Conrad, E., Ong, I., Pattnaik, A. R., Stein, J. M., Shinohara, R. T., Lucas, T. H., Bassett, D. S., Davis, K. A., & Litt, B. (2022). Normative intracranial EEG maps epileptogenic tissues in focal epilepsy. *Brain*, 145(6), 1949-1961. https://doi.org/10.1093/brain/awab480

Bessadok, A., Mahjoub, M. A., & Rekik, I. (2022). Graph neural networks in network neuroscience. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(7), 3847-3867. https://doi.org/10.1109/TPAMI.2021.3062878

Engel, J., McDermott, M. P., Wiebe, S., Langfitt, J. T., Stern, J. M., Dewar, S., Sperling, M. R., Gardiner, I., Erba, G., Fried, I., Jacobs, M., Vinters, H. V., Mintzer, S., & Kieburtz, K. (2012). Early surgical therapy for drug-resistant temporal lobe epilepsy: A randomized trial. *JAMA*, 307(9), 922-930. https://doi.org/10.1001/jama.2012.220

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30, 1024-1034.

Jacobs, J., Staba, R., Asano, E., Otsubo, H., Wu, J. Y., Zijlmans, M., Mohamed, I., Kahane, P., Dubeau, F., Bhardwaj, R., Mathern, G. W., Bernal, B., & Bhattacharjee, M. (2012). High-frequency oscillations (HFOs) in clinical epilepsy. *Progress in Neurobiology*, 98(3), 302-315. https://doi.org/10.1016/j.pneurobio.2012.03.001

Katzmann, A., Mühlberg, A., Sühling, M., Nittka, M., & Hoppe, B. F. (2020). Deep random forests for small sample size prediction with medical imaging data. *IEEE International Symposium on Biomedical Imaging (ISBI)*, 2020, 1543-1547. https://doi.org/10.1109/ISBI45749.2020.9098420

Kini, L. G., Bernabei, J. M., Mikhail, F., Hadar, P., Shah, P., Khambhati, A. N., Oechsel, K., Archer, R., Boccanfuso, J., Conrad, E., Stein, J. M., Das, S., Kheder, A., Lucas, T. H., Davis, K. A., Bassett, D. S., & Litt, B. (2019). Virtual resection predicts surgical outcome for drug-resistant epilepsy. *Brain*, 142(12), 3892-3905. https://doi.org/10.1093/brain/awz303

Kwan, P., & Brodie, M. J. (2000). Early identification of refractory epilepsy. *New England Journal of Medicine*, 342(5), 314-319. https://doi.org/10.1056/NEJM200002033420503

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 60. https://doi.org/10.1186/s40537-019-0197-0

Chen, T., Kornblith, S., Swersky, K., Norouzi, M., & Hinton, G. E. (2020). Big self-supervised models are strong semi-supervised learners. *Advances in Neural Information Processing Systems*, 33, 22243-22255.

Varatharajah, Y., Berry, B. M., Cimbalnik, J., Kremen, V., Van Gompel, J., Stead, M., Brinkmann, B. H., Iyer, R., & Worrell, G. (2018). Integrating artificial intelligence with real-time intracranial EEG monitoring to automate interictal identification of seizure onset zones in focal epilepsy. *Journal of Neural Engineering*, 15(4), 046035. https://doi.org/10.1088/1741-2552/aac3dc

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations*.

Whalen, S., Schreiber, J., Noble, W. S., & Pollard, K. S. (2022). Navigating the pitfalls of applying machine learning in genomics. *Nature Reviews Genetics*, 23(3), 169-181. https://doi.org/10.1038/s41576-021-00434-9

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *International Conference on Learning Representations*.

---

## Figures

**Figure 1.** ROC curve showing discrimination performance on the test set. The GraphSAGE model reached 0.768 AUC on the combined dataset.
*File: data/processed/figures/roc_curve.png*

**Figure 2.** Optuna optimization history across 50 Bayesian search trials.
*File: data/processed/figures/optuna_optimization.png*

**Figure 3.** Preprocessing steps illustrated: (A) raw signal, (B) after notch filtering, (C) after common average reference, (D) after bandpass filtering.
*File: data/processed/figures/preprocessing_demo.png*

**Figure 4.** Common average reference demonstration. Top: raw single-channel signal. Bottom: after CAR removes global noise while preserving focal activity.
*File: data/processed/figures/car_demo.png*

**Figure 5.** Multi-channel view of CAR effects. Each row shows a channel before (left) and after (right) referencing.
*File: data/processed/figures/car_multichannel.png*

**Figure 6.** Distribution of amplitude changes across channels after CAR, showing noise reduction.
*File: data/processed/figures/car_statistics.png*

**Figure 7.** Bandpass filter characteristics: (A) frequency response with 1-250 Hz passband, (B) example filtered signal, (C) power spectral density across frequency bands.
*Files: data/processed/figures/bandpass_filter_response.png, bandpass_demo.png, bandpass_bands.png*

**Figure 8.** Example graph from an iEEG array. Nodes (electrodes) colored by SOZ label; edges show functional connectivity above threshold.
*File: data/processed/figures/graph_demo.png*

**Figure 9.** Test set confusion matrix showing true/false positive and negative counts.
*File: data/processed/figures/confusion_matrix.png*

**Figure 10.** Training curves: (A) self-supervised pretraining loss, (B) supervised fine-tuning loss.
*Files: data/processed/figures/pretrain_loss.png, train_loss.png*

---

*Manuscript prepared: May 2026*
