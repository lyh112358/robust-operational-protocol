# Robust Operational Protocol and Observation Window Timing for Cross-Semester Academic Risk Prediction Under Capacity Constraints

## Abstract
In digital higher education administration, early academic risk prediction serves as a critical operational instrument for optimizing supportive interventions and steering instructional planning. However, existing research predominantly relies on identically distributed random cross-validation or post-hoc threshold sweeping across test cohorts once final grades are revealed. Such post-hoc paradigms fundamentally mask cross-semester temporal concept drift, rendering deployed models ineffective because operational classification thresholds cannot be determined prior to term commencement. To resolve the challenge of rigid intervention capacity budgets in real-world academic management, this paper proposes an industry-grade **Frozen Warning Ratio Protocol**: In the validation academic year, an optimal cutoff cohort size $k^*$ that maximizes the balanced discrimination metric (G-Mean = $\sqrt{\mathrm{Recall} \times \mathrm{Specificity}}$) is solved subject to an administrative safety constraint ($\mathrm{Recall} \ge 0.80$). The resulting warning ratio $p = k^*/n_{\mathrm{val}}$ is physically frozen alongside a deterministic boundary fallback mechanism. In the subsequent, strictly unseen test academic year, the system executes a single forward blind inference truncating precisely $\mathrm{round}(p \times n_{\mathrm{test}})$ students, completely eliminating future label leakage. The protocol is empirically validated across two distinct educational environments: a six-year institutional blended core course (2020–2025) and the large-scale public OULAD benchmark (2013–2014). The underlying feature engineering combines temporal behavioral dynamics with milestone academic evaluations while strictly preserving structural missingness (Structural NaNs) caused by non-compliance. Hyperparameter optimization employs a two-stage exploration-then-exploitation regime regularized by an a priori sample-variance-based candidate pruning rule.

Our extensive empirical findings uncover divergent temporal intervention dynamics across instructional modalities: In the institutional blended course, the early semester window at Stage 2 ($T=2$) exhibits decisive operational screening utility (test G-Mean 0.7213–0.7443), successfully capturing over 70% of eventual failing students within an approximate 31% intervention quota, thereby securing the most actionable lead time for remedial support. The middle stages ($T=3,4$) encounter an operational plateau driven by submission delays and cramming noise, which induces an over-compression trap at Stage 4 that severely impairs test recall. A qualitative breakthrough is deferred until the final sprint stage ($T=5$), where high-signal mock comprehensive examinations elevate the test G-Mean to an average of 0.8138 (peaking at 0.8284). Conversely, open online courses in OULAD establish an early baseline at Stage 2 and undergo rapid informational saturation at Stage 3. System-wide benchmarking demonstrates that tuned tree ensembles consistently outperform single-feature heuristics, logistic regression, TabPFN, and LSTM during the early window. Furthermore, an unsearched default Random Forest baseline exhibits an identical jump at Stage 5, confirming that this late-stage breakthrough is driven by the physical release of instructional information rather than search artifacts. Back-testing across conflicting candidate pools shows that our small-sample Top-5 regularization achieves a 75% out-of-sample win rate, effectively insulating models against validation overtuning. Sensitivity and cross-course evaluations demonstrate that a single campus-wide quota policy matches fine-grained per-course tuning. This study bridges the gap between offline algorithmic benchmarking and deployment-ready, capacity-constrained operational engineering.

**Keywords:** Educational Data Mining, Academic Early Warning, Frozen Warning Ratio, Cross-Temporal Generalization, Operational Research, Intervention Timing, Structural Missingness

---

## 1 Introduction

In modern higher education administration, identifying students at risk of course failure prior to summative assessments is essential for timely pedagogical intervention and targeted resource allocation [2,3]. However, a substantial methodological disconnect persists between academic literature and operational deployment. In real-world educational governance, early warning systems are governed by two non-negotiable operational constraints: first, **cross-temporal forward generalization**, requiring predictive pipelines to execute strictly blind, single-pass inference on future cohorts; and second, **rigid capacity limitations**, wherein advisory staff, counseling hours, and remedial recitation seats are bounded by pre-allocated institutional budgets that must be locked before the academic term begins.

Prevailing studies commonly evaluate predictive efficacy by sweeping classification thresholds post-hoc across the receiver operating characteristic (ROC) or precision-recall curves of the test set to report optimal F1-scores or G-Mean values [1,10]. This practice introduces severe data leakage by conditioning decision boundaries on the unknown global posterior label distribution of the test cohort. Because the actual failure rate and score distribution of future cohorts cannot be observed prior to term completion, such post-hoc thresholds are operationally uncomputable at term start. Furthermore, conventional reliance on identically distributed (i.i.d.) random cross-validation ignores cross-semester concept drift [13], creating a false sense of security that results in deployment failure when models encounter uncalibrated real-world cohorts.

To formalize these implementation hurdles, we synthesize **three practical challenges (Industrial Challenges)** that impede educational early warning systems:
- **C1: The Capacity-Threshold Paradox**: Academic administrators require an explicit, pre-allocated headcount quota, whereas conventional machine learning pipelines output continuous uncalibrated risk probabilities whose optimal classification thresholds depend on unobservable future label distributions.
- **C2: The Validation Overtuning Dilemma**: In small-sample validation cohorts, aggressively optimizing empirical performance over vast hyperparameter spaces overfits idiosyncratic annual variance, leading to catastrophic generalization collapse in subsequent academic years.
- **C3: The Intervention Timing Trade-off**: Early interventions (e.g., initial course weeks) offer the greatest pedagogical malleability and remedial runway but rely on sparse behavioral footprints; late-stage predictions benefit from cumulative assessment certainty but leave insufficient time for meaningful instructional remediation.

To overcome these challenges, this paper introduces the **Frozen Warning Ratio Protocol**. Operating on a historical validation academic year, the protocol solves for the exact cutoff ranking $k^*$ that maximizes the geometric mean of recall and specificity (G-Mean) under a mandatory administrative safety constraint ($\mathrm{Recall} \ge 0.80$). The resulting warning ratio $p = k^*/n_{\mathrm{val}}$ is physically frozen alongside a deterministic boundary fallback rule. When deployed on an unseen test cohort of size $n_{\mathrm{test}}$, the system unconditionally flags the top $\mathrm{round}(p \times n_{\mathrm{test}})$ ranked students in a strictly single-pass forward manner. Evaluated across 6 discrete instructional stages with observation windows $T \in \{2, 3, 4, 5\}$, our framework demonstrates practical operational robustness across both institutional blended and open web-based learning environments.

The primary contributions of this work are threefold:
1. **Reformulating the Cross-Temporal Evaluation Paradigm**: We formalize an operational evaluation protocol that decouples model risk scoring from administrative quota allocation, establishing a leakage-free benchmark equipped with deterministic boundary fallbacks for capacity-constrained educational environments.
2. **Uncovering Early-Window Utility and Temporal Dynamics**: We empirically demonstrate that Stage 2 ($T=2$) offers decisive early screening utility in physical blended courses (test G-Mean 0.7213–0.7443), capturing over 70% of at-risk students within an approximate 31% quota. Furthermore, we uncover the educational mechanisms driving the mid-term plateau, formulate the over-compression trap, and explain the late-stage qualitative breakthrough triggered by mock examinations.
3. **Validating Overtuning Regularization and Operational Parsimony**: Grounded in sample-variance theory, we prove that an a priori Top-5 candidate pruning rule provides an effective inductive bias against validation overtuning. Through comprehensive benchmarking and multi-course evaluations, we demonstrate that tuned tree ensembles dominate alternative architectures in the early window, while a unified institutional quota matches fine-grained per-course optimization, executing within seconds on standard hardware.

---

## 2 Related Work and Industrial Considerations

### 2.1 Academic Risk Prediction and Cross-Temporal Evaluation Bias
Predicting student academic failure is a foundational topic in educational data mining and learning analytics [2,3,6]. Early approaches concentrated on demographic attributes and static historical backgrounds [8,12], whereas recent literature leverages fine-grained clickstream logs collected by learning management systems (LMS) [7,9,11]. For example, the Open University Learning Analytics Dataset (OULAD) introduced by Kuzilek et al. [7] accelerated research into time-series behavioral modeling, while Adnan et al. [10] investigated early risk identification across standardized course milestones.

Nevertheless, an extensive evaluation bias persists throughout the literature. In a systematic review of 161 machine learning applications for online student risk prediction between 2014 and 2024, Shi et al. [1] revealed that the vast majority of studies relied entirely on internal k-fold cross-validation; **only 9 out of the 161 studies (5.6%) conducted external temporal validation** across separate cohorts. Bond et al. [5] similarly highlighted a systemic deficit in reporting ecological validity and external generalizability across artificial intelligence applications in higher education. As emphasized by Gašević et al. [8] and Schwerter et al. [13], instructional contexts, course designs, and baseline failure rates vary widely across semesters. Shuffling multi-year student logs into i.i.d. splits yields overly optimistic estimates that fail under actual temporal drift. Our protocol enforces strict forward-chaining temporal evaluation across distinct academic years.

### 2.2 Class Imbalance and the Engineering Failure of Post-Hoc Tuning
Academic failure is characterized by severe class imbalance [6,15]. In university-level compulsory courses, failure rates rarely exceed 10% (e.g., 3.85% in our institutional training cohort), forming an extreme long-tail distribution [15]. To mitigate this skewness, two academic conventions are frequently employed: synthetic minority oversampling (e.g., SMOTE) [14], and post-hoc threshold sweeping across test ROC/PR curves to maximize F1 or G-Mean scores [10,15].

Both strategies break down in production engineering. SMOTE computes Euclidean distances across continuous feature spaces, requiring complete vectors that force synthetic imputation over sparse interaction logs, thereby corrupting informative missingness patterns [6]. More critically, post-hoc threshold sweeping assumes that decision-makers have access to test-set ground truth to identify the optimal cutoff. In production, uncalibrated continuous probabilities drift unpredictably across cohorts, causing fixed probability thresholds to generate highly erratic warning cohort sizes that overwhelm counseling staff.

Recently, Li et al. [15] modeled student failure in a blended *Fundamentals of Programming* course by incorporating a recall-penalty term into a Bayesian optimization objective $J(\theta) = \text{G-Mean} - \lambda \cdot \max(0, R_{\min} - R)$ and applying advanced kinematic features (such as LE and LS) to guide heterogeneous ensembles, achieving a test G-Mean of 0.7691. However, their experimental evaluation still relied on sweeping decision thresholds over $[0.01, 0.99]$ on the test cohort post-hoc, while uncompleted tasks were imputed using a heuristic 1.3$\times$ time penalty based on historical pilots. Consequently, physical quota pre-allocation and the structural integrity of real-world missingness remained unaddressed.

### 2.3 Deep Sequential Models, Tabular Foundation Models, and Hyperparameter Overtuning
Extracting behavioral latent representations from raw clickstreams provides substantial predictive gains [15]. In sequential deep learning, Kusumawardani and Alfarozi [18] showed that weekly feature aggregation outperformed noisy daily logs on OULAD, and that **removing positional encodings from Transformer encoders improved generalization**, underscoring that fine-grained calendar timestamps often introduce high-frequency behavioral noise. Furthermore, Shi et al. [1] observed that in studies directly comparing deep learning against classical machine learning, **70% of the competitive classical models utilized ensemble techniques**.

Concurrently, tabular foundation models such as TabPFN have emerged as strong contenders [16,17]. In *Nature*, Hollmann et al. [17] demonstrated that prior-data fitted networks (TabPFN v2) yield competitive inference on small tabular datasets ($\le 10,000$ samples, $\le 500$ features) within seconds via in-context learning. As surveyed by Jiang et al. [19], these foundation models eliminate downstream fine-tuning. However, because their synthetic priors are constructed from mathematical structural causal models, foundation models cannot natively incorporate domain-specific operational constraints (such as hard recall safety margins) and struggle with authentic, behavioral non-compliance missingness under temporal drift.

Finally, while hyperparameter optimization is known to impact predictive performance [14], AutoML theory cautions that on small validation samples, unconstrained search spaces overfit random validation noise, precipitating validation overtuning. Establishing an a priori regularized candidate pool is therefore an essential architectural requirement for real-world deployment.

---

## 3 Data Representation and Engineering Substrate

### 3.1 Cross-Semester Datasets and Problem Formulation
We construct our evaluation testbeds across two heterogeneous educational contexts, defining course failure as the positive class and strictly removing students who formally withdrew (Withdrawn):
- **Institutional Blended Compulsory Course**: Spanning six consecutive academic cohorts (2020–2025) of *Fundamentals of Programming*, a core STEM undergraduate course. Cohorts 2020–2023 constitute the training set ($N=3,633$, failure rate 3.85%); the 2024 cohort serves as the validation set for ratio freezing ($N=871$, failure rate 9.76%); and the 2025 cohort acts as the completely blind operational test set ($N=910$, failure rate 10.22%). This environment exhibits authentic class-prior drift across academic years.
- **Open University Learning Analytics Dataset (OULAD)**: A public benchmark encompassing modular distance learning [7]. We pool cohorts 2013B and 2013J for training ($N=9,489$); cohort 2014B serves as the validation set ($N=4,153$, failure rate 35.11%); and cohort 2014J constitutes the blind test set ($N=5,714$, failure rate 26.69%), strictly honoring the chronological sequence of February (B) and October (J) presentations.

### 3.2 Process Behavioral Dynamics Latent Variables
The instructional lifecycle is partitioned into $M=6$ discrete, equal-interval stages. We evaluate observation windows spanning stages 1 through $T$, where $T \in \{2, 3, 4, 5\}$. Feature channels capture video interactions, chapter reviews, coding homework, and overall activity (all) for the private course, and resources, quizzes, forum interactions, and global activity (all) for OULAD.

For student $i$ in course offering $c$ at stage $t$, let $\mathcal{K}_{i,t}$ denote the set of completed learning tasks with submission timestamps $\tau_{i,k}$. The feature substrate integrates baseline assessment metrics with five kinematic latent variables:
1. **Milestone Performance Metrics**: Frequency (count), arithmetic mean (mean), maximum score (max), and last recorded score (last) within each channel.
2. **Learning Engagement (LE)**: Normalized proximity to the earliest task completion timestamp within offering $c$ (higher scores denote proactive early completion):
   $$\mathrm{LE}_{i,t} = \frac{1}{|\mathcal{K}_{i,t}|} \sum_{k \in \mathcal{K}_{i,t}} \left(1 - \frac{\tau_{i,k} - \tau_k^{\min}}{\tau_k^{\max} - \tau_k^{\min} + \epsilon}\right)$$
3. **Learning Speed Regularity (LS)**: Normalized Shannon entropy across the inter-task completion intervals, quantifying pacing consistency:
   $$\mathrm{LS}_{i,t} = -\sum_{j} p_{i,j} \log_2 (p_{i,j} + \epsilon)$$
4. **Learning Acceleration (LAcc)**: Scaled second-order differences of task completion timestamps, capturing transitions between cramming and disengagement:
   $$\mathrm{LAcc}_{i,t} = \mathrm{Scale}(\Delta^2 \tau_{i,k})$$
5. **Relative Progress Index (RPI)**: Deviation from the offering-wide median task completion timestamp $\tau_k^{\mathrm{med}}$, normalized by the interquartile range:
   $$\mathrm{RPI}_{i,t} = \frac{1}{|\mathcal{K}_{i,t}|} \sum_{k \in \mathcal{K}_{i,t}} \frac{\tau_k^{\mathrm{med}} - \tau_{i,k}}{\mathrm{IQR}(\tau_k) + \epsilon}$$
6. **Task Completion Rate (CR)**: Proportion of assigned tasks fulfilled: $\mathrm{CR}_{i,t} = |\mathcal{K}_{i,t}| / |\mathcal{K}_t|$.

### 3.3 Educational Semantics of Structural Missingness
We apply a cross-stage first-order difference operator $\Delta f_{i,m,t} = f_{i,m,t} - f_{i,m,t-1}$. Crucially, **all unobserved assessment records and their downstream differences are preserved as native structural missing values (Structural NaNs)**.

**Educational Domain Definition**: In educational analytics, missing assessment records do not stem from sensor malfunctions or random survey non-responses (i.e., Missing Completely at Random or Missing at Random). Instead, they represent an intentional **non-compliance or non-action state**, wherein a student purposefully skipped, abandoned, or failed to submit an assigned academic task. Submitting work and scoring zero reflects an entirely different academic mental state than failing to submit altogether. Imputing these records with sample means, medians, or heuristic constant multipliers [15] smooths out vital failure signals by falsely conflating disengaged students with compliant peers.

By leaving unobserved scores as native NaNs, modern decision tree algorithms can naturally learn optimal default split directions for non-compliant behaviors. After removing all-empty columns, feature dimensions for $T \in \{2, 3, 4, 5\}$ are 116, 232, 389, and 581 for the institutional course, and 115, 231, 376, and 548 for OULAD.

---

## 4 Operational Evaluation Protocol and System Specifications

### 4.1 Frozen Warning Ratio and Boundary Fallback Mechanism
Model discrimination is evaluated via the geometric mean of recall and specificity, $\text{G-Mean} = \sqrt{\mathrm{Recall} \times \mathrm{Specificity}}$. G-Mean penalizes false negatives and false positives symmetrically, prevents degenerate majority-class predictions under heavy skewness, and evaluates rank cutoffs independently of uncalibrated probability thresholds.

In the validation academic year, all $n_{\mathrm{val}}$ students are ranked in descending order of predicted risk scores. Given a predetermined institutional safety constraint ($\mathrm{Recall} \ge \alpha$, baseline $\alpha=0.80$), the optimal cutoff rank $k^*$ is solved via:
$$k^* = \arg\max_{k \in \mathcal{S}_{\alpha}} \text{G-Mean}(k), \quad \text{where } \mathcal{S}_{\alpha} = \{k \in [1, n_{\mathrm{val}}] \mid \mathrm{Recall}(k) \ge \alpha\}$$
**Deterministic Boundary Fallback Mechanism**: If severe distribution shifts or stringent recall requirements render the feasible set empty ($\mathcal{S}_{\alpha} = \emptyset$), the system activates a deterministic fallback rule:
$$k^* = \min \left( \arg\max_{k \in [1, n_{\mathrm{val}}]} \mathrm{Recall}(k) \right)$$
This fallback selects the smallest cohort size that achieves maximum empirical recall, ensuring mathematical determinism under any empirical distribution. The operational warning ratio is then physically frozen:
$$p = \frac{k^*}{n_{\mathrm{val}}}$$
When scoring the subsequent unseen test cohort of size $n_{\mathrm{test}}$, the administrative team strictly flags the top $k_{\mathrm{test}} = \max(1, \min(n_{\mathrm{test}}, \mathrm{round}(p \times n_{\mathrm{test}})))$ students in a single forward pass, reporting objective test recall and G-Mean without any post-hoc adjustments.

### 4.2 Industrial Architecture: Cost-Sensitive Weighting vs. SMOTE
Our protocol enforces internal algorithmic cost-sensitive weighting (negative-to-positive ratio for XGBoost; balanced class weights for LightGBM, Random Forest, and Extra Trees) while strictly rejecting SMOTE:
1. **Preserving Capacity Semantics**: The warning ratio $p$ represents tangible human advisory capacity. SMOTE generates synthetic interpolations in feature space, distorting realistic decision boundary densities.
2. **Protecting Structural Missingness**: Distance-based oversampling requires dense numeric matrices, forcing arbitrary imputation that destroys the diagnostic value of structural NaNs.
3. **Decoupling Optimization from Operations**: Cost-sensitive weights guide tree splits during loss minimization, while the external ratio $p$ governs final headcount allocations, cleanly separating statistical learning from administrative planning.

### 4.3 Two-Stage Search and A Priori Candidate Pruning
To balance global exploration with local refinement, hyperparameter optimization runs across three independent random seeds (2026, 42, 1221). The search comprises **40 initial quasi-random exploration trials followed by 20 Optuna Tree-structured Parzen Estimator (TPE) exploitation trials**. All configurations are ranked by historical Out-of-Fold Cross-Validation (OOF CV) G-Mean on the training set.

To guard against validation overtuning, we enforce an **a priori candidate pruning rule** based on empirical sample variance:
- **Small-Sample Institutional Course (Top-5 Pruning)**: Given the limited validation cohort ($N=871$), single-year empirical metrics carry substantial sampling variance. Informed by prior setup experiments conducted strictly without test-set access, the search space is constrained to the top 5 OOF CV configurations ($k_{\mathrm{pool}}=5$), from which the model achieving the highest validation G-Mean under the recall constraint is selected.
- **Large-Sample OULAD Benchmark (Top-10 Selection)**: Supported by a larger validation cohort ($N=4,153$), sampling variance is inherently attenuated, permitting selection across the top 10 OOF candidates ($k_{\mathrm{pool}}=10$).

**No-Data-Snooping Protocol**: Candidate pool thresholds are locked prior to accessing the test data. The test cohorts (2025 cohort and 2014J presentation) remain physically blind throughout hyperparameter search, early stopping, and quota freezing. Inference is executed strictly once per configuration.

Boosting models implement an early stopping patience of 50 rounds against validation loss. Early stopping tracks G-Mean across 99 quantile thresholds, imposing a linear 4$\times$ penalty whenever recall drops below 0.80: $\text{Score}_{\mathrm{stop}} = \text{G-Mean} - 4 \times \max(0, 0.80 - \mathrm{Recall})$. This barrier forces boosting iterations to maintain adequate safety margins.

---

## 5 Main Empirical Analysis: Dual-Track Temporal Dynamics

### 5.1 Institutional Blended Course: Early Value, Mid-Term Plateau, and Late Jump
Table 1 reports the blind test performance across seeds on the institutional blended course (2025 test cohort, $N=910$).

**Table 1.** Formal test performance on the institutional blended course (Top-5 selection, 2025 test cohort, $N=910$). Parentheses indicate 95% student-level bootstrap confidence intervals (1,000 resamples).

| Seed | $T$ | Champion Model | Validation $\to$ Test G-Mean (95% CI) | Test AUC | Test Recall | Test Size $k$ | Frozen Ratio $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2026 | 2 | LightGBM | 0.7760 $\to$ 0.7213 (0.6736–0.7707) | 0.7773 | 0.7097 | 284 | 0.312285 |
| 2026 | 3 | XGBoost  | 0.7872 $\to$ 0.7443 (0.6896–0.7845) | 0.7891 | 0.7312 | 266 | 0.292767 |
| 2026 | 4 | XGBoost  | 0.8040 $\to$ 0.7415 (0.6934–0.7945) | 0.8023 | 0.7097 | 250 | 0.274397 |
| 2026 | 5 | LightGBM | 0.8691 $\to$ 0.8284 (0.7807–0.8621) | 0.8831 | 0.8172 | 207 | 0.227325 |
| 42   | 2 | LightGBM | 0.7876 $\to$ 0.7443 (0.6880–0.7880) | 0.7627 | 0.7419 | 276 | 0.303100 |
| 42   | 3 | XGBoost  | 0.8036 $\to$ 0.7358 (0.6838–0.7787) | 0.7880 | 0.7312 | 280 | 0.307692 |
| 42   | 4 | LightGBM | 0.8071 $\to$ 0.7311 (0.6667–0.7822) | 0.7894 | 0.6667 | 224 | 0.245695 |
| 42   | 5 | LightGBM | 0.8718 $\to$ 0.7875 (0.7273–0.8463) | 0.8458 | 0.7312 | 192 | 0.211251 |
| 1221 | 2 | Random Forest | 0.7713 $\to$ 0.7410 (0.6949–0.7857) | 0.7886 | 0.7527 | 291 | 0.320321 |
| 1221 | 3 | LightGBM | 0.7940 $\to$ 0.7024 (0.6486–0.7529) | 0.7619 | 0.6774 | 285 | 0.313433 |
| 1221 | 4 | LightGBM | 0.8127 $\to$ 0.7103 (0.6491–0.7688) | 0.7914 | 0.6237 | 214 | 0.235362 |
| 1221 | 5 | LightGBM | 0.8721 $\to$ 0.8254 (0.7766–0.8709) | 0.8768 | 0.7742 | 170 | 0.187141 |

Three key temporal insights emerge from these empirical trajectories:
1. **Decisive Operational Utility at Stage 2 ($T=2$)**: Stage 2 (roughly one-third into the term) represents the golden window for intervention. Here, tuned tree ensembles achieve a robust test G-Mean of 0.7213–0.7443 (AUC 0.7627–0.7886) and test recall of 0.7097–0.7527. By committing roughly 30%–32% of cohort capacity, institutions capture over 70% of ultimate failures early enough to execute meaningful remedial advising.
2. **Mid-Term Plateau and the Over-Compression Trap**: While validation G-Mean climbs monotonically across stages ($0.7783 \to 0.7949 \to 0.8079 \to 0.8710$), 1,000 bootstrap resamples demonstrate that test performance across $T=2,3,4$ forms an overlapping plateau (mean test G-Means: 0.7355, 0.7275, and 0.7276).
   - *Behavioral Mechanism*: Mid-term course components consist primarily of regular homework and low-stakes formative quizzes. At-risk students frequently engage in deadline-driven batch submissions and peer copying, injecting transient behavioral noise that artificially masks genuine skill deficits.
   - *The Over-Compression Trap*: Misled by this apparent mid-term separability, validation models become overly confident, aggressively compressing the frozen quota from **276–291 students at $T=2$ (approx. 30%–32% of the cohort) down to 214–250 students at $T=4$ (approx. 23%–27%)**. In the 2025 test cohort, this premature budget reduction causes severe false negatives, dropping test recall to 0.6237–0.7097 at $T=4$.
3. **Late-Stage Sprint Breakthrough at Stage 5 ($T=5$)**: In the final sprint stage ($T=5$), test G-Means jump to peak levels across all seeds (0.8284, 0.7875, 0.8254; mean 0.8138). For seed 2026, the 95% bootstrap confidence interval at $T=5$ (0.7807–0.8621) completely separates from its $T=2$ counterpart (0.6736–0.7707).
   - *Educational Mechanism*: Stage 5 introduces high-stakes **mock comprehensive examinations and milestone capstone submissions**. These summative assessments neutralize temporary cramming, exposing authentic variance in subject mastery and driving a genuine leap in model discrimination.

### 5.2 Open University (OULAD): Early Baseline and Rapid Saturation
Table 2 outlines model trajectories on the OULAD benchmark (2014J test presentation, $N=5,714$).

**Table 2.** Formal test performance on OULAD (Top-10 selection, 2014J test presentation, $N=5,714$).

| Seed | $T$ | Model | Validation $\to$ Test G-Mean (95% CI) | Test AUC | PR-AUC | Test Recall | Test Size $k$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2026 | 2 | XGBoost  | 0.8727 $\to$ 0.7788 (0.7669–0.7924) | 0.8726 | 0.7920 | 0.7639 | 2028 |
| 2026 | 3 | LightGBM | 0.8826 $\to$ 0.8243 (0.8136–0.8350) | 0.9161 | 0.8630 | 0.8374 | 2067 |
| 2026 | 4 | XGBoost  | 0.8980 $\to$ 0.8408 (0.8293–0.8505) | 0.9305 | 0.8785 | 0.8328 | 1903 |
| 2026 | 5 | XGBoost  | 0.9114 $\to$ 0.8495 (0.8390–0.8606) | 0.9394 | 0.8931 | 0.8754 | 2071 |
| 42   | 2 | LightGBM | 0.8761 $\to$ 0.7832 (0.7712–0.7962) | 0.8733 | 0.7901 | 0.7672 | 2010 |
| 42   | 3 | LightGBM | 0.8868 $\to$ 0.8258 (0.8147–0.8378) | 0.9181 | 0.8641 | 0.8302 | 2014 |
| 42   | 4 | LightGBM | 0.9002 $\to$ 0.8423 (0.8315–0.8531) | 0.9329 | 0.8835 | 0.8407 | 1936 |
| 42   | 5 | XGBoost  | 0.9094 $\to$ 0.8484 (0.8378–0.8590) | 0.9379 | 0.8928 | 0.8603 | 1996 |
| 1221 | 2 | XGBoost  | 0.8731 $\to$ 0.7771 (0.7642–0.7896) | 0.8691 | 0.7885 | 0.7561 | 1996 |
| 1221 | 3 | XGBoost  | 0.8856 $\to$ 0.8272 (0.8151–0.8380) | 0.9153 | 0.8627 | 0.8275 | 1987 |
| 1221 | 4 | LightGBM | 0.9002 $\to$ 0.8293 (0.8185–0.8402) | 0.9213 | 0.8676 | 0.8295 | 1981 |
| 1221 | 5 | XGBoost  | 0.9116 $\to$ 0.8444 (0.8344–0.8550) | 0.9362 | 0.8898 | 0.8741 | 2105 |

In distance education, the temporal dynamics follow a starkly different trajectory:
- **High Early Usability at $T=2$**: Mean test G-Mean reaches 0.7797 (AUC 0.8691–0.8733), showing that high-volume digital clickstreams reliably capture baseline disengagement early on.
- **Phase Transition and Saturation at $T=3$**: Moving from $T=2$ to $T=3$, test G-Means surge by +0.0455, +0.0426, and +0.0501 across seeds (mean 0.8258), with confidence intervals separating cleanly. Beyond Stage 3, gains flatten markedly: the total G-Mean improvement from $T=3$ to $T=5$ is roughly 0.02, with adjacent intervals heavily overlapping. In distance learning, students who fail to submit the first major milestone assignment at Stage 3 have effectively abandoned the course, leaving little marginal information for subsequent stages to extract.

---

## 6 Baseline Benchmarking and Overtuning Defense

### 6.1 Comprehensive Baseline Comparisons
Under the identical frozen ratio protocol, we evaluate our tuned tree ensembles against a single-feature baseline, logistic regression (LR), TabPFN (v3.5), an LSTM sequence model, and unsearched default Random Forest and XGBoost baselines. Table 3 aggregates performance across the critical early window ($T=2$) and the final sprint ($T=5$), reporting arithmetic means across seeds.

**Table 3.** Comprehensive baseline comparison on the institutional blended course (Validation 2024: $N=871$; Test 2025: $N=910$). Multi-seed configurations report arithmetic means.

| Model Specification | Window $T$ | Val G-Mean | Val Recall | Val $k$ | Test AUC | Test G-Mean | Test Recall | Test $k$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Our Tuned Ensembles (Top-5 Pool)** | **2** | **0.7783** | **0.8157** | **271.7** | **0.7762** | **0.7355** | **0.7348** | **283.7** |
| **Our Tuned Ensembles (Top-5 Pool)** | **5** | **0.8710** | **0.8784** | **181.7** | **0.8686** | **0.8138** | **0.7742** | **189.7** |
| Single-Feature `LE_hw__w12` | 2/5 | 0.7390 | 0.8706 | 367.0 | 0.7767 | 0.7142 | 0.8172 | 383.0 |
| Logistic Regression (LR, pool=5) | 2 | 0.7402 | 0.8314 | 338.0 | 0.7738 | 0.7047 | 0.7599 | 353.3 |
| Logistic Regression (LR, pool=5) | 5 | 0.8221 | 0.8667 | 246.7 | 0.8302 | 0.7594 | 0.7491 | 257.7 |
| TabPFN (v3.5, 8 estimators) | 2 | 0.7585 | 0.8118 | 298.0 | 0.7571 | 0.6929 | 0.6882 | 311.0 |
| LSTM (Recurrent, weight 24.95) | 2 | 0.7529 | 0.8824 | 356.0 | 0.7699 | 0.6999 | 0.7742 | 372.0 |
| Fixed Random Forest (500 trees) | 2 | 0.7383 | 0.8470 | 352.0 | 0.7711 | 0.7174 | 0.8029 | 367.7 |
| Fixed Random Forest (500 trees) | 5 | 0.8141 | 0.8079 | 209.7 | 0.8750 | 0.8327 | 0.8387 | 219.0 |
| Fixed XGBoost (Default strong cfg) | 2 | 0.7400 | 0.8471 | 349.7 | 0.7214 | 0.6821 | 0.7312 | 365.0 |
| Fixed XGBoost (Default strong cfg) | 5 | 0.8701 | 0.8510 | 158.7 | 0.8275 | 0.7513 | 0.6487 | 166.0 |

Key industrial conclusions derived from Table 3 include:
1. **Tuned Ensembles Dominate the Early Window ($T=2$)**: At $T=2$, our tuned tree models (mean test G-Mean 0.7355, point estimates 0.7213–0.7443) consistently outperform the single-feature baseline (0.7142), logistic regression (0.7047), TabPFN (0.6929), LSTM (0.6999), and default XGBoost (0.6821). Notably, logistic regression collapses entirely at $T=3$, plummeting to a test G-Mean of 0.5775 and flagging 485 students (53.3% of the cohort), which violates practical advisory capacity. Tabular foundation and deep sequence models suffer under small-sample constraints and structural NaNs in early sparse windows.
2. **Fixed Random Forest Confirms Objective Jump at Stage 5**: Unsearched default Random Forest models jump to a mean test G-Mean of 0.8327 at $T=5$ (seed 1221 reaches 0.8413, AUC 0.8758). **This orthogonal control confirms that the late-stage performance surge is driven by authentic instructional information release (mock exams) rather than hyperparameter search artifacts**. Because the default Random Forest achieved lower validation G-Means in 2024 (0.8073–0.8233) than tuned LightGBM (0.8691–0.8721), our pre-term protocol selected LightGBM without test label leakage.
3. **Fixed XGBoost Degrades Without Tuning**: Default XGBoost degrades across all windows, reaching only 0.7513 at $T=5$ with recall dropping to 0.6487. Off-the-shelf gradient boosting hyperparameter presets are ill-equipped for severe class imbalance and temporal drift without rigorous optimization.

### 6.2 Candidate Pool Regularization Against Validation Overtuning
Across 48 experimental splits on the institutional course, exactly 16 tasks selected conflicting configurations between the Top-5 ($k_{\mathrm{pool}}=5$) and Top-10 ($k_{\mathrm{pool}}=10$) rules. Paired back-testing on the unseen 2025 test cohort demonstrates:
- **75% Generalization Win Rate**: Top-5 selections outperform Top-10 in 12 out of 16 conflicting tasks, achieving an average net G-Mean gain of **+0.0135**.
- **Preventing Overtuning Collapse**: At $T=5$, the unconstrained Top-10 rule severely overfits the validation cohort. For seed 1221, Top-10 selects Trial 23 based on an inflated validation G-Mean of 0.8789, only to collapse to 0.7462 on the test set. Conversely, Top-5 constrains the selection to Trial 44 (validation score 0.8721), which secures a test G-Mean of 0.8254—a net advantage of **+0.0792**.

On small validation cohorts, empirical scores carry high sampling variance. Constraining selection to the top-ranked OOF CV candidates acts as an effective inductive bias that protects against overtuning.

---

## 7 Operational Simplification and Sensitivity Analysis

### 7.1 Sensitivity of the Safety Recall Threshold
Evaluating administrative safety thresholds $\alpha \in \{0.70, 0.75, 0.80, 0.85\}$ yields the following operational behaviors:
- **Constraint Relaxation ($\alpha \in [0.70, 0.80]$)**: Across all 12 champion configurations, the resulting cutoff quotas $k^*$, ratios $p$, and test G-Means are **completely identical (zero variation)** to the baseline $\alpha=0.80$. The unconstrained empirical maximum on the validation set naturally satisfies $\mathrm{Recall} \ge 0.80$, rendering relaxed constraints operationally inert.
- **Tightening to $\alpha = 0.85$**: Under this stringent threshold, candidate solutions that naturally satisfied the tighter safety margin retained their frozen quotas intact (e.g., Seed 42 at $T=3$ naturally achieved a validation recall of 0.8588, keeping $k^*=268$ and test G-Mean at 0.7358, while all seeds at $T=5$ inherently satisfied $\ge 0.85$). Conversely, early and mid-term windows falling short of 0.85 adaptively expanded their quotas to guarantee administrative safety, yielding smooth performance curves (Seed 2026: $0.6999 \to 0.7161 \to 0.7596 \to 0.8284$; Seed 42: $0.7110 \to 0.7358 \to 0.7417 \to 0.7875$). Overall, the system exhibits absolute parametric invariance across typical administrative bounds ($\alpha \in [0.70, 0.80]$), with maximum discrimination consistently locking at Stage 5.

### 7.2 Operational Parsimony: Unified vs. Per-Course Quotas
Educational institutions routinely distribute warning lists on a course-by-course basis. On OULAD, we contrast a single institution-wide warning ratio against independent per-course ratio tuning ($p_c$):
1. **Allocation Topology**: Global cohort ranking (Table 2, seed 2026 at $T=2$: 0.7788) vs. independent per-course ranking (combined test G-Mean: 0.8268). Per-course ranking prevents stricter courses from dominating institutional warning lists and represents the superior deployment topology.
2. **Quota Optimization**: Within this per-course ranking framework, tuning course-specific quotas $p_c$ against applying a single unified frozen ratio $p$ (33%–37%) yields an insignificant G-Mean difference ranging from **$-0.0041$ to $+0.0066$** across all 12 splits. Fine-grained per-course ratio tuning yields negligible statistical benefit. Central administrations can deploy a single global warning ratio across all departments without loss of predictive performance.

### 7.3 Calendar-Day Windowing Comparison
A natural calendar-day baseline on OULAD (seed 2026, 109 features) shows that at Day 14, test G-Means for logistic regression and LightGBM are 0.5743 (AUC 0.6628) and 0.5681 (AUC 0.6737), requiring excessive warning lists encompassing 64% of the cohort. At Day 28, logistic regression reaches 0.5955 and LightGBM reaches 0.6109, flagging 59%–61% of students. These calendar-day models lag our pedagogical stage-based framework by roughly 0.17 in G-Mean, demonstrating that milestone-aligned stage partitioning filters out calendar noise far more effectively.

---

## 8 Industrial Deployment Playbook and Operational Footprint

We synthesize our empirical findings into three concrete deployment guidelines for educational institutions:
1. **Blended Campus Course Playbook**: In physical campus environments, institutions should launch early proactive advising and faculty intervention at Stage 2 ($T=2$), where an approximate 31% warning quota captures over 70% of ultimate failures. During the mid-term plateau ($T=3,4$), administrators must resist the urge to aggressively trim intervention budgets, avoiding the over-compression trap. At Stage 5, a refined, high-confidence quota (~20%) should be deployed for final remedial exam preparation.
2. **Online MOOC Playbook**: In open online learning environments, behavioral attrition crystallizes early. Predictive separability peaks at Stage 3 ($T=3$) following the first major milestone assessment, after which marginal gains saturate. Administrators must launch primary re-engagement interventions decisively across the early-to-mid stages.
3. **Operational Footprint and Compute Efficiency**: The entire training, tuning, and dual-term scoring pipeline executes within seconds on commodity hardware. TabPFN requires roughly 15 seconds on a GPU but depends heavily on high-end VRAM (exceeding 12 hours on a CPU alone), whereas an LSTM requires roughly 300 seconds on a CPU. In contrast, tree ensemble retraining completes in roughly 10 seconds. Our primary system runs seamlessly on a single institutional desktop PC without requiring distributed computing infrastructure.

---

## 9 Limitations and Ethical Boundaries

1. **Empirical Scope**: The private dataset is drawn from a core STEM programming course at a single university, while OULAD pools five modular courses. Further cross-institutional validation is needed to assess transferability across humanities and social sciences.
2. **Ethical Guidelines and Algorithmic Justice**: Models must strictly target academic failure support; students who formally withdrew are excluded. Warning rosters must be used exclusively to guide remedial support, instructor check-ins, and tutoring access. They must never serve punitive tracking, course dismissal, or stigmatizing purposes, and data management must adhere to institutional privacy policies.

---

## 10 Conclusion

Operational early warning systems require actionable intervention quotas frozen prior to the start of the academic term. The Frozen Warning Ratio Protocol resolves the problem of post-hoc threshold sweeping, establishing a leakage-free benchmark for capacity-constrained educational environments. Our empirical evaluations demonstrate that physical blended courses offer high early screening utility at Stage 2, encounter an over-compression trap during mid-term homework cramming, and achieve a summative breakthrough at Stage 5. Conversely, open online courses reach informational saturation by Stage 3. Cost-sensitive tree ensembles paired with an a priori Top-5 candidate pruning rule provide strong protection against validation overtuning, while a single campus-wide quota policy simplifies administrative maintenance. This work shifts the focus of educational predictive modeling from offline metric optimization toward practical, deployment-ready operational engineering.

---

## References

1. Shi, X., Wang, Y., Ding, C., Gao, B.W., Shen, X.: Applications of machine learning for at-risk student prediction in online education: A 10-year systematic review. Journal of Computer Assisted Learning 41(1), e70058 (2025). https://doi.org/10.1111/jcal.70058
2. Baker, R.S.J.d., Yacef, K.: The state of educational data mining in 2009: A review and future visions. Journal of Educational Data Mining 1(1), 3–17 (2009)
3. Romero, C., Ventura, S.: Educational data mining: A review of the state of the art. IEEE Transactions on Systems, Man, and Cybernetics, Part C 40(6), 601–618 (2010). https://doi.org/10.1109/TSMCC.2010.2053532
4. Romero, C., Ventura, S.: Educational data mining and learning analytics: An updated survey. WIREs Data Mining and Knowledge Discovery 10(3), e1355 (2020). https://doi.org/10.1002/widm.1355
5. Bond, M., Khosravi, H., De Laat, M., Bergdahl, N., Negrea, V., Oxley, E., Pham, P., Chong, S.W., Siemens, G.: A meta-systematic review of artificial intelligence in higher education: a call for research integrity, accountability and transparency. International Journal of Educational Technology in Higher Education 21, 4 (2024). https://doi.org/10.1186/s41239-023-00436-2
6. Zhou, Q., Guan, X., He, L., Fang, H., Yin, C.: Educational data mining for smart education: Progress and trends. Journal of Software 26(11), 3026–3042 (2015). https://doi.org/10.13328/j.cnki.jos.004887
7. Kuzilek, J., Hlosta, M., Zdrahal, Z.: Open University Learning Analytics dataset. Scientific Data 4, 170171 (2017). https://doi.org/10.1038/sdata.2017.171
8. Gašević, D., Dawson, S., Rogers, T., Gasevic, D.: Learning analytics should not promote one size fits all: The effects of instructional conditions in predicting academic success. The Internet and Higher Education 28, 68–84 (2016). https://doi.org/10.1016/j.iheduc.2015.10.002
9. Macfadyen, L.P., Dawson, S.: Mining LMS data to develop an “early warning system” for educators: A proof of concept. Computers & Education 54(2), 588–599 (2010). https://doi.org/10.1016/j.compedu.2009.09.008
10. Adnan, M., Habib, A., Ashraf, J., Mussadiq, S., Raza, A.A., Abid, M., Bashir, M., Khan, S.U.: Predicting at-risk students at different percentages of course length for early intervention using machine learning models. IEEE Access 9, 7519–7539 (2021). https://doi.org/10.1109/ACCESS.2021.3049446
11. Feng, W., Tang, J., Liu, T.X.: Understanding dropouts in MOOCs. In: Proceedings of the AAAI Conference on Human Computation and Crowdsourcing (HCOMP), pp. 517–526 (2019)
12. Gardner, J., Yu, R., Nguyen, Q., Tingley, D., Rafferty, A., Brooks, C.: Cross-institutional transfer learning for educational models: Implications for model performance and algorithmic fairness. In: Proceedings of the 54th Hawaii International Conference on System Sciences (HICSS), pp. 1472–1481 (2021)
13. Schwerter, J., Fleischer, J., Wirth, J., Leutner, D.: Cross-course generalizability of SRL-aligned predictive models using digital learning traces. Journal of Learning Analytics 10(2), 56–73 (2023). https://doi.org/10.18608/jla.2023.7747
14. Rimal, S., Sharma, S., Alsadoon, A.: Hyperparameter tuning of machine learning algorithms for student performance prediction: A comparative study. Multimedia Tools and Applications 83, 74349–74364 (2024). https://doi.org/10.1007/s11042-024-18426-2
15. Li, Y., Chen, J., Yin, Z., Jia, H., Wu, Z.: Kinematics-inspired behavioral dynamics and risk-constrained ensemble learning for student performance prediction in blended environments. In: Proceedings of the 19th International Conference on Knowledge Science, Engineering and Management (KSEM 2026). LNAI, vol. 16631, pp. 381–391. Springer, Cham (2027). https://doi.org/10.1007/978-981-92-2759-4_28
16. Hollmann, N., Müller, S., Eggensperger, K., Hutter, F.: TabPFN: A transformer that solves small tabular classification problems in a second. In: Proceedings of the International Conference on Learning Representations (ICLR 2023) (2023)
17. Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Bischl, B., Hutter, F.: Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025). https://doi.org/10.1038/s41586-024-08328-6
18. Kusumawardani, S.S., Alfarozi, R.P.: Transformer encoder model for sequential prediction of student performance based on their log activities. IEEE Access 11, 23154–23166 (2023). https://doi.org/10.1109/ACCESS.2023.3246122
19. Jiang, M., Liu, Y., Cai, Z., Zhou, Z., Ye, F.: Representation learning for tabular data: A comprehensive survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025). https://doi.org/10.1109/TPAMI.2025.3530777
20. Chen, J., Feng, J., Sun, X., Wu, N.: Feature matching based heterogeneous transfer learning for student performance prediction. IEEE Transactions on Learning Technologies 15(4), 482–495 (2022). https://doi.org/10.1109/TLT.2022.3188556
