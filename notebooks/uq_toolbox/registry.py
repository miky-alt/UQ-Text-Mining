from enum import Enum
from lm_polygraph.estimators import *
from uqlm import (
    UQEnsemble, WhiteBoxUQ, BlackBoxUQ, LongTextUQ
)
from uqlm.judges import *

class UQLibrary(Enum):
    POLYGRAPH = "lm_polygraph"
    UQLM = "uqlm"

UQ_REGISTRY = {
    UQLibrary.POLYGRAPH: {
        "Attention": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": AttentionScore,
            "category": "Information-based",
            "compute": "Medium",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies sequence-level uncertainty by computing the entropy of attention weights, which reflects the model's focus distribution and internal token-level indecision during generation. [Reference: https://openreview.net/forum?id=LYx4w3CAgy]"
        },
        "DegMat": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "estimator_class": DegMat,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Computes sequence uncertainty and confidence scores using the degree matrix derived from a response similarity graph. High-degree nodes represent well-connected, consistent responses in high-confidence regions, whereas sequence uncertainty is quantified via the normalized trace complement ($U_{\text{Deg}} = \text{trace}(mI - D)/m^2$). [Reference: (Lin et al., 2023, https://arxiv.org/abs/2305.19187]"
        },
        "Eccentricity": {
                    "supported_granularity": ["sequence"],
                    "min_required_mode": "black",
                    "estimator_class": Eccentricity,
                    "category": "Sample diversity",
                    "compute": "High",
                    "memory": "Low",
                    "need_training": False,
                    "description": "Quantifies sequence uncertainty and individual confidence scores by constructing response coordinate embeddings via graph Laplacian eigenvectors. Uncertainty is measured as the norm of the offset from the average embedding ($U_{\text{Ecc}} = \|[v_1'^\top, \dots, v_m'^\top]\|_2$), while confidence corresponds to the negative distance from the center ($C_{\text{Ecc}}(x, s_j) = -\|v_j'\|_2$).[Reference: (Lin et al., 2023, https://arxiv.org/abs/2305.19187]"
                },
        "BoostedProbSequence": {
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "white",
            "estimator_class": BoostedProbSequence,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Derives sequence-level quality by aggregating token-level scores via the BOOSTEDPROB approach, specifically designed to mitigate model underconfidence in ambiguous text generation tasks. [Reference: https://aclanthology.org/2025.emnlp-main.166.pdf]"
        },
        "ClaimCondProb": {
            "supported_granularity": ["claim"],
            "min_required_mode": "white",
            "estimator_class": ClaimConditionedProbability,
            "category": "Sample diversity",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Isolates claim-level uncertainty by filtering token probabilities through NLI entailment, effectively removing noise arising from surface-form variations and claim types. [Reference: https://arxiv.org/pdf/2403.04696]"
        },
        "CocoaMSP": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": CocoaMSP,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty by synthesizing Maximum Sequence Probability with semantic consistency signals within the CoCoA framework to provide a more robust confidence measure. [Reference: https://openreview.net/pdf?id=h0wj0pdPjC]"
        },
        "CocoaMTE": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": CocoaMTE,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Combines Mean Token Entropy with consistency metrics within the CoCoA framework to generate a comprehensive uncertainty estimate that accounts for both token confidence and output variability. [Reference: https://openreview.net/pdf?id=h0wj0pdPjC]"
        },
        "CocoaPPL": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": CocoaPPL,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Integrates Perplexity scores with semantic consistency measures within the CoCoA framework to enhance uncertainty quantification for individual generated sequences. [Reference: https://openreview.net/pdf?id=h0wj0pdPjC]"
        },
        "EigenScore": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": EigenScore,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Estimates sequence uncertainty by calculating the LogDet of the covariance matrix of sentence embeddings, exploiting dense information in internal model states to capture semantic dispersion. [Reference: https://openreview.net/forum?id=Zj12nzlQbz]"
        },
        "FisherRao": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": FisherRao,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies sequence-level uncertainty by computing the geodesic distance between probability distributions on a Riemannian manifold using the Fisher information matrix, offering a robust geometric metric to detect distribution divergence in text generation. [Reference: https://arxiv.org/pdf/2212.09171.pdf]"
        },
        "Focus": {
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "white",
            "estimator_class": Focus,
            "category": "Information-based",
            "compute": "Medium",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty by imitating human factuality checking: it prioritizes informative keywords, propagates uncertainty from preceding tokens via attention weights to penalize overconfidence, and corrects probabilities using entity types and inverse document frequency (IDF) to mitigate underconfidence. [Reference: https://arxiv.org/abs/2311.13230]"
        },
        "KernelLanguageEntropy": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "estimator_class": KernelLanguageEntropy,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies sequence-level semantic uncertainty by computing the von Neumann entropy of kernels derived from pairwise semantic similarities between generations, offering a more fine-grained and expressive measure than cluster-based semantic entropy. [Reference: https://arxiv.org/pdf/2405.20003]"
        },
        "LabelProb": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "estimator_class": LabelProb,
            "category": "Information-based",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Estimates sequence uncertainty by calculating the relative frequency of generated response samples, providing a robust black-box approximation of predictive confidence when internal model probabilities are inaccessible. [Reference: https://doi.org/10.1162/tacl_a_00737]"
        },
        "LexicalSimilarity": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "estimator_class": LexicalSimilarity,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty by measuring the lexical overlap between multiple generated responses, based on the hypothesis that higher lexical consistency reflects greater model confidence; it serves as a baseline that, unlike semantic methods, does not account for semantic equivalence (i.e., different wordings with the same meaning). [Reference: https://arxiv.org/abs/2302.09664]"
        },
        "Linguistic1S": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "estimator_class": Linguistic1S,
            "category": "Reflexive",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Elicits calibrated confidence scores by prompting the model to express uncertainty using pre-defined linguistic terms (e.g., 'highly likely'), which are then mapped to numerical probabilities derived from human survey data, allowing for direct self-evaluation of confidence. [Reference: https://arxiv.org/abs/2305.14975]"
        },
        "LUQ": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "estimator_class": LUQ,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty in long-form generation by assessing sentence-level consistency via NLI classifiers, effectively mitigating the pitfalls of high lexical overlap in extended texts. [Reference: https://aclanthology.org/2024.emnlp-main.299.pdf]"
        },
        "MahalanobisDistanceSeq": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": MahalanobisDistanceSeq,
            "category": "Density-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": True,
            "description": "Quantifies uncertainty by modeling class-conditional Gaussian distributions on hidden layer embeddings and measuring the Mahalanobis distance of new samples from the nearest class centroid, providing a robust metric for Out-of-Distribution (OOD) detection and adversarial robustness. [Reference: https://arxiv.org/abs/1807.03888]"
        },
        "MaximumSequenceProbability": {
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "white",
            "estimator_class": MaximumSequenceProbability,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "A foundational baseline that quantifies sequence-level uncertainty as the inverse of the total sequence log-probability; despite its simplicity, it is a robust benchmark against which more complex methods should be measured to ensure valid performance comparisons. [Reference: https://doi.org/10.1162/tacl_a_00737]"
        },
        "MaximumTokenProbability": {
            "supported_granularity": ["token"],
            "min_required_mode": "white",
            "estimator_class": MaximumTokenProbability,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Estimates token-level uncertainty by directly retrieving the log-probability of the generated token from the model's output distribution; represents a fundamental white-box feature of autoregressive language models."
        },
        "NumSemSets": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "estimator_class": NumSemSets,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty by clustering multiple generated responses into semantically equivalent sets using NLI-based similarity; the number of resulting clusters serves as a robust proxy for the semantic dispersion of the model's output distribution. [Reference: https://arxiv.org/abs/2305.19187]"
        },
        "PTrue": {
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "white",
            "estimator_class": PTrue,
            "category": "Reflexive",
            "compute": "Medium",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty by prompting the model to evaluate the truthfulness of its own generated samples, interpreting the probability assigned to the 'True' token as a confidence score regarding the validity of the generated content. [Reference: https://arxiv.org/abs/2207.05221]"
        },
        "Perplexity": {
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "white",
            "estimator_class": Perplexity,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty via the exponentiated average negative log-likelihood of generated tokens, serving as a foundational baseline to measure the model's confidence in its output distribution. [Reference: https://doi.org/10.1162/tacl_a_00330]"
        },
        "RAUQ": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": RAUQ,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty by recurrently fusing token-level confidence measures with activation patterns from identified 'uncertainty-aware' attention heads, achieving efficient hallucination detection in a single forward pass. [Reference: https://arxiv.org/abs/2505.20045]"
        },
        "RDESeq": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": RDESeq,
            "category": "Density-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": True,
            "description": "Performs Robust Density Estimation (RDE) in the feature space by leveraging Kernel PCA for dimensionality reduction and Minimum Covariance Determinant (MCD) for outlier-resistant covariance estimation. [Reference: Yoo et al. (2022), https://doi.org/10.18653/v1/2022.findings-acl.289]"
        },
        "RMDSeq": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": RelativeMahalanobisDistanceSeq,
            "category": "Density-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": True,
            "description": "Computes a contrastive OOD score by subtracting the Mahalanobis distance of a background distribution from the in-domain Mahalanobis distance, effectively normalizing for layer-specific variance. [Reference: Ren et al. (2023), https://doi.org/10.48550/arXiv.2209.15558]"
        },
        "SelfCertainty": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": SelfCertainty,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies confidence by measuring the KL-divergence between the predicted token distribution and a uniform distribution at each generation step, indicating prediction peakedness. [Reference: Kang et al. (2025), https://doi.org/10.48550/arXiv.2502.18581]"
        },
        "Verbalized1S": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "estimator_class": Verbalized1S,
            "category": "Reflexive",
            "compute": "Medium",
            "memory": "Low",
            "need_training": False,
            "description": "Extracts confidence by prompting the model to output a numerical probability or linguistic expression of certainty alongside the answer in a single generation stage. [Reference: Tian et al. (2023), https://doi.org/10.48550/arXiv.2305.14975]"
        },
        "Verbalized2s": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "estimator_class": Verbalized2S,
            "category": "Reflexive",
            "compute": "Medium",
            "memory": "Low",
            "need_training": False,
            "description": "Extracts calibrated confidence scores by querying the model for its probability of correctness in a secondary, follow-up prompt after the initial answer generation. [Reference: Tian et al. (2023), https://doi.org/10.48550/arXiv.2305.14975]"
        },
        "SemanticEntropy": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": SemanticEntropy,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Estimates semantic uncertainty by clustering generated sequences via bidirectional entailment and computing the entropy of the resulting meaning distribution. [Reference: Kuhn et al. (2023), https://doi.org/10.48550/arXiv.2302.09664]"
        },
        "TokenEntropy": {
            "supported_granularity": ["token"],
            "min_required_mode": "white",
            "estimator_class": TokenEntropy,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Calculates the entropy of the softmax probability distribution at each decoding step, quantifying lexical uncertainty. [Reference: Fomicheva et al. (2020), https://doi.org/10.1162/tacl_a_00330]"
        },
        "MeanTokenEntropy": {
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "white",
            "estimator_class": MeanTokenEntropy,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Computes sentence-level uncertainty by averaging token-level entropy scores (Softmax-Ent) over the generated sequence. [Reference: Fomicheva et al. (2020), https://doi.org/10.1162/tacl_a_00330]"
        },
        "PMI": {
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "white",
            "estimator_class": PointwiseMutualInformation,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies the informativeness and relevance of output tokens by calculating the Positive Pointwise Mutual Information (PPMI) between input utterances and generated responses. [Reference: Takayama & Arase (2019), https://aclanthology.org/W19-4115/]"
        },
        "MPMI": {
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "white",
            "estimator_class": MeanPointwiseMutualInformation,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Estimates sequence informativeness and relevance by computing the arithmetic mean of token-level Pointwise Mutual Information (PMI) scores between input and output. [Reference: Takayama & Arase (2019), https://aclanthology.org/W19-4115/]"
        },
        "SentSAR": {
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "estimator_class": SentenceSAR,
            "category": "Sample diversity",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty by shifting attention to semantically relevant sentences, mitigating the bias where irrelevant content disproportionately influences total predictive entropy. [Reference: Duan et al. (2024), https://doi.org/10.48550/arXiv.2307.01379]"
        }
    },
    UQLibrary.UQLM: {
            "MeanTokenNegentropy": {
            "scorer_id": "mean_token_negentropy", 
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "wrapper_class": WhiteBoxUQ,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies sequence certainty by aggregating token-level entropies derived from the model's logit distribution. Computes a confidence score via negentropy (-H), offering an intrinsic measure of predictive uncertainty based on the model's output log-probabilities. [Reference: Duan et al. (2026), https://doi.org/10.48550/arXiv.2510.11170]"
        },
        "SemanticSetsConfidence": {
            "scorer_id": "semantic_sets_confidence",
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "wrapper_class": BlackBoxUQ,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Estimates confidence by grouping stochastic samples into semantically consistent sets. Higher confidence is assigned to responses falling into dominant semantic clusters, reflecting stability in the underlying generation manifold rather than superficial token variance. [Reference: Lin et al. (2023), https://arxiv.org/pdf/2305.19187]"
        },
        "ExactMatch": {
            "scorer_id": "exact_match", 
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "wrapper_class": BlackBoxUQ,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies confidence via 'Sampling Repetition': measures the fraction of stochastic samples that exactly match the greedy (temperature=0) response. Higher repetition ratios indicate lower model epistemic uncertainty. [Reference: Cole et al. (2023), https://arxiv.org/abs/2305.14613]"
        },
        "Entailment": {
            "scorer_id": "entailment", 
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "black",
            "wrapper_class": {"sequence": BlackBoxUQ, "claim": LongTextUQ},
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Derives an 'Observed Consistency' score by leveraging NLI models to classify directional logical entailment between stochastic samples and the reference response. Reliability is assessed by aggregating these directional signals to detect semantic contradictions. [Reference: Chen & Mueller (2023), https://arxiv.org/abs/2308.16175]"
        },
        "NonContradiction": {
            "scorer_id": "noncontradiction", 
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "black",
            "wrapper_class": {"sequence": BlackBoxUQ, "claim": LongTextUQ},
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Estimates reliability as the Non-Contradiction Probability (NCP), defined as the aggregation of extrinsic probabilities (1 - P(contradiction)) across NLI-evaluated samples relative to a reference. Quantifies semantic stability in black-box scenarios. [Reference: Chen & Mueller (2023), https://arxiv.org/abs/2308.16175]"
        },
        "BertScore": {
            "scorer_id": "bert_score",
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "black",
            "wrapper_class": {"sequence": BlackBoxUQ, "claim": LongTextUQ},
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies factual consistency by computing the average BERTScore between a reference response and stochastic samples. Uses contextual embeddings to derive an inconsistency score; lower aggregate distance from samples indicates higher factual reliability. [Reference: Manakul et al. (2023), https://arxiv.org/abs/2303.08896]"
        },
        "NormalizedCosineSim": {
            "scorer_id": "cosine_sim",
            "supported_granularity": ["sequence", "claim"],
            "min_required_mode": "black",
            "wrapper_class": {"sequence": BlackBoxUQ, "claim": LongTextUQ},
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Measures response proximity by computing the normalized cosine similarity between contextual sentence embeddings of the reference and sampled outputs. Higher values signify higher semantic manifold density and increased model certainty. [Reference: Shorinwa et al. (2025), https://arxiv.org/abs/2412.05563]"
        },
        "WhiteSemanticNegentropy": {
            "scorer_id": "semantic_negentropy",
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "wrapper_class": WhiteBoxUQ,
            "category": "Information-based", 
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty by utilizing the LLM's internal token-level logit distributions to weight semantic meaning estimates. This white-box approach improves upon naive entropy by conditioning the certainty assessment directly on the model's internal predictive distribution, providing a more granular signal than black-box sampling alone. [Reference: Farquhar et al. (2024), https://doi.org/10.1038/s41586-024-07421-0]"
        },
        "BlackSemanticNegentropy": {
            "scorer_id": "semantic_negentropy",
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "wrapper_class": BlackBoxUQ,
            "category": "Sample diversity",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies uncertainty by clustering stochastic samples into semantically equivalent meaning-groups via bidirectional entailment. It computes discrete semantic entropy over these clusters and applies normalization to map the score to a [0, 1] confidence range, where higher values indicate increased semantic consistency. This normalization corrects for the number of samples (m+1) to ensure the metric is bounded and interpretable[cite: 4]. [Reference: Farquhar et al. (2024), https://doi.org/10.1038/s41586-024-07421-0]"
        },
        "PTrue": {
            "scorer_id": "p_true",
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "wrapper_class": WhiteBoxUQ,
            "category": "Reflexive",
            "compute": "Medium",
            "memory": "Low",
            "need_training": False,
            "description": "Quantifies confidence via self-reflection: the model is prompted to classify its own previous response as 'True' or 'False'. The confidence score is derived from the token probability of the 'True' label, effectively serving as an internal self-consistency check. [Reference: Kadavath et al. (2022), https://arxiv.org/abs/2207.05221]"
        },
        "ProbabilityMargin": {
            "scorer_id": "probability_margin",
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "wrapper_class": WhiteBoxUQ,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Calculates sequence-level confidence by averaging the probability margin—the difference between the top-1 ($p_{t_{j1}}$) and top-2 ($p_{t_{j2}}$) token probabilities—across all decoding steps $j$. High margins signify decisive token selection, whereas small margins denote model ambiguity or potential uncertainty. [Reference: Farr et al. (2024), https://arxiv.org/abs/2408.08217]"
        },
        "MonteCarloProbability": {
            "scorer_id": "monte_carlo_probability",
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "wrapper_class": WhiteBoxUQ,
            "category": "Sample diversity",
            "compute": "Medium",
            "memory": "Low",
            "need_training": False,
            "description": "Computes a robust confidence score by averaging the length-normalized sequence probabilities (geometric mean of token probabilities) of the original response and multiple stochastic samples. This Monte Carlo estimation (MCSP) improves upon single-sequence probability by accounting for the model's predictive variance across the generation manifold. [Reference: Kuhn et al. (2023), https://arxiv.org/abs/2302.09664]"
        },
        "SemanticDensity": {
            "scorer_id": "semantic_density",
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "wrapper_class": WhiteBoxUQ,
            "category": "Sample diversity",
            "compute": "Medium",
            "memory": "Low",
            "need_training": False,
            "description": "Estimates response correctness by approximating a probability density function (PDF) in semantic space. It integrates semantic similarity—derived from NLI-based contradiction and neutrality scores—with length-normalized token-probability weighting to prioritize density in regions of probable correctness. [Reference: Qiu & Miikkulainen (2024), https://arxiv.org/abs/2405.13845]"
        },
        "SequenceProbability": {
            "scorer_id": "sequence_probability",
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "wrapper_class": WhiteBoxUQ,
            "category": "Information-based",
            "compute": "Low",
            "memory": "Low",
            "need_training": False,
            "description": "Calculates sequence-level likelihood as the geometric mean of token probabilities (length-normalized) by default, or as the raw joint probability when configured. Length-normalization is critical to ensure fair comparisons across responses of varying lengths. [Reference: Vashurin et al. (2024), https://arxiv.org/abs/2406.15627]"
        },
        "ConsistencyAndConfidence": {
            "scorer_id": "consistency_and_confidence",
            "supported_granularity": ["sequence"],
            "min_required_mode": "white",
            "wrapper_class": WhiteBoxUQ,
            "category": "Sample diversity",
            "compute": "Medium",
            "memory": "Medium",
            "need_training": False,
            "description": "Hybrid approach (CoCoA) that calculates the product of length-normalized token probability (LNTP) and normalized cosine similarity (NCS) across multiple sampled responses. This dual-signal requirement ensures that high uncertainty scores are assigned only when the model demonstrates both token-level confidence and semantic consistency across generations. [Reference: Vashurin et al. (2025), https://arxiv.org/abs/2502.04964]"
        },
        "BSDetector": {
            "scorer_id": "bs_detector",
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "wrapper_class": UQEnsemble,
            "uqlm_default_ensemble": True,
            "category": "Ensemble Scorers",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Ensemble method that aggregates three signals: Non-Contradiction Probability (56%), Exact Match Rate (14%), and Ternary Self-Reflection (30%). This weighted combination leverages consistency-based and self-judgment metrics to provide a robust framework for hallucination detection. [Reference: Chen & Mueller (2023), https://arxiv.org/abs/2308.16175]"
        },
        "GeneralizedEnsemble": {
            "scorer_id": "generalized_ensemble",
            "supported_granularity": ["sequence"],
            "min_required_mode": "black",
            "wrapper_class": UQEnsemble,
            "category": "Ensemble Scorers",
            "compute": "High",
            "memory": "Low",
            "need_training": False,
            "description": "Flexible ensemble framework enabling the combination of any black-box, white-box, or LLM-as-a-Judge scorers. Supports weighted aggregation of components and automated weight tuning via Optuna or grid search to optimize objectives such as ROC-AUC, F1-score, or log-loss. [Reference: Chen & Mueller (2023), [Reference: Bouchard & Chauhan (2025), https://arxiv.org/abs/2504.19254]"
        },
    }
}

def list_uq_techniques(
    library: UQLibrary = None,
    mode: str = None,
    granularity: str = None,
    category: str = None,
    compute: str = None,
    need_training: bool = None
):
    """
    Query interface for uncertainty techniques. 
    Filters and displays the registry contents including descriptions and supported granularities.
    """
    print("\n🔍 UQ FRAMEWORK CORE ENGINE - METHOD TAXONOMY AND HARDWARE COST ANALYSIS")

    separator = "=" * 160
    header = f"{'TECHNIQUE':<35} | {'LIBRARY':<12} | {'MODE':<6} | {'CATEGORY':<20} | {'TRAIN':<5} | {'COMPUTE':<8} | {'DESCRIPTION'}"

    print(separator)
    print(header)
    print(separator)

    count = 0
    for lib_enum, techniques in UQ_REGISTRY.items():
        if library and lib_enum != library:
            continue
            
        for tech_name, info in techniques.items():
            req_mode = info.get("min_required_mode", "N/A")
            cat = info.get("category", "General")
            comp = info.get("compute", "Low")
            train = "Yes" if info.get("need_training", False) else "No"
            desc = info.get("description", "No description available.")
            
            # --- Filtri ---
            if mode and req_mode.lower() != mode.lower(): 
                continue
            if compute and comp.lower() != compute.lower(): 
                continue
            if need_training is not None and info.get("need_training", False) != need_training: 
                continue
            if category and category.lower() not in cat.lower(): 
                continue
                
            # ✨ Gestione corretta di supported_granularity (che è una lista)
            if granularity:
                supported_list = info.get("supported_granularity", [])            
                
                if not any(granularity.lower() in g.lower() for g in supported_list):
                    continue

            # Stampa riga con inclusione della descrizione
            print(f"{tech_name:<35} | {lib_enum.value:<12} | {req_mode:<6} | {cat:<20} | {train:<5} | {comp:<8} | {desc}")
            count += 1

    print(separator)
    print(f"🎯 Query completed. Displaying {count} matched uncertainty techniques.\n")
