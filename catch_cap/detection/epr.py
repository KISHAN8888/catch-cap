"""Entropy Production Rate (EPR) detection logic.

Based on the paper: "Learned Hallucination Detection in Black-Box LLMs
using Token-level Entropy Production Rate" (arXiv:2509.04492)

EPR measures model uncertainty via Shannon entropy computed from top-K
token probabilities at each generation step. WEPR extends this with
learned weights for improved hallucination detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..config import EPRConfig
from ..types import EPRAnalysis, GenerationResult, TokenLogProb


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_entropy_from_logprobs(logprobs: List[float]) -> float:
    """Compute Shannon entropy (in bits) from log probabilities.

    H = -Σ p_i * log₂(p_i)

    Args:
        logprobs: List of log probabilities (natural log)

    Returns:
        Entropy in bits
    """
    if not logprobs:
        return 0.0

    # Convert logprobs to probabilities
    probs = [math.exp(lp) for lp in logprobs]

    # Normalize probabilities (they may not sum to 1 if truncated)
    prob_sum = sum(probs)
    if prob_sum <= 0:
        return 0.0
    probs = [p / prob_sum for p in probs]

    # Compute entropy: H = -Σ p * log₂(p)
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def _compute_entropic_contribution(logprob: float) -> float:
    """Compute the entropic contribution of a single token.

    s_k = -p_k * log₂(p_k)

    Args:
        logprob: Log probability of the token

    Returns:
        Entropic contribution (always non-negative)
    """
    p = math.exp(logprob)
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p)


def _epr_to_probability(epr_score: float, threshold: float = 0.80) -> float:
    """Convert EPR score to hallucination probability.

    Uses a sigmoid-like mapping centered around the threshold.

    Args:
        epr_score: EPR score in bits
        threshold: Center point for the sigmoid (default: 0.80)

    Returns:
        Probability between 0 and 1
    """
    # Scale factor controls steepness (higher = steeper transition)
    scale = 3.0

    # Sigmoid centered at threshold
    # P(hallucination) = sigmoid(scale * (epr - threshold))
    x = scale * (epr_score - threshold)
    return 1.0 / (1.0 + math.exp(-x))


# ============================================================================
# EPR Detector
# ============================================================================

@dataclass
class EPRDetector:
    """Detect hallucination risk using Entropy Production Rate.

    This implements the methodology from arXiv:2509.04492 which uses
    token-level entropy from top-K logprobs to detect hallucinations
    in a single generation pass.

    Example:
        config = EPRConfig(top_k=10, epr_threshold=0.80)
        detector = EPRDetector(config)

        analysis = detector.analyse(generation_result)
        if analysis.is_suspicious:
            print(f"Potential hallucination detected! EPR={analysis.epr_score:.2f}")
    """

    config: EPRConfig

    def analyse(self, response: GenerationResult) -> EPRAnalysis:
        """Analyze a response using EPR and optionally WEPR.

        Args:
            response: Generation result with token_logprobs populated

        Returns:
            EPRAnalysis with entropy scores and uncertainty flags
        """
        token_logprobs = response.token_logprobs

        # Fallback if no top-K logprobs available
        if not token_logprobs:
            return self._fallback_analysis(response)

        total_tokens = len(token_logprobs)
        if total_tokens == 0:
            return EPRAnalysis(
                epr_score=0.0,
                wepr_score=None,
                hallucination_probability=0.0,
                token_uncertainties=None,
                high_uncertainty_tokens=None,
                is_suspicious=False,
            )

        # Compute per-token entropies
        token_entropies: List[float] = []
        token_uncertainties: List[float] = []
        high_uncertainty_tokens: List[Tuple[int, str, float]] = []

        max_entropy = math.log2(self.config.top_k) if self.config.top_k > 1 else 1.0

        for idx, token_lp in enumerate(token_logprobs):
            # Get top-K logprobs for this token position
            top_logprobs = [lp for _, lp in token_lp.top_logprobs[:self.config.top_k]]

            # Compute entropy for this position
            entropy = _compute_entropy_from_logprobs(top_logprobs) if top_logprobs else 0.0
            token_entropies.append(entropy)

            # Compute uncertainty score (normalized to 0-1 range)
            uncertainty = min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0
            token_uncertainties.append(uncertainty)

            # Flag high-uncertainty tokens
            if uncertainty >= self.config.token_uncertainty_threshold:
                high_uncertainty_tokens.append((idx, token_lp.token, uncertainty))

        # Compute EPR: average entropy across all tokens
        epr_score = sum(token_entropies) / total_tokens

        # Compute WEPR if weights are available
        wepr_score = None
        if self.config.wepr_weights:
            wepr_score = self._compute_wepr(token_logprobs)

        # Compute hallucination probability
        if wepr_score is not None:
            # Use WEPR-based probability (sigmoid of WEPR score)
            hallucination_probability = 1.0 / (1.0 + math.exp(-wepr_score))
        else:
            # Use EPR-based probability (heuristic sigmoid centered at threshold)
            hallucination_probability = _epr_to_probability(
                epr_score,
                self.config.epr_threshold
            )

        # Determine if suspicious
        is_suspicious = self._is_suspicious(
            epr_score=epr_score,
            hallucination_prob=hallucination_probability,
            high_uncertainty_count=len(high_uncertainty_tokens),
            total_tokens=total_tokens
        )

        return EPRAnalysis(
            epr_score=epr_score,
            wepr_score=wepr_score,
            hallucination_probability=hallucination_probability,
            token_uncertainties=token_uncertainties,
            high_uncertainty_tokens=high_uncertainty_tokens if high_uncertainty_tokens else None,
            is_suspicious=is_suspicious,
        )

    def _compute_wepr(self, token_logprobs: List[TokenLogProb]) -> float:
        """Compute Weighted Entropy Production Rate.

        WEPR = (1/L) * Σⱼ S_β(q, t<j)
        where S_β = β₀ + Σₖ βₖ * sₖⱼ
        and sₖⱼ = -pₖⱼ * log₂(pₖⱼ) is the entropic contribution of rank k at position j

        Args:
            token_logprobs: List of TokenLogProb for each position

        Returns:
            WEPR score (can be negative, unlike EPR)
        """
        if not self.config.wepr_weights:
            return 0.0

        weights = list(self.config.wepr_weights)
        total_weighted_entropy = 0.0

        for token_lp in token_logprobs:
            # Get top-K logprobs
            top_lps = token_lp.top_logprobs[:self.config.top_k]

            # Compute weighted sum: β₀ + Σₖ βₖ * sₖⱼ
            position_score = self.config.wepr_bias  # β₀

            for k, (_, logprob) in enumerate(top_lps):
                if k + 1 < len(weights):  # weights[0] is bias, weights[1] is for rank 1
                    entropic_contrib = _compute_entropic_contribution(logprob)
                    position_score += weights[k + 1] * entropic_contrib

            total_weighted_entropy += position_score

        # Average across tokens
        return total_weighted_entropy / len(token_logprobs) if token_logprobs else 0.0

    def _is_suspicious(
        self,
        epr_score: float,
        hallucination_prob: Optional[float],
        high_uncertainty_count: int,
        total_tokens: int
    ) -> bool:
        """Determine if the response is suspicious based on EPR metrics.

        A response is flagged as suspicious if ANY of these conditions are met:
        1. WEPR probability exceeds threshold (if WEPR is configured)
        2. EPR score exceeds threshold
        3. Too many tokens have high uncertainty
        """
        # Check 1: WEPR probability threshold
        if self.config.wepr_weights and hallucination_prob is not None:
            if hallucination_prob >= self.config.wepr_threshold:
                return True

        # Check 2: EPR score threshold
        if epr_score >= self.config.epr_threshold:
            return True

        # Check 3: High uncertainty token ratio
        if total_tokens > 0:
            high_uncertainty_ratio = high_uncertainty_count / total_tokens
            if high_uncertainty_ratio >= self.config.high_uncertainty_ratio_threshold:
                return True

        return False

    def _fallback_analysis(self, response: GenerationResult) -> EPRAnalysis:
        """Fallback analysis when only legacy logprobs are available.

        This provides limited functionality using just the selected token's logprob.
        """
        logprobs = response.logprobs

        if not logprobs:
            return EPRAnalysis(
                epr_score=0.0,
                wepr_score=None,
                hallucination_probability=0.0,
                token_uncertainties=None,
                high_uncertainty_tokens=None,
                is_suspicious=False,
            )

        # Estimate entropy from single logprobs (rough approximation)
        avg_logprob = sum(logprobs) / len(logprobs)

        # Convert to pseudo-entropy
        pseudo_entropy = -avg_logprob / math.log(2)  # Convert to bits
        pseudo_entropy = max(0.0, min(pseudo_entropy, 10.0))  # Clamp

        hallucination_probability = _epr_to_probability(
            pseudo_entropy,
            self.config.epr_threshold
        )
        is_suspicious = pseudo_entropy >= self.config.epr_threshold

        return EPRAnalysis(
            epr_score=pseudo_entropy,
            wepr_score=None,
            hallucination_probability=hallucination_probability,
            token_uncertainties=None,
            high_uncertainty_tokens=None,
            is_suspicious=is_suspicious,
        )

    def get_confidence_level(self, analysis: EPRAnalysis) -> str:
        """Get human-readable confidence level from analysis.

        Args:
            analysis: EPRAnalysis result

        Returns:
            One of: "high", "medium", "low", "very_low"
        """
        epr = analysis.epr_score

        if epr < 0.5:
            return "high"
        elif epr < 0.80:
            return "medium"
        elif epr < 1.2:
            return "low"
        else:
            return "very_low"


# ============================================================================
# Convenience Functions
# ============================================================================

def create_detector(
    top_k: int = 10,
    epr_threshold: float = 0.80,
    token_uncertainty_threshold: float = 0.7,
) -> EPRDetector:
    """Create an EPR detector with common settings.

    Args:
        top_k: Number of top logprobs to use
        epr_threshold: EPR threshold for suspicious flag
        token_uncertainty_threshold: Per-token uncertainty threshold

    Returns:
        Configured EPRDetector
    """
    config = EPRConfig(
        top_k=top_k,
        epr_threshold=epr_threshold,
        token_uncertainty_threshold=token_uncertainty_threshold,
    )
    return EPRDetector(config)


def quick_check(response: GenerationResult, threshold: float = 0.80) -> bool:
    """Quick check if a response is potentially hallucinated.

    Args:
        response: Generation result with token_logprobs
        threshold: EPR threshold

    Returns:
        True if suspicious, False otherwise
    """
    detector = create_detector(epr_threshold=threshold)
    analysis = detector.analyse(response)
    return analysis.is_suspicious
