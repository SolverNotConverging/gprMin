"""Typed policy states for eigenmode anchor and modal-basis resolution.

The numerical eigenmode code produces observations such as modal overlaps and
forward-power classifications.  This module describes how those observations
move through the policy state machines.  It intentionally contains no field
arrays or logging so transitions can be reviewed and tested independently of
the FDFD and FDTD implementations.
"""

from dataclasses import dataclass, replace
from enum import Enum, IntEnum

import numpy as np


class AnchorRequestKind(Enum):
    """How the user requested the anchor frequencies."""

    AUTO = "auto"
    EXPLICIT = "explicit"


class AnchorTrackingOutcome(Enum):
    """Final tracking strategy retained for one mode."""

    BROADBAND = "broadband"
    GUARD_TRIMMED = "guard_trimmed"
    SINGLE_FALLBACK = "single_fallback"


class AnchorFallbackReason(Enum):
    """Why automatic resolution collapsed to the centre anchor."""

    NONE = "none"
    TRACKING_MISMATCH = "tracking_mismatch"
    DISCONNECTED_PROPAGATION = "disconnected_propagation"


class GuardTrimSide(Enum):
    """Spectral guard removed after a tracking failure."""

    NONE = "none"
    LOWER = "lower"
    UPPER = "upper"


class ModeResolutionState(Enum):
    """States traversed while resolving one modal anchor branch."""

    CANDIDATES = "candidates"
    TRACKING = "tracking"
    GUARD_TRIMMED = "guard_trimmed"
    TRACKED = "tracked"
    CLASSIFYING_POWER = "classifying_power"
    SINGLE_FALLBACK = "single_fallback"
    RESOLVED = "resolved"
    FAILED = "failed"


_MODE_TRANSITIONS = {
    ModeResolutionState.CANDIDATES: {
        ModeResolutionState.TRACKING,
        ModeResolutionState.FAILED,
    },
    ModeResolutionState.TRACKING: {
        ModeResolutionState.GUARD_TRIMMED,
        ModeResolutionState.TRACKED,
        ModeResolutionState.SINGLE_FALLBACK,
        ModeResolutionState.FAILED,
    },
    ModeResolutionState.GUARD_TRIMMED: {
        ModeResolutionState.TRACKING,
        ModeResolutionState.FAILED,
    },
    ModeResolutionState.TRACKED: {
        ModeResolutionState.CLASSIFYING_POWER,
        ModeResolutionState.FAILED,
    },
    ModeResolutionState.SINGLE_FALLBACK: {
        ModeResolutionState.CLASSIFYING_POWER,
        ModeResolutionState.RESOLVED,
        ModeResolutionState.FAILED,
    },
    ModeResolutionState.CLASSIFYING_POWER: {
        ModeResolutionState.SINGLE_FALLBACK,
        ModeResolutionState.RESOLVED,
        ModeResolutionState.FAILED,
    },
    ModeResolutionState.RESOLVED: set(),
    ModeResolutionState.FAILED: set(),
}


@dataclass(frozen=True)
class ModeResolutionTrace:
    """Immutable, validated audit trail for one mode-resolution attempt."""

    states: tuple = (ModeResolutionState.CANDIDATES,)

    @property
    def current(self):
        return self.states[-1]

    def advance(self, next_state):
        if next_state not in _MODE_TRANSITIONS[self.current]:
            raise RuntimeError(
                "Invalid eigenmode policy transition: "
                f"{self.current.value} -> {next_state.value}."
            )
        return replace(self, states=(*self.states, next_state))


class PortInitialisationState(Enum):
    """States traversed while the port coordinator retries anchor sets."""

    REQUESTED = "requested"
    ATTEMPTING = "attempting"
    RETRY_GUARD = "retry_guard"
    RETRY_SINGLE = "retry_single"
    COMMITTED = "committed"
    FAILED = "failed"


class PortRetryOutcome(Enum):
    """Final candidate-set strategy selected by the port coordinator."""

    BROADBAND = "broadband"
    GUARD_TRIMMED = "guard_trimmed"
    SINGLE_FALLBACK = "single_fallback"

    @property
    def legacy_auto_policy_name(self):
        return {
            PortRetryOutcome.BROADBAND: "auto_broadband",
            PortRetryOutcome.GUARD_TRIMMED: "auto_broadband_guard_trimmed",
            PortRetryOutcome.SINGLE_FALLBACK: "auto_single_fallback",
        }[self]


_PORT_TRANSITIONS = {
    PortInitialisationState.REQUESTED: {
        PortInitialisationState.ATTEMPTING,
        PortInitialisationState.FAILED,
    },
    PortInitialisationState.ATTEMPTING: {
        PortInitialisationState.RETRY_GUARD,
        PortInitialisationState.RETRY_SINGLE,
        PortInitialisationState.COMMITTED,
        PortInitialisationState.FAILED,
    },
    PortInitialisationState.RETRY_GUARD: {
        PortInitialisationState.ATTEMPTING,
        PortInitialisationState.FAILED,
    },
    PortInitialisationState.RETRY_SINGLE: {
        PortInitialisationState.ATTEMPTING,
        PortInitialisationState.FAILED,
    },
    PortInitialisationState.COMMITTED: set(),
    PortInitialisationState.FAILED: set(),
}


@dataclass(frozen=True)
class PortInitialisationTrace:
    """Immutable, validated audit trail for coordinated port retries."""

    states: tuple = (PortInitialisationState.REQUESTED,)

    @property
    def current(self):
        return self.states[-1]

    def advance(self, next_state):
        if next_state not in _PORT_TRANSITIONS[self.current]:
            raise RuntimeError(
                "Invalid eigenmode port transition: "
                f"{self.current.value} -> {next_state.value}."
            )
        return replace(self, states=(*self.states, next_state))


@dataclass(frozen=True)
class ModeAnchorResolution:
    """Structured result of resolving one mode across candidate anchors.

    Index tuples refer to the original candidate-anchor bank.  The legacy
    policy string is derived only at compatibility/output boundaries.
    """

    mode_index: int
    requested: AnchorRequestKind
    tracking: AnchorTrackingOutcome
    fallback_reason: AnchorFallbackReason
    guard_trim_side: GuardTrimSide
    nonpropagating_excluded: bool
    tracked_indices: tuple
    source_power_indices: tuple
    monitor_reference_indices: tuple
    nonpropagating_indices: tuple
    trace: ModeResolutionTrace

    def __post_init__(self):
        tracked = set(self.tracked_indices)
        power = set(self.source_power_indices)
        references = set(self.monitor_reference_indices)
        nonpropagating = set(self.nonpropagating_indices)
        if not power:
            raise ValueError("A resolved eigenmode requires at least one source-power anchor.")
        if not power <= references <= tracked:
            raise ValueError(
                "Eigenmode anchor resolution invariant failed: source-power anchors "
                "must be tracked monitor references."
            )
        if not nonpropagating <= tracked:
            raise ValueError("Non-propagating anchor indices must belong to the tracked branch.")
        if self.trace.current is not ModeResolutionState.RESOLVED:
            raise ValueError("A mode anchor resolution must end in the resolved state.")
        if self.tracking is AnchorTrackingOutcome.SINGLE_FALLBACK:
            if len(self.source_power_indices) != 1 or len(self.monitor_reference_indices) != 1:
                raise ValueError("A single-anchor fallback must collapse both anchor banks.")
            if self.fallback_reason is AnchorFallbackReason.NONE:
                raise ValueError("A single-anchor fallback requires an explicit reason.")
        elif self.fallback_reason is not AnchorFallbackReason.NONE:
            raise ValueError("A broadband anchor resolution cannot carry a fallback reason.")

    @property
    def uses_single_fallback(self):
        return self.tracking is AnchorTrackingOutcome.SINGLE_FALLBACK

    @property
    def permits_endpoint_source_coverage(self):
        """Whether source synthesis intentionally extrapolates its endpoint basis."""

        return self.nonpropagating_excluded or (
            self.requested is AnchorRequestKind.AUTO
            and self.tracking
            in {
                AnchorTrackingOutcome.GUARD_TRIMMED,
                AnchorTrackingOutcome.SINGLE_FALLBACK,
            }
        )

    @property
    def legacy_policy_name(self):
        """Return the existing serialized policy name for compatibility."""

        if self.uses_single_fallback:
            return "auto_single_fallback"
        policy = (
            "auto_broadband"
            if self.requested is AnchorRequestKind.AUTO
            else "explicit"
        )
        if self.tracking is AnchorTrackingOutcome.GUARD_TRIMMED:
            policy += "_guard_trimmed"
        if self.nonpropagating_excluded:
            policy += "_nonpropagating_trimmed"
        return policy

    def force_single_fallback(self, reason=AnchorFallbackReason.TRACKING_MISMATCH):
        """Return the compatibility result for an outer coordinated fallback."""

        if len(self.source_power_indices) != 1:
            raise ValueError("Forced single fallback requires a one-anchor resolved bank.")
        trace = ModeResolutionTrace().advance(ModeResolutionState.TRACKING)
        trace = trace.advance(ModeResolutionState.SINGLE_FALLBACK)
        trace = trace.advance(ModeResolutionState.CLASSIFYING_POWER)
        trace = trace.advance(ModeResolutionState.RESOLVED)
        return replace(
            self,
            requested=AnchorRequestKind.AUTO,
            tracking=AnchorTrackingOutcome.SINGLE_FALLBACK,
            fallback_reason=reason,
            guard_trim_side=GuardTrimSide.NONE,
            monitor_reference_indices=self.source_power_indices,
            trace=trace,
        )


class ModalBasisKind(IntEnum):
    """Mathematical interpretation of a prepared monitor basis."""

    NONE = 0
    POWER_WAVE = 1
    GENERALIZED = 2


class ModalReferenceKind(IntEnum):
    """Anchor bank selected for one frequency/mode pair."""

    NONE = 0
    PROPAGATING_BANK = 1
    EVANESCENT_RUN = 2
    OUTER_ENDPOINT = 3
    LEGACY_REFERENCE = 4


class ModalNormalizationKind(IntEnum):
    """Normalization applied after interpolating a modal profile."""

    NONE = 0
    UNIT_REAL_POWER = 1
    UNIT_BALANCED_EH = 2


@dataclass(frozen=True)
class ModalBasisPlan:
    """Array-backed, reviewable plan for monitor reference construction.

    ``interpolation_weights`` and ``profile_scales`` intentionally remain
    separate.  E/H profiles use their product, whereas effective index uses
    interpolation weights alone. ``profile_scales`` stores only the per-mode,
    per-anchor balanced-reference scale; power-wave profiles use scale one.
    """

    basis_kind: np.ndarray
    reference_kind: np.ndarray
    normalization_kind: np.ndarray
    reference_run_id: np.ndarray
    interpolation_weights: np.ndarray
    profile_scales: np.ndarray
    decomposition_eligible: np.ndarray

    def __post_init__(self):
        basis_shape = self.basis_kind.shape
        if len(basis_shape) != 2:
            raise ValueError("Eigenmode modal basis kind must have shape (frequency, mode).")
        for name in (
            "reference_kind",
            "normalization_kind",
            "reference_run_id",
            "decomposition_eligible",
        ):
            if getattr(self, name).shape != basis_shape:
                raise ValueError(f"Eigenmode modal basis {name} shape does not match {basis_shape}.")
        nf, nm = basis_shape
        if self.interpolation_weights.ndim != 3:
            raise ValueError("Eigenmode interpolation weights must have shape (mode, anchor, frequency).")
        if self.interpolation_weights.shape[0] != nm or self.interpolation_weights.shape[2] != nf:
            raise ValueError("Eigenmode interpolation weights do not match the modal basis dimensions.")
        expected_profile_shape = (
            self.interpolation_weights.shape[0],
            self.interpolation_weights.shape[1],
        )
        if self.profile_scales.shape != expected_profile_shape:
            raise ValueError(
                "Eigenmode profile scales must have shape (mode, anchor), matching "
                "the interpolation weights."
            )
        has_basis = self.basis_kind != int(ModalBasisKind.NONE)
        partition = np.sum(self.interpolation_weights, axis=1).T
        if np.any(has_basis & ~np.isclose(partition, 1.0, rtol=0.0, atol=1e-14)):
            raise ValueError("Eigenmode modal-basis interpolation weights must form a partition of unity.")
        power = self.basis_kind == int(ModalBasisKind.POWER_WAVE)
        generalized = self.basis_kind == int(ModalBasisKind.GENERALIZED)
        if np.any(
            power
            & (self.normalization_kind != int(ModalNormalizationKind.UNIT_REAL_POWER))
        ):
            raise ValueError("A power-wave modal basis must use unit-real-power normalization.")
        if np.any(
            generalized
            & (self.normalization_kind != int(ModalNormalizationKind.UNIT_BALANCED_EH))
        ):
            raise ValueError("A generalized modal basis must use balanced E/H normalization.")

    @property
    def power_wave_eligible(self):
        return self.basis_kind == int(ModalBasisKind.POWER_WAVE)

    @property
    def generalized(self):
        return self.basis_kind == int(ModalBasisKind.GENERALIZED)
