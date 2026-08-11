import pytest

from gprMax.eigenmode_policy import (
    AnchorFallbackReason,
    AnchorRequestKind,
    AnchorTrackingOutcome,
    GuardTrimSide,
    ModeAnchorResolution,
    ModeResolutionState,
    ModeResolutionTrace,
    PortInitialisationState,
    PortInitialisationTrace,
    PortRetryOutcome,
)


def _resolved_trace(*intermediate):
    trace = ModeResolutionTrace()
    for state in intermediate:
        trace = trace.advance(state)
    return trace.advance(ModeResolutionState.RESOLVED)


def test_mode_resolution_trace_validates_transitions():
    trace = ModeResolutionTrace().advance(ModeResolutionState.TRACKING)
    trace = trace.advance(ModeResolutionState.GUARD_TRIMMED)
    trace = trace.advance(ModeResolutionState.TRACKING)
    trace = trace.advance(ModeResolutionState.TRACKED)
    trace = trace.advance(ModeResolutionState.CLASSIFYING_POWER)
    trace = trace.advance(ModeResolutionState.RESOLVED)

    assert trace.current is ModeResolutionState.RESOLVED
    with pytest.raises(RuntimeError, match="resolved -> tracking"):
        trace.advance(ModeResolutionState.TRACKING)


def test_port_initialisation_trace_validates_retry_sequence():
    trace = PortInitialisationTrace().advance(PortInitialisationState.ATTEMPTING)
    trace = trace.advance(PortInitialisationState.RETRY_GUARD)
    trace = trace.advance(PortInitialisationState.ATTEMPTING)
    trace = trace.advance(PortInitialisationState.RETRY_SINGLE)
    trace = trace.advance(PortInitialisationState.ATTEMPTING)
    trace = trace.advance(PortInitialisationState.COMMITTED)

    assert trace.states == (
        PortInitialisationState.REQUESTED,
        PortInitialisationState.ATTEMPTING,
        PortInitialisationState.RETRY_GUARD,
        PortInitialisationState.ATTEMPTING,
        PortInitialisationState.RETRY_SINGLE,
        PortInitialisationState.ATTEMPTING,
        PortInitialisationState.COMMITTED,
    )
    assert (
        PortRetryOutcome.GUARD_TRIMMED.legacy_auto_policy_name
        == "auto_broadband_guard_trimmed"
    )


def test_structured_resolution_derives_legacy_policy_and_coverage():
    resolution = ModeAnchorResolution(
        mode_index=1,
        requested=AnchorRequestKind.AUTO,
        tracking=AnchorTrackingOutcome.GUARD_TRIMMED,
        fallback_reason=AnchorFallbackReason.NONE,
        guard_trim_side=GuardTrimSide.LOWER,
        nonpropagating_excluded=True,
        tracked_indices=(1, 2, 3),
        source_power_indices=(2, 3),
        monitor_reference_indices=(1, 2, 3),
        nonpropagating_indices=(1,),
        trace=_resolved_trace(
            ModeResolutionState.TRACKING,
            ModeResolutionState.GUARD_TRIMMED,
            ModeResolutionState.TRACKING,
            ModeResolutionState.TRACKED,
            ModeResolutionState.CLASSIFYING_POWER,
        ),
    )

    assert resolution.legacy_policy_name == (
        "auto_broadband_guard_trimmed_nonpropagating_trimmed"
    )
    assert resolution.permits_endpoint_source_coverage


def test_resolution_rejects_power_anchor_outside_reference_bank():
    with pytest.raises(ValueError, match="source-power anchors"):
        ModeAnchorResolution(
            mode_index=1,
            requested=AnchorRequestKind.EXPLICIT,
            tracking=AnchorTrackingOutcome.BROADBAND,
            fallback_reason=AnchorFallbackReason.NONE,
            guard_trim_side=GuardTrimSide.NONE,
            nonpropagating_excluded=False,
            tracked_indices=(0, 1),
            source_power_indices=(0,),
            monitor_reference_indices=(1,),
            nonpropagating_indices=(),
            trace=_resolved_trace(
                ModeResolutionState.TRACKING,
                ModeResolutionState.TRACKED,
                ModeResolutionState.CLASSIFYING_POWER,
            ),
        )
