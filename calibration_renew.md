# Calibration Renewal Plan

Working notes for issue #32 and the calibration workflow redesign.

## Goals

- Keep a fully manual calibration mode controlled with number keys.
- Add an automatic mode where targets advance without number keys, while SPACE still controls when measurement starts.
- Keep the target animated while gaze data is being collected.
- Keep the experimenter in control of collection timing.
- Keep the design simple and consistent with the rest of DeToX.

## Public API

The public API remains a single method:

```python
def calibrate(
    self,
    # Calibration procedure
    calibration_points=5,
    pacing='manual',
    presentation_order='random',

    # Stimuli
    infant_stims=True,
    shuffle=True,
    stim_size='big',

    # Attention
    anim_type='zoom',
    collection_anim_type='zoom',
    audio=True,

    # Results
    visualization_style='circles',
):
```

### Decisions

- Use `pacing`, not the generic `mode`.
- Use `presentation_order`, not the generic `order`.
- Keep `anim_type` as the existing pre-collection animation setting for backward compatibility.
- Add `collection_anim_type` for the settling/collection phase.
- Do not add separate public `calibrate_manual()` and `calibrate_auto()` functions.
- Remove the public `auto_paced` and `inter_point_delay` arguments.

### Validation

- `pacing` must be `'manual'` or `'auto'`.
- `anim_type` and `collection_anim_type` must be `'zoom'` or `'trill'`.
- `presentation_order` applies only to automatic pacing.
- `presentation_order` is either `'random'` or a complete list/tuple permutation of the 1-based calibration point numbers.
- Manual pacing ignores `presentation_order`.
- Animation names are validated at the public API boundary.

## Interaction Modes

### Manual pacing

```text
Calibration starts
→ Blank screen

Press number
→ That calibration target appears

Press Space
→ Settle, then collect data for that target

Collection succeeds
→ Target disappears
→ Wait for another number

Press Enter
→ Show calibration results
```

- Number keys provide complete control over target presentation.
- Space always means “collect the visible target”.
- Nothing advances automatically.
- The experimenter may revisit a point with its number key during the pass.
- Enter preserves the existing early-results workflow.

### Automatic pacing

```text
Calibration starts
→ First target in the presentation order appears

Press Space
→ Settle, then collect data for that target

Collection succeeds
→ Next unfinished target appears immediately

Last target succeeds
→ Show calibration results
```

- The first target appears automatically.
- SPACE still controls when collection starts.
- The next target appears only after successful collection.
- Attempts start only when the experimenter presses SPACE; there is no timed automatic collection.
- Number keys are ignored in automatic pacing; manual mode is the way to choose arbitrary points.
- Enter remains available as an early-results escape hatch.

### Failure

- A reported collection failure does **not** advance the target.
- The same point remains active so SPACE can retry it.
- Automatic pacing does not remove a failed point from its queue.

### Escape

- Escape while waiting or settling aborts immediately.
- Escape while Tobii collection is running sets an abort request.
- The display keeps animating until the Tobii call returns.
- Data is cleared and calibration mode is left only after the worker has finished.

## Presentation Order

- `presentation_order='random'` is the default for automatic pacing.
- Random means a shuffled permutation of the current pass, not random selection with repetition.
- An explicit order uses 1-based point numbers, for example `[1, 5, 2, 3, 4]`.
- An explicit order must contain every calibration point exactly once.
- Retry passes filter the requested order down to the selected retry points while preserving relative order.
- `shuffle` remains a separate concept: it shuffles which stimulus images belong to points, not the point order.

## Collection Engine

Replace the current manual/auto collection split with one shared frame-driven presentation engine.

### Function names

```text
_present_and_collect_calibration_points
    _prepare_automatic_point_order
    _activate_calibration_target
    _update_settling_and_start_collection
    _start_point_data_collection
    _update_point_data_collection
```

The first four live in `BaseCalibrationSession`. The final two are subclass collection implementations.

### Structure decisions

- `_present_and_collect_calibration_points` owns the main loop and state transitions.
- It is one shared engine; there are not two complete manual/auto collection loops.
- Mode-specific behaviour is limited to:
  - the initial target;
  - whether number keys select targets;
  - what happens after successful collection.
- Do not use numbers in function names such as `_1_...` or `_1_1_...`.
- Use numbered section comments inside functions to show reading order.
- Keep the main loop moderately long rather than splitting every branch into tiny helper functions.
- Keep only three substantial helpers in the base class; input handling, state transitions and drawing stay visible in the main loop.
- Remove `_auto_paced_collection_phase()`.
- Remove `inter_point_delay` and all blank intervals between points.
- Do not create many tiny helpers such as separate audio start/stop or frame-draw methods.

### Presentation states

```text
WAITING_FOR_COLLECTION
        │ Space
        ▼
SETTLING_ON_TARGET
        │ focus time elapsed
        ▼
COLLECTING_DATA
        │ success
        ├── manual → hidden target, wait for number
        ├── auto   → next ordered target, or finish
        └── failure → same target, wait for Space to retry
```

### Helpers

#### `_prepare_automatic_point_order`

- Returns an empty list for manual pacing.
- Shuffles the current pass for `'random'`.
- Filters explicit 1-based orders to the current pass.
- Used for both the initial pass and retry passes.

#### `_activate_calibration_target`

- Sets the target position.
- Resets stimulus orientation.
- Restarts the target animation clock.
- Starts looping attention audio if necessary.
- Resets audio volume to full.

#### `_update_settling_and_start_collection`

- Advances the focus timer.
- Fades audio without blocking the frame loop.
- Starts data collection when the focus period ends.
- Returns whether collection has started.

#### `_start_point_data_collection`

- Starts collection for the current point.
- Tobii implementation submits `collect_data(x, y)` to a background worker.
- Only the SDK collection call belongs in the worker; PsychoPy and calibration state stay on the main thread.

#### `_update_point_data_collection`

- Called once per frame while collecting.
- Returns:
  - `None` while collection is still running;
  - `True` on success;
  - `False` on reported failure.
- Tobii implementation polls a `Future` and reads its result only after `done()` is true.
- Real worker exceptions propagate on the main thread.

### Comment style

Use the existing DeToX style: numbered or named section comments, plus meaningful conceptual explanations.

```python
# --- 1. Prepare Points for This Pass ---
# Calibration requires known target positions. Automatic pacing prepares a
# queue so the participant cannot anticipate the next point location.

# --- 2. Initialize Presentation State ---
# ...

# --- 3. Present and Collect Points Frame by Frame ---
# --- 3a. Process Keyboard Input ---
# --- 3b. Update Settling ---
# --- 3c. Update Data Collection ---
# --- 3d. Draw Current Frame ---
```

Comments should explain both the practical step and the calibration reason behind it, not merely restate the code.

## Threading and Timing

- `Future` is a standard-library result handle, not a thread.
- One `ThreadPoolExecutor(max_workers=1)` handles Tobii collection for a calibration session.
- Do **not** use a dummy `sleep(0)` task to warm the thread. That optimisation was rejected as unnecessary and janky.
- Create the executor once during Tobii calibration setup.
- Submit the real `collect_data()` call only after the participant has had the focus period to settle.
- Submit returns immediately and creates a Future for the point.
- Poll `future.done()` once per frame.
- Call `future.result()` only after `done()` is true, otherwise the main thread blocks.
- A running `collect_data()` cannot be cancelled through the supported public API.
- Shut down the executor with `wait=True` before leaving Tobii calibration mode.
- The first collection may pay thread-startup overhead; subsequent collections reuse the worker.
- No application-level delay is inserted after settling.
- There is no blank screen between points; auto advances on the next frame after a successful collection.
- The settling deadline is checked at frame boundaries, so collection starts on the first frame at or after the focus deadline.
- Measure real tracker startup timing before adding any further optimisation.

### Avoid work after the deadline

- Avoid inserting blocking SDK calls between the end of settling and `collect_data()`.
- If `discard_data()` or another preparation call measures noticeably, move it before the deadline rather than delaying collection.
- Do not submit actual collection before settling ends; that would contaminate calibration samples.

## Animations

Two animation phases:

1. **Attention animation** before collection.
2. **Collection animation** during settling and data collection.

Both phases support the existing primitives: `zoom` and `trill`. All four combinations are allowed through:

```python
anim_type='trill'
collection_anim_type='zoom'
```

### Recommended default sequence

- Attention: `trill` at the large preset.
- Collection: `zoom` between roughly 50% and 10% of the nominal maximum size.

Reasons:

- The collaborator preferred trill followed by zoom.
- Zoom keeps the target centre fixed during measurement.
- Shrinking the target encourages fixation on the exact calibration point.
- The collection animation runs throughout settling and Tobii collection, so the target does not appear frozen.

### Direct size settings

Every phase and preset size is configured directly in
`ETSettings.AnimationSettings`. No runtime midpoint or threshold calculation
is performed.

- Big preset:
  - attention zoom: 9.5–16% of screen height;
  - collection zoom: 5–9.5%;
  - attention trill: 12.75%;
  - collection trill: 7.25%.
- Small preset:
  - attention zoom: 3–6% of screen height;
  - collection zoom: 1–3%;
  - attention trill: 4.5%;
  - collection trill: 2%.
- Attention and collection trill rotation ranges are also configured
  independently.
- `stim_size='big'/'small'` selects the preset.
- Reset orientation smoothly when changing out of a trill animation.

### Current defaults

- Attention animation: `trill`.
- Collection animation: `zoom`.
- Attention and collection sizes are explicit in `AnimationSettings`.
- Both phases are active, with a smooth blend during `focus_time`.

## Current Status

Completed:

- [x] Add public `pacing` and `presentation_order`.
- [x] Add public `collection_anim_type`.
- [x] Validate pacing, animation types and automatic order.
- [x] Apply the grouped `calibrate()` signature.
- [x] Propagate configuration through both session constructors.
- [x] Propagate pacing and presentation order through both `run()` methods.
- [x] Keep the existing collection implementation temporarily as a bridge.
- [x] Update tests for the new public contract.
- [x] Full test suite passes after the API wiring stage.
- [x] Add direct attention and collection size settings for both presets.
- [x] Point the current attention animation at the new attention fields.
- [x] Add zoom and trill pose calculations with a selectable animation phase.
- [x] Add focused tests for bounds, phase selection, rotation and pauses.
- [x] Replace the two collection loops with one shared presentation loop.
- [x] Add automatic point ordering and target activation helpers.
- [x] Add temporary blocking collection adapters.
- [x] Remove `_auto_paced_collection_phase()` and the blank interval.
- [x] Update tests for manual pacing, automatic pacing, retries and aborts.
- [x] Move the focus period into the presentation loop.
- [x] Fade audio incrementally from elapsed settling time.
- [x] Remove blocking focus waits and the stepped audio fade from collectors.
- [x] Advance mouse sampling from the presentation loop.
- [x] Reset pending mouse samples when calibration is aborted.
- [x] Run Tobii collection through one executor and Future per session.
- [x] Poll Tobii completion while target animation continues.
- [x] Check Tobii success/failure and propagate worker exceptions.
- [x] Defer Escape only while Tobii collection is active.
- [x] Finish pending collection before leaving calibration mode.

- [x] Blend from the displayed attention pose into the collection animation.
- [x] Reuse `focus_time` as the transition duration; no separate timing setting.
- [x] Convert the calculated height-unit size once before blending and drawing.
- [x] Default to attention `trill` and collection `zoom`.

Still temporary:

- Static API/vignette documentation still describes the old auto-paced arguments.

## Remaining Implementation Order

1. [x] Add the zoom/trill calculation helpers using the explicit animation settings.
2. [x] Replace `_collection_phase()` with `_present_and_collect_calibration_points()`.
3. [x] Delete `_auto_paced_collection_phase()` and the blank interval.
4. [x] Implement the shared manual/auto state machine.
5. [x] Convert focus/fade handling to non-blocking frame updates.
6. [x] Convert mouse sample collection to frame-based updates.
7. [x] Add background Tobii collection through a single executor and Future.
8. [x] Handle failure, Escape, executor shutdown and calibration-mode cleanup.
9. [x] Add the smooth transition into the continuous collection animation.
10. [x] Switch the recommended default to attention `trill`, collection `zoom`.
11. [ ] Perform a real Tobii hardware timing/animation test.
12. [ ] Update API and vignette documentation.

## Explicitly Rejected

- Separate public manual and automatic calibration functions.
- Separate complete manual and automatic collection loops.
- Automatic mode with number-key overrides.
- Automatic collection without a Space press.
- `auto_paced` and `inter_point_delay` public arguments.
- Blank intervals between automatic targets.
- Numbered function names.
- Many tiny helper functions.
- Dummy worker warm-up with `sleep(0)`.
- Cancelling a running Tobii collection with `Future.cancel()`.
- Leaving calibration mode or discarding data while the worker is active.
- Blocking focus waits or audio fades in the main presentation loop.

## Open Items for Testing

- Exact animation sizes and timings for both phases.
- Actual first-collection worker startup overhead on the target hardware.
- Whether point-data discarding needs preparation before the focus deadline.
- Real-tracker behaviour when Escape is pressed during `collect_data()`.
- Retry ordering and result-screen interaction with the final shared loop.
