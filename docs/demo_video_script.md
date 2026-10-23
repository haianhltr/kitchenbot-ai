# Demo video script

> **Script, not a recording.** This file is the shot list for issue #25. It fixes what the video shows, how long each shot lasts and the exact words on screen before any trial has been run, so nothing can be chosen after seeing the results. It contains no result. Every empty slot is a hidden to-do comment in the page source whose first words give the number of the GitHub issue that fills it, and `tests/test_docs_skeletons.py` checks those numbers and the total length.

The video has one job: a viewer can tell, at every moment, which controller is moving which part of the robot, and can tell a trained policy from a scripted motion. It is silent by design because the labels carry the information; any audio is the operator's choice at publishing time.

**Length.** The shot list below is a cut of 60 to 90 seconds, as issue #49 specifies. `docs/evaluation.md` section 8 and issue #25 ask for 2 to 3 minutes; the director reconciles the two. A longer cut can reuse the same shots with the full-length S4 successes.

## Rules that apply to every clip

1. **Every clip is one evaluation trial.** It is re-rendered from the seed and the configuration of a trial that is listed in a committed `outputs/eval/*.json` file, through the evaluator's own code path. Nothing is keyframed, hand-driven or played back from a recorded joint trajectory.
2. **The outcome must match the file.** After rendering, the outcome of the clip (success, or the failure tag) is compared with the outcome stored for that seed. If they differ, the clip is not used and the mismatch is written into the limitations of the report.
3. **Selection is fixed here and is never redone by eye.** The clips of shots 02 and 04 to 09 use the first successful seeds of their result file in ascending order (for S0, successful means the robot did not fall), the failure clips use the first seed in ascending order for each of the most frequent failure tags, and the comparison clip of shot 03 uses the first seed with no filter on its outcome. A clip is not replaced because it looks better or worse than another.
4. **Labels are read, not typed.** The words on screen are produced from the run block of the JSON file and from the live environment by the table in "Label text" below.
5. **Speed.** Real time is the default. A clip longer than its slot may be sped up by a whole factor, and the corner text always shows the factor. There is no slow motion and no cut inside a clip.
6. **Which results.** Clips use the unseen split of the scenario when its file exists, and otherwise the seen split; the corner text names the split either way. Synthetic result files and files from a modified working tree are never used for the final cut.

<!-- TODO(#25): implement rules 1 to 6 in scripts/record_demo.py and state in its help text which result file and seed each clip was rendered from -->

## Shot list

| Shot | Scenario | Base | Camera | Seconds | Lower-body label | Failure tag |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | card | none | card | 4 | title card | none |
| 02 | S0 | balance policy | wide | 6 | `own policy v1` | none |
| 03 | S0 | balance policy | split-wide | 8 | `own policy v1` and `Agile baseline` | none |
| 04 | S1 | balance policy | medium-right | 6 | `own policy v1` | none |
| 05 | S2 | balance policy | hand-close | 7 | `own policy v1` | none |
| 06 | S3 | fixed pelvis | slot-side | 7 | `pelvis fixed (no policy)` | none |
| 07 | S4 | balance policy | wide | 16 | `own policy v1` | none |
| 08 | S4 | balance policy | wide | 5 | `own policy v1` | none |
| 09 | S4 | balance policy | wide | 5 | `own policy v1` | none |
| 10 | S4 | balance policy | wide | 4 | `own policy v1` | most frequent |
| 11 | S4 | balance policy | wide | 4 | `own policy v1` | second most frequent |
| 12 | S4 | balance policy | wide | 4 | `own policy v1` | third most frequent |
| 13 | card | none | card | 5 | closing card | none |

The lower-body label in the table is the default configuration. Which lower-body controller and which hand the final clips show follows from the result files, through the label table below.

<!-- TODO(#8): record which lower-body controller the director chose for the MVP; if it is the Agile baseline, the default label of shots 02 and 04 to 12 changes to the Agile text and nothing else -->
<!-- TODO(#18): record the outcome of the Dex3 or Dex1 decision; the hand label of shots 05 to 12 follows it through the label table -->

## Label text

The overlay has four rows. The row names are fixed; the values come from this table.

| Row | When | Value |
| --- | --- | --- |
| LOWER BODY | `base` is floating and `policy` is own | `own policy v1` |
| LOWER BODY | `base` is floating and `policy` is agile | `Agile baseline` |
| LOWER BODY | `base` is fixed | `pelvis fixed (no policy)` |
| ARM | scenario S0 | `held posture (no IK)` |
| ARM | scenarios S1 to S4 | `DifferentialIK (right), held posture (left)` |
| HAND | scenarios S0 and S1 | `held open (no grasp)` |
| HAND | `hand` is kinematic | `KinematicFollowHand (attach baseline)` |
| HAND | `hand` is dex3 | `Dex3SynergyHand` |
| HAND | `hand` is dex1 | `Dex1GripperHand` |
| STATE | scenarios S0 and S1 | `n/a (no planner)` |
| STATE | scenarios S2 to S4 | the planner state as it occurs: `idle`, `pre_grasp`, `grasp`, `lift`, `transport`, `align`, `insert`, `release`, `done` or `failed` |

The names follow [architecture.md](architecture.md) sections 5 and 8. The attach baseline always carries its suffix because it is not a grasp (decision D007). When `base` is fixed the policy field is ignored and the row says that no policy is active.

## Shot details

Each block is the exact text drawn in the top-left panel while the shot starts. The STATE value changes as described under the block.

### Shot 01: title card

```
KitchenBot AI
A simulated Unitree G1 loads a dishwasher rack
Isaac Lab and Isaac Sim, PhysX
Every clip is one evaluation trial. Nothing is keyframed.
```

What happens: the card is shown on the plain card background; no motion.

<!-- TODO(#25): keep the last line only if the check of rule 1 passed for every clip in the cut -->

### Shot 02: S0, stand under pushes

```
LOWER BODY  own policy v1
ARM         held posture (no IK)
HAND        held open (no grasp)
STATE       n/a (no planner)
```

What happens: the robot stands on the flat floor while external pushes land; the clip starts one second before the first push of the trial and shows the push flash and the response. Camera `wide`.

<!-- TODO(#7): result file, seed and commit of the S0 clip, chosen by rule 3 -->

### Shot 03: S0, own policy against the Agile baseline

Left half of the frame:

```
LOWER BODY  own policy v1
ARM         held posture (no IK)
HAND        held open (no grasp)
STATE       n/a (no planner)
```

Right half of the frame:

```
LOWER BODY  Agile baseline
ARM         held posture (no IK)
HAND        held open (no grasp)
STATE       n/a (no planner)
```

What happens: the same seed, the same pushes and the same camera pose in both halves, each half rendered from its own result file and joined side by side. This is the project's first robot-learning comparison, so the clip is not filtered on outcome: whatever each controller did is shown, including a fall.

<!-- TODO(#7): the two result files and the shared seed, chosen by rule 3 with no outcome filter -->

### Shot 04: S1, reach with the balance policy active

```
LOWER BODY  own policy v1
ARM         DifferentialIK (right), held posture (left)
HAND        held open (no grasp)
STATE       n/a (no planner)
```

What happens: a reach target appears in front of the robot and the right hand moves to it while the legs keep balance. The target is a marker drawn by the environment, not by the overlay. Camera `medium-right`.

<!-- TODO(#16): result file, seed and commit of the S1 clip -->
<!-- TODO(#15): the reach target marker in the composite environment -->

### Shot 05: S2, grasp and lift

```
LOWER BODY  own policy v1
ARM         DifferentialIK (right), held posture (left)
HAND        Dex3SynergyHand
STATE       pre_grasp
```

What happens: the hand closes on the rim of the plate and lifts it clear of the counter. STATE steps through `pre_grasp`, `grasp` and `lift`. Camera `hand-close`.

<!-- TODO(#20): result file, seed and commit of the S2 clip with the balance policy active -->

### Shot 06: S3, place in the rack

```
LOWER BODY  pelvis fixed (no policy)
ARM         DifferentialIK (right), held posture (left)
HAND        Dex3SynergyHand
STATE       transport
```

What happens: the plate held in the hand is carried over the rack, aligned with a slot, inserted and released. STATE steps through `transport`, `align`, `insert` and `release`. The label says the pelvis is fixed because the scenario is run that way; the shot is not presented as balance. Camera `slot-side`.

<!-- TODO(#21): result file, seed and commit of the fixed-base S3 clip; GAP, no issue schedules S3 with the balance policy, so the label stays "pelvis fixed" unless such a file exists -->

### Shot 07: S4, full task, first success

```
LOWER BODY  own policy v1
ARM         DifferentialIK (right), held posture (left)
HAND        Dex3SynergyHand
STATE       idle
```

What happens: the whole task from a random plate pose to the released plate, at real time if the trial fits the slot, otherwise sped up by a whole factor shown in the corner. STATE steps through `idle`, `pre_grasp`, `grasp`, `lift`, `transport`, `align`, `insert`, `release` and `done`. Camera `wide`.

<!-- TODO(#22): result file, seed and commit of the first successful S4 trial in ascending seed order -->

### Shot 08: S4, second success

```
LOWER BODY  own policy v1
ARM         DifferentialIK (right), held posture (left)
HAND        Dex3SynergyHand
STATE       idle
```

What happens: the second successful trial in ascending seed order, sped up by a whole factor if needed. STATE as in shot 07. Camera `wide`.

<!-- TODO(#22): seed of the second successful S4 trial -->

### Shot 09: S4, third success

```
LOWER BODY  own policy v1
ARM         DifferentialIK (right), held posture (left)
HAND        Dex3SynergyHand
STATE       idle
```

What happens: the third successful trial in ascending seed order, sped up by a whole factor if needed. STATE as in shot 07. Camera `wide`.

<!-- TODO(#22): seed of the third successful S4 trial -->

### Shot 10: S4, failure with the most frequent tag

```
LOWER BODY  own policy v1
ARM         DifferentialIK (right), held posture (left)
HAND        Dex3SynergyHand
STATE       idle
FAILURE: <tag>
```

What happens: the first trial in ascending seed order that carries the most frequent failure tag in the S4 result file. The clip ends one second after the failure banner appears. `<tag>` is one of the seven failure tags of [evaluation.md](evaluation.md), read from the file. Camera `wide`, real time.

<!-- TODO(#22): the failure histogram that ranks the tags, and the seed with the most frequent one -->

### Shot 11: S4, failure with the second most frequent tag

```
LOWER BODY  own policy v1
ARM         DifferentialIK (right), held posture (left)
HAND        Dex3SynergyHand
STATE       idle
FAILURE: <tag>
```

What happens: as shot 10, for the second most frequent failure tag. If fewer than two tags occur, the shot is dropped and its seconds go to shot 13, which keeps the total inside the limit.

<!-- TODO(#22): the seed with the second most frequent tag -->

### Shot 12: S4, failure with the third most frequent tag

```
LOWER BODY  own policy v1
ARM         DifferentialIK (right), held posture (left)
HAND        Dex3SynergyHand
STATE       idle
FAILURE: <tag>
```

What happens: as shot 10, for the third most frequent failure tag. If fewer than three tags occur, the shot is dropped and its seconds go to shot 13.

<!-- TODO(#22): the seed with the third most frequent tag -->

### Shot 13: closing card

```
Measured with scripts/evaluate.py
Every number in the report traces to a file in outputs/eval
github.com/haianhltr/kitchenbot-ai
```

What happens: the success-rate figure, unchanged, as produced by `scripts/plot_results.py` from the same result files, with the three lines above under it and the commit in the corner. The card makes no claim of its own; the numbers are in the figure.

<!-- TODO(#24): the figure outputs/figures/success_rates.png from the final result files -->
<!-- TODO(#26): the release or repository address printed on the card, once the release exists; keep the second line only if the traceability check of the report passed -->

## Camera presets

| Camera | Frames | Moves |
| --- | --- | --- |
| `wide` | The robot, the counter and the open dishwasher in one view from the front at three quarters | Fixed in the world |
| `split-wide` | Two `wide` views side by side, one per controller | Fixed in the world |
| `medium-right` | The torso, the right arm and the reach target from the front right at hand height | Fixed in the world |
| `hand-close` | The right palm and the plate at a short distance | Follows the right palm |
| `slot-side` | The plate and the rack slot it enters, seen from the side at rack height | Fixed in the world |

<!-- TODO(#25): the pose of each camera as numbers in a committed file under configs/eval, so that the video can be recorded again -->

## Overlay specification

This is what `scripts/record_demo.py` (#25) implements. All sizes are fractions of the frame height H, so the overlay scales with the resolution; the viewer is set to 1920 by 1080.

**Panel, top left.** Anchored 0.02 H from the left and top edges. Four rows named LOWER BODY, ARM, HAND and STATE, row pitch 0.045 H, text height 0.03 H, a monospaced font (DejaVu Sans Mono, which ships with matplotlib, and Pillow, which matplotlib installs, so no new dependency). Black background at 60 percent opacity with 0.01 H padding. Row names in light grey `#BBBBBB`, values in white `#FFFFFF`, except the LOWER BODY value, which is coloured and carries a vertical bar 0.004 H wide at the left edge of the panel in the same colour.

**Colours of the lower-body value.** `own policy v1` in sky blue `#56B4E9`, `Agile baseline` in orange `#E69F00`, `pelvis fixed (no policy)` in light grey `#BBBBBB`. The pair is colour-blind safe; the words always carry the meaning, the colour never carries it alone.

**State row.** Changes on the frame where the planner enters the new state, with no fade and no smoothing.

**Corner, bottom right.** Anchored 0.02 H from the right and bottom edges, one line of text height 0.02 H on the same 60 percent black strip: scenario, split, seed, commit and playback speed, in the form `S4 unseen  seed <seed>  commit <7 hex digits>  <speed>x`. The commit is the first seven digits of `git_commit` in the run block, followed by `+dirty` if `git_dirty` is true. The speed is `1x` for real time and is always shown.

**Push flash.** In scenario S0 the word `PUSH` appears at the top centre, text height 0.04 H, vermillion `#D55E00`, for half a second after each external push event, without the force value.

**Outcome banner.** Bottom centre, 0.08 H above the bottom edge, text height 0.05 H, white on 90 percent opaque colour, shown from the frame on which the trial ends until the last frame of the clip. `SUCCESS` on bluish green `#009E73`, or `FAILURE: <tag>` on vermillion `#D55E00`, where `<tag>` is one of `fall`, `ik_infeasible`, `grasp_miss`, `drop`, `rack_collision`, `misplace` or `timeout`, read from the result file for the seed and never typed.

**Split shot.** Each half has its own panel, as above, and a top rule 0.006 H high in the colour of its lower-body value. The corner strip is shared and sits at the bottom right of the full frame.

**Cards.** Background `#101418`, text centred, the first line at 0.07 H in white and the others at 0.035 H in light grey. The closing card places the figure within 0.7 H of height, unscaled in content.

**Behaviour.** The overlay is burned into every frame by `record_demo.py` from live state, not added as a subtitle track. Values come from the run block of the result file and from the environment on that frame. If any value cannot be read the recording stops with an error instead of drawing a blank. The script refuses synthetic result files and, for the final cut, files whose `git_dirty` is true; rehearsal clips may use them and then show `+dirty` in the corner. Text that is not derived from these sources is not allowed on screen, apart from the card text above.

<!-- TODO(#25): the ffmpeg steps for the side-by-side join, the whole-factor speed-up and the final concatenation -->
