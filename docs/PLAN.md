# IR Drone Tag: Adversarial RL with Sim-to-Real

Master plan for a group project. Two quadrotors play infrared laser tag against each other. Each learns its own attack/defend policy through self-play and learns to predict its opponent. Policies are trained in simulation and transferred to real Crazyflie drones.

This file is shared by every teammate. Put it in your Claude Project's knowledge, then start one chat per workstream (see "How to use this file").

---

## 0. Instructions for Claude (read first, every chat)

You are helping a team member work on ONE workstream of this project. Follow these rules in every chat that uses this file.

1. **Identify the workstream.** At the start of a chat, if the user has not said which workstream (WS0 to WS6) they are working on, ask. Stay inside that workstream's scope. If a task belongs to another workstream, say so and describe what to hand off.
2. **Load current state before working.** Ask the user to paste (or check project knowledge for) these files from the repo:
   - `docs/status/WS<n>.md` for this workstream
   - `docs/contracts.md`
   - the status files of any upstream workstreams this one depends on (see the dependency table in section 5)

   If none exist yet, this is the first session. Create the status file from the template in section 9.
3. **The repo is the source of truth, not chat memory.** Teammates use separate Claude accounts, and Claude memory is not shared between them. All durable state lives in the repo: status files, contracts, decision log, configs. Chat or Claude memory is only a convenience for the individual user.
4. **Keep the status file updated.** At the end of every work session, after any milestone, or whenever the user says "wrap up" or "status", output the complete updated `docs/status/WS<n>.md` in one code block, ready to commit. Never output a partial diff of it. Also update your own memory (if this Claude has memory enabled) with a one-line summary of where this workstream stands.
5. **Contracts are shared, and you never change them silently.** `docs/contracts.md` defines the interfaces between workstreams: observation and action specs, config schemas, frames, units, rates, file formats. If work requires a change:
   - stop and flag it as **CONTRACT CHANGE PROPOSAL**, giving what changes, why, and which workstreams are affected
   - let the user agree it with the team
   - after agreement, output the updated `docs/contracts.md` in full and bump its version number
6. **Log decisions.** For any non-obvious design choice, output a line for `docs/decisions.md` in this format: `YYYY-MM-DD | WS<n> | decision | reason | alternatives considered`.
7. **Reason before implementing.** For anything non-trivial, give a short plan and the key tradeoffs before writing code. Keep explanations terse.
8. **Verify library APIs.** Crazyflow, Crazyswarm2, cflib and the Crazyflie firmware evolve. Check current docs or source before relying on a function signature. Do not invent APIs.
9. **Safety rules in section 8 are non-negotiable.** Never suggest a visible laser. Never suggest removing the safety filter or geofence for real flights.
10. **Code conventions:**
    - Python 3.11+
    - SI units everywhere
    - Type hints on all functions
    - Configs in YAML under `config/`, never hardcoded constants
    - Tests in `tests/` for anything other workstreams depend on
    - Small, reviewable commits

---

## 1. How to use this file (for humans)

1. **One-time setup (WS0 owner):** create the GitHub repo using the layout in section 4. Commit this file as `docs/PLAN.md`, plus empty `docs/contracts.md`, `docs/decisions.md` and `docs/status/`.
2. **Each teammate:** create a Claude Project and add this file to its knowledge. Optionally also add `docs/contracts.md` and the status files you care about, and refresh them when they change.
3. **Start one chat per workstream you own.** Name it like `WS2 RL Training`. First message:

   ```
   I'm working on WS<n> (<name>). Here are the current files:
   <paste docs/status/WS<n>.md>
   <paste docs/contracts.md>
   <paste upstream status files if relevant>
   Today I want to: <goal>
   ```

4. **End every session** with "wrap up". Commit the status file Claude outputs, along with any decision log lines.
5. **When a chat gets long,** start a fresh chat for the same workstream and paste the latest status file. The status file is designed to make this lossless.
6. **Weekly sync:** each owner skims the other status files. Contract change proposals get agreed here, or async in the group chat.

---

## 2. Project definition

### Research question
Does explicit opponent modeling (predicting the opponent's future motion) produce policies that are more robust against unseen opponents than implicit memory alone? Does that advantage survive transfer to real drones?

### The game (v0 rules, tunable via config)
- **Players:** two drones in a bounded indoor arena. The game is symmetric: each drone attacks and defends at the same time.
- **Emitter:** each drone has a fixed, forward-facing IR emitter. To aim, a drone must point its body at the target, so yaw is part of the action.
- **Hit:**
  - the target is inside the shooter's emitter cone (half-angle `cone_half_angle`)
  - the target is within `max_range`
  - there is clear line of sight
  - all of the above hold continuously for `dwell_time` seconds
  - the hit is then registered with probability from the calibrated detection model `p_detect(distance, incidence_angle)`
- **Health and episode end:** each drone has `hp` hit points. A hit costs 1 hp, then a `hit_cooldown` applies before that shooter can score again. The episode ends when a drone reaches 0 hp, at `max_episode_time`, or on a crash or geofence violation.
- **Reward (per drone):**
  - +1 per hit dealt
  - −1 per hit taken
  - −5 for a crash or geofence violation
  - small time penalty optional
  - aim-angle shaping only if learning stalls, and it must be logged in `decisions.md`
- **v1 options (later):** a discrete fire action with ammo; an asymmetric mode (attacker vs. defender guarding a zone); obstacles that block line of sight.

### Policy variants to compare
| ID | Architecture | Purpose |
|---|---|---|
| A | MLP, no memory | Baseline |
| B | GRU | Implicit opponent modeling |
| C | GRU + auxiliary head predicting opponent relative position at t+1..t+H (aux loss only) | Does prediction as a training signal help? |
| D | C + prediction fed into the policy input | Does using the prediction explicitly help? |

### Evaluation metrics
- Win rate and hit differential, measured against:
  - the self-play pool
  - held-out opponents: scripted heuristics, plus policies trained from other random seeds
- Opponent prediction error (variants C and D) over the horizon.
- Exploitability: train a best-response agent against each frozen policy and report its win rate.
- Sim-to-real gap: the same evaluation matrix on hardware, with fewer episodes.
- IR model fidelity: geometric hits computed from mocap poses vs. hits the IR receivers actually registered.

---

## 3. Technical stack

| Layer | Choice | Notes |
|---|---|---|
| Simulator | Crazyflow (JAX + MuJoCo) | Identified Crazyflie dynamics, domain randomization, multi-drone, vectorized |
| Hit model | Custom pure JAX function inserted into the Crazyflow pipeline | Cone, line of sight, dwell timer, `p_detect` table |
| RL | PPO written in JAX (PureJaxRL style), fully jitted | Self-play league with a checkpoint pool |
| Dev machine | macOS on Apple Silicon, CPU JAX | Development, debugging, short runs |
| Training compute | CUDA GPU (lab, campus cluster or cloud) | Large self-play runs |
| Drones | Crazyflie 2.1, or Brushless if payload is tight | Velocity and yaw-rate setpoints to the onboard controller |
| Positioning | Mocap if available, else Lighthouse deck | Ground-truth pose for policy observations |
| Flight software | ROS 2 + Crazyswarm2 on an Ubuntu machine | `cflib` on macOS for early single-drone tests |
| IR hardware | 38 kHz modulated IR LED with collimating tube or lens; TSOP-style receivers; custom deck on a Bitcraze prototyping deck | Eye-safe, encodes the shooter's ID |
| Firmware | Crazyflie out-of-tree app or deck driver | Drives the emitter and decodes hits; hits reported via the log framework |
| Policy deployment | Weights exported from JAX to numpy; inference in a Python ROS 2 node | GRU hidden state carried across steps |

---

## 4. Repo layout

```
drone-tag/
  README.md
  docs/
    PLAN.md              # this file
    contracts.md         # interface specs, versioned
    decisions.md         # decision log
    status/WS0.md .. WS6.md
  config/
    arena.yaml           # bounds, geofence, safety margins
    drone.yaml           # mass, limits, identified dynamics params
    hit_model.yaml       # cone, range, dwell, p_detect table, receiver layout
    randomization.yaml   # domain randomization ranges
    train/*.yaml         # experiment configs
  sim/                   # env, game logic, hit model, safety filter, scripted opponents
  train/                 # PPO, networks, league/self-play, checkpointing
  eval/                  # eval matrix, exploitability, plots
  sysid/                 # system ID scripts, fitted params
  hardware/
    ir_deck/             # schematic, PCB, BOM
    firmware/            # emitter + receiver decoding app
    bench/               # bench test scripts and raw data
  flight/                # ROS 2 packages: policy node, safety filter, hit logger, virtual opponent
  experiments/           # real flight logs, one folder per session
  tests/
```

---

## 5. Workstreams

Each workstream gets its own chat. One person can own several. Suggested split for a 3 to 4 person team:
- Person 1: WS1 + WS3
- Person 2: WS2
- Person 3: WS4
- Person 4: WS5 + WS6

WS0 is shared at the start.

### Dependency table
| Workstream | Depends on | Unblocks |
|---|---|---|
| WS0 Repo & Infra | none | everything |
| WS1 Sim Environment | WS0 | WS2, WS3, WS5 |
| WS2 RL Training | WS1 | WS3, WS6 |
| WS3 Evaluation | WS1, WS2 | WS6 |
| WS4 IR Hardware | WS0 | WS1 (hit model calibration), WS5, WS6 |
| WS5 Flight Stack & Sim-to-Real | WS0, WS4 (payload config) | WS1 (identified dynamics), WS6 |
| WS6 Real-World Experiments & Write-up | all | none |

---

### WS0: Repo & Infrastructure
**Goal:** everyone can clone, install, run tests and train on a GPU within one command each.

**Tasks**
1. Create the repo layout above, a README with setup steps, and a Python env (pixi or conda) that pins JAX, Crazyflow and the other dependencies.
2. Set up CI: lint (ruff), type check, and run `tests/` on push.
3. Write `docs/contracts.md` v0.1 from section 6 of this plan.
4. Secure GPU compute: check lab, campus cluster and cloud options. Document how to launch a training job.
5. Set up experiment tracking (Weights & Biases or TensorBoard) and a naming convention for runs and checkpoints.

**Done when:** a fresh clone installs on macOS and on the GPU machine, `pytest` passes, and a dummy training job logs metrics remotely.

---

### WS1: Simulation Environment
**Goal:** a fast, vectorized two-drone tag environment in Crazyflow whose observations, actions, rates and safety behavior match the real system exactly.

**Tasks**
1. **Crazyflow familiarization:** spawn 2 drones per world and N worlds. Confirm the control mode accepts velocity + yaw-rate setpoints at 50 Hz with the physics at 500 Hz, and record what the API actually supports.
2. **Arena:** bounds and geofence from `config/arena.yaml`, and a termination on geofence violation.
3. **Hit model** (pure JAX):
   - cone check, range check, line of sight (trivial without obstacles)
   - a dwell timer per shooter-target pair
   - a cooldown
   - `p_detect` lookup from `config/hit_model.yaml`; start with a placeholder table and swap in WS4 bench data later
   - optionally, which receiver registered the hit (body-shadowing model)
4. **Game logic:** hp, reward, termination, episode timer, and per-agent ego-centric observations per the contract.
5. **Safety filter, identical to the real one:** clamp velocity, enforce the geofence margin, enforce minimum separation and no vertical stacking (downwash). The policy must train with the same filter that runs on hardware.
6. **Domain randomization** from `config/randomization.yaml`:
   - mass and thrust scale
   - latency, as a buffer of delayed observations and actions
   - observation noise
   - battery sag, as a thrust decay over the episode
7. **Velocity estimation matching real:** if real velocity comes from differencing mocap poses, compute it the same way in sim.
8. **Scripted opponents:**
   - pure-pursuit aimer
   - random-jink evader
   - orbit-and-shoot
   - stationary target (for debugging)
9. **Visualization:** a MuJoCo viewer or matplotlib replay of trajectories, emitter cones and hits.
10. **Tests:**
    - the hit model on hand-constructed geometric cases
    - dwell and cooldown timing
    - safety filter clamping
    - determinism under a fixed seed

**Done when:** scripted vs. scripted produces sensible hit rates, the replay video looks right, throughput is measured and recorded, and all tests pass.

**Pitfalls**
- Never include anything in observations that real hardware cannot provide, such as the opponent's actions or the true hidden state.
- Keep everything jittable: no Python control flow on traced values.

---

### WS2: RL Training
**Goal:** trained policies for variants A to D via self-play, with reproducible configs.

**Tasks**
1. **JAX PPO, fully jitted with the environment.** Validate it on a single-drone task first: hover, then reach a target.
2. **Single-agent sanity check:** train against a frozen scripted opponent. The learned policy must beat the scripted one.
3. **Networks:**
   - A: MLP
   - B: GRU
   - C: GRU + prediction head (predicts opponent relative position at t+1..t+H, H around 10 steps; auxiliary MSE loss)
   - D: C with the prediction concatenated into the policy input; decide whether to stop gradients and log the choice
4. **Self-play league:**
   - a pool of past checkpoints
   - opponent sampling weighted toward opponents the current policy struggles against (prioritized fictitious self-play)
   - periodic snapshotting
   - Elo or TrueSkill tracking across the pool
5. **Train each variant across at least 3 seeds.** Policies from other seeds serve as held-out opponents for WS3.
6. **Export trained weights** to a framework-free format (npz) plus a reference numpy inference function. WS5 runs this on hardware, so add a test that numpy inference matches JAX inference.
7. **Re-train** once WS5 delivers identified dynamics and WS4 delivers the calibrated hit model. Early training uses defaults.

**Done when:** all variants train stably, Elo rises over training, exported policies pass the inference-match test, and configs reproduce results.

**Pitfalls**
- **Strategy cycling:** watch league Elo and keep old checkpoints in the pool.
- **Reward hacking:** for example, both drones hiding in corners. Inspect replays every run.
- **Hyperparameter drift across variants:** keep them identical across A to D except the architecture.

---

### WS3: Evaluation & Analysis
**Goal:** the evaluation pipeline and figures that answer the research question.

**Tasks**
1. **Evaluation matrix:** every policy against every other policy and every scripted opponent, N episodes each, reporting win rate, hit differential and time-to-first-hit.
2. **Exploitability:** train a best-response agent against each frozen policy (reusing WS2 code) and report its win rate.
3. **Prediction quality** (variants C and D): error against horizon, and against opponent type.
4. **Robustness sweeps:** performance as latency, noise and mass deviate beyond the training randomization ranges.
5. **Real-data analysis tools** (used by WS6):
   - load flight logs
   - compute geometric hits from mocap poses
   - compare against IR-registered hits
   - compute sim-vs-real metric gaps
6. **Plotting:** consistent figure style and a script that regenerates every figure from saved results.

**Done when:** one command regenerates the full sim evaluation and all figures from saved checkpoints.

---

### WS4: IR Hardware
**Goal:** a light, reliable IR tag deck on the Crazyflie, plus calibrated detection data for the sim.

**Tasks**
1. **Part selection:**
   - IR LED (940 nm typical) plus a collimating tube or lens to set the beam width
   - 38 kHz TSOP-style receivers
   - a Bitcraze prototyping or breakout deck
   - Record mass and current draw for every part.
2. **Breadboard bench test, before any drone work:**
   - emitter at 38 kHz from a microcontroller
   - receivers decoding a simple ID code
   - measure detection probability against distance and against angle, for both the emitter's off-axis angle and the receiver's incidence angle
   - repeat with the mocap or Lighthouse system running, to check for IR interference
3. **Write results to `config/hit_model.yaml`** (`max_range`, `cone_half_angle`, `p_detect` table). Notify WS1 and WS2 with a status update.
4. **Deck design:**
   - one forward emitter
   - several receivers around the frame for coverage; record which receiver fired
   - minimal mass
   - Schematic, PCB (or perfboard v0) and BOM go in `hardware/ir_deck/`.
5. **Firmware:** a Crazyflie out-of-tree app or deck driver that generates the 38 kHz carrier with a timer, encodes the drone ID, decodes received codes, applies debounce and dwell logic matching the sim definition, and exposes hit events and counters through the log framework.
6. **Mounted calibration:** repeat the angle sweep with the deck on the drone, to capture body shadowing. Update `hit_model.yaml`.
7. **Freeze the payload configuration** (deck, markers, battery) and report final mass to WS5.

**Done when:** the deck reliably registers hits with correct IDs on a flying drone, the mounted calibration data is committed, and the payload config is frozen.

**Pitfalls**
- Eye safety: only IR LEDs, never laser diodes.
- Sunlight and mocap IR can cause false triggers. Test with the real lab lighting and mocap running.
- Keep the dwell logic in firmware identical to the sim definition, or the hit statistics won't match.

---

### WS5: Flight Stack & Sim-to-Real
**Goal:** safe, reliable flight of real Crazyflies under policy control, with the sim tuned to match.

**Tasks**
1. **Basic flight:**
   - Crazyswarm2 on Ubuntu with mocap or Lighthouse
   - takeoff, hover, land
   - velocity and yaw-rate setpoint control
   - start with `cflib` on macOS for single-drone tests if useful
2. **Safety layer:**
   - a geofence
   - the same safety filter as WS1 (shared logic, ported to a real-time node)
   - a command timeout that triggers hover then land
   - a hardware kill switch
   - battery voltage cutoff
3. **Latency measurement:** command a step and timestamp the onset of motion in mocap. Report the distribution of end-to-end latency to WS1 for `randomization.yaml`.
4. **System identification on the frozen payload config:**
   - fly velocity and yaw step responses
   - fit Crazyflow's identified-dynamics model
   - commit the parameters to `config/drone.yaml` and the scripts to `sysid/`
   - repeat for each physical drone if they differ
5. **Policy node:** load the exported npz policy, build observations exactly per the contract (same rate, frames and velocity estimation as sim), run inference at 50 Hz, carry GRU state, and pass outputs through the safety filter before sending setpoints.
6. **Virtual opponent mode:** one real drone plays against a simulated opponent rendered from a policy or script, with hits computed geometrically. This is the safe first step before two real drones.
7. **Logging:**
   - poses
   - commands
   - policy outputs
   - opponent predictions
   - IR hit events
   - battery

   Use one timestamped log per flight in a format WS3 can load.

**Done when:** a real drone flies a trained policy against a virtual opponent with no safety interventions over several episodes, then two real drones complete full games.

**Pitfalls**
- Downwash between the drones: enforce vertical separation in the safety filter.
- Clock sync between the mocap, the ROS machine and the drone logs.
- Battery state changes dynamics. Log voltage and swap batteries at a fixed threshold.

---

### WS6: Real-World Experiments & Write-up
**Goal:** run the evaluation matrix on hardware and produce the final report, video and repo.

**Tasks**
1. **Experiment protocol:**
   - which policy pairs to fly
   - number of episodes
   - battery swap rules
   - randomized order
   - a pre-flight checklist
2. **Run the sessions,** with one folder per session in `experiments/` containing logs plus a notes file.
3. **Analysis using WS3 tools:**
   - real win rates and hit differentials
   - sim-vs-real gaps
   - IR fidelity (geometric vs. registered hits)
   - opponent prediction error on real data
4. **Write-up:** method, results, failure cases and lessons. Format is report, paper or blog post.
5. **Demo video and a clean public README.**

**Done when:** the real evaluation matrix is complete, the figures regenerate from logs, and the write-up and video are done.

---

## 6. Contracts v0.1 (seed for `docs/contracts.md`)

**Frames and units**
- World frame: right-handed, z up, origin at the arena center on the floor.
- Units: SI (m, s, rad).
- Yaw: about +z, counterclockwise positive.

**Rates**
- Physics: 500 Hz in sim.
- Policy and control: 50 Hz in sim and on hardware.
- Actions are held constant between policy steps.

**Action (per drone, normalized to [−1, 1])**
`[vx, vy, vz, yaw_rate]` in the world frame, scaled by the limits in `config/drone.yaml`, then passed through the safety filter.

**Observation (per drone, ego-centric where noted)**
- Own state:
  - position relative to the arena center (3)
  - velocity in the world frame (3)
  - sin and cos of yaw (2)
  - yaw rate (1)
- Opponent, relative and expressed in own body-yaw frame:
  - position (3)
  - velocity (3)
  - sin and cos of relative yaw (2)
- Opponent history: the last K relative positions (K = 10 default, 3×K values).
- Game state:
  - own hp
  - opponent hp
  - own cooldown remaining
  - opponent cooldown remaining
  - time remaining (normalized)
- Arena: distance to each of the 6 bounds (6).
- Excluded: the opponent's actions, its true internal state, and anything the real system can't provide.

**Hit event (same schema in sim and real logs)**
`{t, shooter_id, target_id, receiver_index, distance, off_axis_angle, incidence_angle}`

**Policy export**
- Weights as `.npz`, plus a JSON metadata file containing:
  - architecture
  - observation normalization statistics
  - contract version
  - training config hash
- The reference numpy inference function lives in `train/export.py`.

**Configs:** `arena.yaml`, `drone.yaml`, `hit_model.yaml`, `randomization.yaml`. Any schema change follows the contract change process.

---

## 7. Phases and gates

| Phase | Rough duration | Work | Gate to pass |
|---|---|---|---|
| P0 Setup | Week 1 | WS0 repo, CI, env, compute. Order Crazyflies and IR parts immediately. | Everyone installs and runs tests |
| P1 Sim core | Weeks 1–3 | WS1 env + scripted opponents. WS2 PPO validated on single drone. WS4 breadboard bench test. | G1: scripted games look right. G2: bench IR range and angle measured; game rules still feasible. |
| P2 Learning | Weeks 3–6 | WS2 single-agent then self-play, variants A–D. WS3 eval pipeline. WS4 deck design + firmware. WS5 basic flight + latency. | G3: self-play stable, Elo rising. G4: deck works on bench and payload is frozen. |
| P3 Calibration | Weeks 6–8 | WS5 system ID on frozen payload. WS4 mounted calibration. WS1 updates configs. WS2 retrains. WS3 sim eval. | G5: sim matches real step responses and hit statistics within agreed tolerance |
| P4 Hardware | Weeks 8–11 | WS5 policy node, virtual opponent mode, then 2 real drones. | G6: full real games with no safety interventions |
| P5 Experiments | Weeks 11–13 | WS6 real evaluation matrix, analysis, write-up, video. | Final deliverables |

Adjust the durations to your semester. The order of the gates matters more than the dates.

---

## 8. Safety rules (non-negotiable)

1. IR LEDs only. No laser diodes of any kind.
2. All real flights run inside a netted or clearly bounded area. No people inside the flight volume during autonomous flight.
3. The safety filter (geofence, velocity limits, separation, timeout) is always active on hardware. The policy never sends commands directly to the drones.
4. A dedicated person holds the kill switch during every flight.
5. Follow LiPo battery handling: no charging unattended, store in a fire-safe bag, discard damaged cells.
6. Hits are registered by IR only. There is no physical contact between drones; capture and hit logic never requires contact.
7. Start every new policy in virtual-opponent mode before flying two real drones together.

---

## 9. Status file template (`docs/status/WS<n>.md`)

```markdown
# WS<n>: <name>
Owner: <name>
Last updated: <YYYY-MM-DD> by <name>
Contract version used: <x.y>

## Current state (one paragraph)
What exists and works right now.

## Done
- <milestone> (<date>), <where in repo>

## In progress
- <task>: <state, next concrete step>

## Next up
- <task>

## Blockers / needs from other workstreams
- Needs from WS<m>: <what> (status: requested/received)

## Outputs other workstreams can use
- <file/config/function>: <what it is, version>

## Key numbers
- e.g. sim throughput, measured latency, IR max range, mass of payload, Elo

## Decisions made this period
- See docs/decisions.md entries dated <dates>

## Known issues / gotchas
- <issue>
```

---

## 10. Open questions to resolve in week 1

- Is lab mocap available, or do we use Lighthouse? This affects positioning, deck stacking and the payload budget.
- What GPU compute is available, and how do we get access?
- Team size and owners for each workstream.
- How many Crazyflies (minimum 2, plus at least 1 spare), and do we need the Brushless version for payload?
- What arena size can we get? `arena.yaml` must match it.
- Who supervises flights and holds the kill switch?