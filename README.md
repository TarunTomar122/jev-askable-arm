# Askable arm

Zero-shot robot tasks with the [TypeSafe](https://typesafe.ai) **Jev** API.

Give a goal in plain English. Jev chains a set of hardcoded primitives. The arm does the thing.

No training. No demos. No policy.

```
"pick up the red cube"
        │
        ▼
┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  scene + goal    │────▶│  Jev (closed    │────▶│  execute     │
│  (xyz, gripper,  │     │   menu of ~30   │     │  hover /     │
│   objects)       │     │   primitives)   │     │  descend /   │
└──────────────────┘     └─────────────────┘     │  close / …   │
        ▲                                        └──────────────┘
        └────────────── repeat until the goal is done ───────────┘
```

Jev never outputs torques or a trajectory. It picks **one** primitive per step. Python runs that move. Repeat.

## What it can do

On [ManiSkill](https://github.com/haosulab/ManiSkill) with a Franka Panda (`PickCube-v1`, `ResetButton-v7`), seed 0:

| Goal | Chain | Picks |
| --- | --- | ---: |
| Pick up the red cube | `hover_over → descend_onto → close_gripper` | 3 |
| Place it on the green target | grasp, lift, hover target, press onto it | 7 |
| Press blue, then red | hover/press blue, hover/press red | 4 |

~1 second per Jev pick.

Type anything that the primitive set can cover, e.g.

- `Move the fingertip upward by at least 4 centimeters.`
- `Press the red button first, then press the blue button.`

## How it works

1. **Catalog.** ~30 mutually exclusive skills: `open_gripper`, `up_1cm`, `hover_over`, `descend_onto`, `press_onto`, `done`, …
2. **State.** Privileged sim xyz, gripper open/closed, object distances, recent chain. No images go to Jev.
3. **Choose.** One System One request: which primitive, and which object if it needs a target.
4. **Act.** A small PD executor drives `pd_ee_delta_pos` for that skill, then we ask again.

The “policy” is whatever sequence Jev strings together.

## Setup

macOS CPU (MoltenVK) is what this was run on.

```bash
git clone https://github.com/TarunTomar122/jev-askable-arm.git
cd jev-askable-arm
uv sync
cp .env.example .env   # put your JEV_API_KEY in here
source scripts/vulkan_env.sh
```

Get a key at [typesafe.ai](https://typesafe.ai). `JEV_API_KEY` and `TYPESAFE_API_KEY` both work.

On macOS you also need [MoltenVK](https://github.com/KhronosGroup/MoltenVK) so SAPIEN can render:

```bash
brew install molten-vk
```

## Run

**Talk to the arm** (click, speak, live captions):

```bash
source scripts/vulkan_env.sh
uv run python scripts/ask_arm_server.py
```

Open http://127.0.0.1:8765 — click once, talk. Browser speech recognition streams the words; when you pause, that sentence is the next goal. Keep talking for the next one.

Typed lab UI: http://127.0.0.1:8765/lab

**Headless suite** (motion → reach → grasp/place → buttons):

```bash
source scripts/vulkan_env.sh
uv run python scripts/run_jev_ask.py --seed 0
```

Results write to `outputs/jev_robotics/ask_arm/results.json`.

## Layout

```
jev_robotics/primitives.py   # catalog, executor, task list
jev_robotics/ask_session.py  # live loop for the browser
jev_robotics/common.py       # Jev HTTP client
scripts/ask_arm_server.py    # local UI
scripts/run_jev_ask.py       # batch eval
web/ask_arm/demo.html        # talk-to-the-arm view
web/ask_arm/index.html       # typed lab
reset_env/                   # two-button ManiSkill env
```

## Notes

- Click-to-talk uses the browser’s on-device speech recognition. Nothing is sent to OpenAI.
- This is **sim**, not a real cell. Perception is privileged state, not a camera model.
- Button episodes are **not** killed on the first press in this repo, so `red then blue` can finish.
- The primitive set is the whole skill library. New tasks = new English, or add a primitive if the catalog cannot express it.
