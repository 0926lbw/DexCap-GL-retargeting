# MuJoCo Four Finger Hand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a first-pass MuJoCo MJCF model for the human four-finger skeleton, using the repo's existing human-hand geometry and joint limits.

**Architecture:** Add a small pure-Python MJCF exporter that reads `hand_reconstruction.human_hand_params.default_params()` and writes a four-finger-only XML file. The XML contains palm, index/middle/ring/pinky kinematic chains, hinge joints, keypoint sites, position actuators for active joints, and equality coupling for passive DIP joints.

**Tech Stack:** Python standard library `xml.etree.ElementTree`, existing `hand_reconstruction` params, `unittest`, MuJoCo MJCF XML.

---

### Task 1: Add MJCF Exporter Tests

**Files:**
- Create: `tests/test_mujoco_export.py`
- Create later: `hand_reconstruction/mujoco_export.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mujoco_export.py` with tests that call:

```python
from hand_reconstruction.human_hand_params import default_params
from hand_reconstruction.mujoco_export import build_four_finger_mjcf
```

The tests must verify:

- root element is `<mujoco model="human_four_finger_hand">`,
- there are no `thumb_*` bodies, joints, actuators, sites, or equalities,
- all four fingers have `mcp`, `pip`, `dip`, and `tip` sites,
- each finger has hinge joints `mcp_abd`, `mcp_flex`, `pip`, `dip`,
- `mcp_flex`, `pip`, and `dip` use the repo flex axis,
- `mcp_abd` uses the repo abduction axis,
- active joints have position actuators except `dip`,
- each `dip` is coupled to its own `pip` with `polycoef="0 0.6 0 0 0"`.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
python3 -m unittest tests.test_mujoco_export
```

Expected: fail with `ModuleNotFoundError: No module named 'hand_reconstruction.mujoco_export'`.

### Task 2: Implement MJCF Exporter

**Files:**
- Create: `hand_reconstruction/mujoco_export.py`

- [ ] **Step 1: Implement `build_four_finger_mjcf`**

Create `build_four_finger_mjcf(params=None, hand="right") -> str`.

Required XML structure:

```text
mujoco model="human_four_finger_hand"
  compiler angle="degree" autolimits="true"
  option timestep="0.002" gravity="0 0 0"
  default
  worldbody
    body name="wrist"
      geom name="palm"
      site name="wrist"
      body name="{finger}_mcp" pos="{attach}" euler="0 0 {yaw_deg}"
        joint name="{finger}_mcp_abd" axis="{abd_axis}" range="{mcp_abd_deg}"
        body name="{finger}_mcp_flex_body"
          joint name="{finger}_mcp_flex" axis="{flex_axis}" range="{mcp_flex_deg}"
          geom name="{finger}_prox_geom" fromto="0 0 0 0 {prox} 0"
          site name="{finger}_mcp"
          body name="{finger}_pip" pos="0 {prox} 0"
            joint name="{finger}_pip" axis="{flex_axis}" range="{pip_deg}"
            geom name="{finger}_mid_geom" fromto="0 0 0 0 {mid} 0"
            site name="{finger}_pip"
            body name="{finger}_dip" pos="0 {mid} 0"
              joint name="{finger}_dip" axis="{flex_axis}" range="{dip_deg}"
              geom name="{finger}_dist_geom" fromto="0 0 0 0 {dist} 0"
              site name="{finger}_dip"
              body name="{finger}_tip" pos="0 {dist} 0"
                site name="{finger}_tip"
  equality
    joint name="{finger}_dip_coupling" joint1="{finger}_dip" joint2="{finger}_pip" polycoef="0 0.6 0 0 0"
  actuator
    position name="{finger}_mcp_abd_act" joint="{finger}_mcp_abd"
    position name="{finger}_mcp_flex_act" joint="{finger}_mcp_flex"
    position name="{finger}_pip_act" joint="{finger}_pip"
```

- [ ] **Step 2: Run tests and confirm GREEN**

Run:

```bash
python3 -m unittest tests.test_mujoco_export
```

Expected: all tests pass.

### Task 3: Add Export Script and Generated XML

**Files:**
- Create: `scripts/export_mujoco_four_finger_hand.py`
- Create: `mujoco/human_four_finger_hand.xml`

- [ ] **Step 1: Write script test**

Extend `tests/test_mujoco_export.py` with a subprocess-free test that imports:

```python
from scripts.export_mujoco_four_finger_hand import export_four_finger_mjcf
```

Call it with a temporary path and verify the file exists and parses as XML.

- [ ] **Step 2: Implement script**

Create a script with:

```python
def export_four_finger_mjcf(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_four_finger_mjcf(), encoding="utf-8")
    return output_path
```

The CLI should default to `mujoco/human_four_finger_hand.xml`.

- [ ] **Step 3: Generate XML**

Run:

```bash
python3 scripts/export_mujoco_four_finger_hand.py
```

Expected: writes `mujoco/human_four_finger_hand.xml`.

### Task 4: Environment and Verification

**Files:**
- Modify: `docs/current-memory.md`

- [ ] **Step 1: Create conda environment**

Run:

```bash
conda create -n dexcap_re python=3.11 -y
```

If network or package resolution fails inside sandbox, rerun with approved escalation.

- [ ] **Step 2: Verify**

Run:

```bash
python3 -m unittest tests.test_mujoco_export tests.test_human_hand_builder tests.test_retargeting
python3 -m py_compile hand_reconstruction/mujoco_export.py scripts/export_mujoco_four_finger_hand.py tests/test_mujoco_export.py
python3 scripts/export_mujoco_four_finger_hand.py
```

Expected: tests pass, compile succeeds, XML regenerates.

- [ ] **Step 3: Update memory**

Add current notes to `docs/current-memory.md`:

- MuJoCo first pass is four-finger only.
- `mujoco/human_four_finger_hand.xml` is generated from existing `human_hand_params.py`.
- DIP is passive via equality coupling to PIP.
- Contact is disabled in first pass.
