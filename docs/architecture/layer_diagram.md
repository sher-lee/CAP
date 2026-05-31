# CAP — Multi-Layer Software Architecture

CAP (Cytology Analysis Project) is a veterinary ear-cytology automation system
that runs entirely on an NVIDIA Jetson Orin Nano, offline. The software is
organized into **nine functional layers** plus a cross-cutting orchestration
tier. Every object that crosses a layer boundary is a formal dataclass defined
in [`common/dataclasses.py`](../../common/dataclasses.py) — the edge labels in
the diagram below are those exact contracts.

## Layer block diagram

```mermaid
flowchart TD
    %% ---------- Cross-cutting / orchestration ----------
    subgraph X["Cross-cutting / Orchestration"]
        direction LR
        CFG["config<br/>cap_config.yaml · config_loader"]
        CMN["common<br/>dataclasses (contracts) · backend selector (real/sim) · logging"]
        WRK["workers<br/>scan_worker · inference_worker · report_worker (QThreads)"]
    end

    %% ---------- Layers ----------
    L6["<b>Layer 6 — UI</b> (PyQt)<br/>main_window · signals · screens · widgets<br/>polygon_tool · tile_viewer"]
    L1["<b>Layer 1 — Hardware Abstraction</b><br/>motor_controller · safety_system · oil_safety<br/>preliminary_focus · per_field_autofocus<br/>coordinate_mapper · scan_region · z_cycle_generator · sim/"]
    L2["<b>Layer 2 — Acquisition</b><br/>camera_interface · capture_sequencer<br/>continuous_scan_controller · frame_grouper · frame_tagger<br/>focus_stacker · pipeline · sim/"]
    L3["<b>Layer 3 — Image Processing</b><br/>debayer · denoise · normalize · resize · pipeline"]
    L4["<b>Layer 4 — AI Inference</b> (YOLOv11)<br/>model_loader · inference · postprocess<br/>aggregator · ai_disabled_mode"]
    L5[("<b>Layer 5 — Data</b><br/>db_manager · crud · schema.sql (SQLite)<br/>audit · backup · export + image filesystem")]
    L7["<b>Layer 7 — Visualization</b> (primary deliverable)<br/>stitcher · tile_builder (DZI) · canvas_placer<br/>annotations · severity · pdf_report · transfer"]
    L8["<b>Layer 8 — Metrics</b><br/>clinic_dashboard · ai_metrics · export"]
    L9["<b>Layer 9 — Retraining</b><br/>corrections · cvat_export"]

    %% ---------- Inter-layer data contracts ----------
    L6 -->|"ScanRegion (polygon to motor steps)"| L1
    L1 -->|"FocusMapResult + per-field Z targets"| L2
    L2 -->|"motor / Z move + autofocus commands"| L1
    L2 -->|"RawFrame (Bayer + motor pos + Z idx)"| L3
    L2 -->|"StackedField (composite + sharpness_map + z_distribution)"| L5
    L2 -->|"StackedField (composite)"| L7
    L2 -->|"ScanProgress (Qt signal)"| L6
    L3 -->|"ProcessedFrame (RGB + focus_score)"| L4
    L4 -->|"Detection[] (bbox, class, conf)"| L5
    L4 -->|"SlideResults (counts, severity, density, summary)"| L5
    L4 -->|"SlideResults"| L7
    L5 -->|"images + SlideResults"| L7
    L5 -->|"historical scan queries"| L8
    L5 -->|"corrections + detections"| L9
    L7 -->|"stitched WSI · annotated tiles · PDF"| L6
    L9 -.->|"retrained YOLOv11 weights (offline, human-in-loop)"| L4

    %% ---------- Orchestration wiring ----------
    CMN -. "class refs: real vs simulation" .-> L1
    CMN -. "class refs: real vs simulation" .-> L2
    WRK -. "Qt signals" .-> L6

    %% ---------- Styling ----------
    classDef layer fill:#eef4fb,stroke:#3b6ea5,stroke-width:1px,color:#11304f;
    classDef data fill:#fdf3e7,stroke:#c8862b,stroke-width:1px,color:#5a3b0a;
    classDef cross fill:#f0f0f0,stroke:#888,stroke-dasharray:3 3,color:#333;
    class L1,L2,L3,L4,L6,L7,L8,L9 layer;
    class L5 data;
    class CFG,CMN,WRK cross;
```

## Interface contract summary

| From → To | Interface (payload) | Direction / nature |
|-----------|--------------------|--------------------|
| L6 → L1 | `ScanRegion` (polygon in motor steps) | Command — defines scan area |
| L1 → L2 | `FocusMapResult` (focal-surface polynomial) + per-field Z targets | Data — focus prediction |
| L2 → L1 | motor/Z move + autofocus calls | Command (via `motor_controller`) |
| L2 → L3 | `RawFrame` (Bayer + motor pos + Z index) | Stream, per Z-depth, via queue |
| L2 → L5 / L7 | `StackedField` (composite + `sharpness_map` + `z_distribution`) | Data — all-in-focus field |
| L2 → L6 | `ScanProgress` | Qt signal — live progress/ETA |
| L3 → L4 | `ProcessedFrame` (RGB + `focus_score`) | Stream — AI-ready image |
| L4 → L5 | `Detection[]` (bbox, class, confidence) | Data — per-field detections |
| L4 → L5 / L7 | `SlideResults` (counts, severity, density, summary) | Data — slide-level aggregate |
| L5 → L7 | images + `SlideResults` | Query — for stitching/PDF |
| L5 → L8 | historical scan/detection queries | Query — analytics |
| L5 → L9 | corrections + detections | Query — dataset curation |
| L7 → L6 | stitched WSI, annotated tiles, PDF | Display / deliverable |
| L9 ⤏ L4 | retrained YOLOv11 weights | **Offline loop** (human-in-loop) |
| Cross-cut | `common.dataclasses`, `backend` (real/sim), `config`, `workers`, Qt `signals` | Orchestration |

## Design notes

- **Z-stacking** (`z_cycle_generator` → `focus_stacker`) captures multiple focal
  planes per field for organism coverage across depths and picks the sharpest
  block per region into `StackedField.sharpness_map` / `z_distribution`. This is
  a discrete Layer 2 stage feeding both storage and visualization.
- **Imaging-first**: Layer 4 has an explicit `ai_disabled_mode`; Layers 5/7 still
  produce the stitched image and PDF when no model is loaded, matching the
  product's "value through imaging alone" priority.
- **Layer 1 is the only hardware owner**, and `backend.py` swaps Layers 1/2 for
  their `sim/` implementations — so Layers 3–9 behave identically in simulation
  and on real hardware.
