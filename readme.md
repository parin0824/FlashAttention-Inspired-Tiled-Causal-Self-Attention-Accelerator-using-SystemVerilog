# FlashAttention Inspired Tiled Causal Self Attention Accelerator using SystemVerilog

> **Fixed-function FPGA/ASIC core** — computes scaled dot-product causal self-attention over a token sequence using IO-aware blocked tiling, a fully pipelined datapath, and numerically stable fixed-point softmax.

| | |
|---|---|
| **Top module** | `attention_top` |
| **Target** | Xilinx 7-series (Vivado · `.bit` + `.xsa` provided) |
| **Language** | SystemVerilog (IEEE 1800-2017) |
| **Clock domain** | Single — synchronous active-low reset |
| **Status** | Released v1.0 |

---

## Table of Contents

1. [Mathematical Background](#1-mathematical-background)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Hierarchy](#3-module-hierarchy)
4. [Tiled Algorithm — Two-Pass Flow](#4-tiled-algorithm--two-pass-flow)
5. [Controller FSM](#5-controller-fsm)
6. [Module Reference](#6-module-reference)
7. [Parameters & Data Formats](#7-parameters--data-formats)
8. [AXI-Lite Register Map](#8-axi-lite-register-map)
9. [Performance Counters](#9-performance-counters)
10. [Pipeline Optimisations](#10-pipeline-optimisations)
11. [Latency Analysis](#11-latency-analysis)
12. [File Listing & Compile Order](#12-file-listing--compile-order)
13. [Simulation & Build](#13-simulation--build)
14. [Signal Ownership Reference](#14-signal-ownership-reference)

---

## 1. Mathematical Background

For each query position `q` in a sequence of length `N`, causal self-attention is:

```
score[q][k]  =  ( Q[q] · K[k] ) >>> SCALE_SHIFT     for k ≤ q
             =  NEG_INF                               for k > q   (causal mask)

weight[q][k] =  exp( score[q][k] − max_k(score[q]) )
                ─────────────────────────────────────────────
                Σ_k  exp( score[q][k] − max_k(score[q]) )

O[q]         =  Σ_k  weight[q][k] × V[k]
```

**Design choices:**

| Decision | Choice | Reason |
|---|---|---|
| Tiling strategy | IO-aware `BR × BC` blocking | Minimises memory bandwidth; reuses loaded tiles |
| Causal masking | Sequential pipeline register post-score | Zero-latency overhead; absorbed by FSM gap |
| Softmax numerics | Row-max subtraction before exp | Prevents fixed-point overflow in exp LUT |
| Normalisation | `(exp × 0xFFFF) / row_sum` — direct divide | Avoids precision collapse on single-token rows |
| Pipeline stages | 1 register per critical path | Each long combinational path isolated per clock |
| Data format | INT8 inputs · INT32 accumulators · Q0.16 probs | Hardware-efficient; sufficient dynamic range |
| Scale shift | `>>> 2` (÷4) | Maps INT8 dot products into exp LUT input range |

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    attention_axi_wrapper                          │
│  AXI4-Lite Slave  ·  IRQ  ·  Q/K/V/O memory  ·  perf counters  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                      attention_top                         │  │
│  │                                                            │  │
│  │  ┌──────────────────┐   ┌──────────────────────────────┐  │  │
│  │  │ attention_       │   │          addr_gen            │  │  │
│  │  │ controller       │──▶│  (purely combinational)      │  │  │
│  │  │ 27-state FSM     │   │  q_tile_base · kv_tile_base  │  │  │
│  │  └────────┬─────────┘   │  global_q_idx · global_k_idx │  │  │
│  │           │ start/en    └──────────────────────────────┘  │  │
│  │           │ signals                                        │  │
│  │  ─────────┼──────────── PASS A ─────────────────────────  │  │
│  │           │                                                │  │
│  │  ┌────────▼──────┐  q_tile   ┌────────────┐              │  │
│  │  │   q_buffer    │──────────▶│            │              │  │
│  │  │  [BR][D] INT8 │           │ score_     │              │  │
│  │  └───────────────┘           │ engine     │score_tile    │  │
│  │  ┌───────────────┐  k_tile   │ (pipelined │──────────────│  │
│  │  │   kv_buffer   │──────────▶│  MAC array)│     ┌────────▼──┐│
│  │  │[BC][D] K+V    │           └────────────┘     │ mask_unit ││
│  │  └──────┬────────┘                               │(causal)   ││
│  │         │ v_tile (Pass B)                        └─────┬─────┘│
│  │         │                                             │masked │
│  │  ─────────────────────────── SOFTMAX ─────────────────│─────  │
│  │                                                        │       │
│  │  ┌─────────────────────────────────────────────────────▼────┐ │
│  │  │                    row_score_store  [N] INT32            │ │
│  │  └────────────────────────────┬─────────────────────────────┘ │
│  │                               │ row_scores[N]                 │
│  │  ┌────────────────────────────▼─────────────────────────────┐ │
│  │  │  row_max_unit          row_sum_unit          normalizer  │ │
│  │  │  5-level binary tree   exp LUT · sum acc     mul/div     │ │
│  │  │  → row_max             → exp_row[N], row_sum → prob_row  │ │
│  │  └──────────────────────────────────────────────────────────┘ │
│  │         │ row_sum    ┌─────────────┐                          │
│  │         └───────────▶│reciprocal_  │row_sum_recip             │
│  │                      │lut (unused) │ (kept, not wired to norm)│
│  │                      └─────────────┘                          │
│  │  ─────────────────────────── PASS B ──────────────────────── │
│  │                                                                │
│  │  ┌───────────────────────────────────────────────┐            │
│  │  │          weighted_sum_engine                  │            │
│  │  │  BC=4 parallel MACs · adder tree · acc[D]    │            │
│  │  │  prob_slice[BC] (sliced from prob_row in top) │            │
│  │  └──────────────────────────┬────────────────────┘            │
│  │                             │ out_row[D]                      │
│  │  ┌──────────────────────────▼────────────────────┐            │
│  │  │              output_buffer                    │            │
│  │  │  writes O_mem[global_q_idx][:]               │            │
│  │  └──────────────────────────────────────────────┘            │
│  │                                                                │
│  │  ┌──────────────────────────────────────────────────────────┐ │
│  │  │                   perf_counters                          │ │
│  │  │  cycle · score · softmax · wsum · load · stall          │ │
│  │  └──────────────────────────────────────────────────────────┘ │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Hierarchy

```
attention_axi_wrapper          ← AXI4-Lite wrapper (FPGA integration)
└── attention_top              ← Top-level integration
    ├── attention_controller   ← 27-state control FSM
    ├── addr_gen               ← Combinational index/address calculator
    ├── q_buffer               ← Query tile register file [BR][D]
    ├── kv_buffer              ← Key+Value tile register file [BC][D]
    ├── score_engine           ← BR×BC pipelined dot-product array  [Amogha]
    ├── mask_unit              ← Causal mask, 1-cycle pipeline reg   [Amogha]
    ├── row_score_store        ← N-wide score row accumulator        [Amogha]
    ├── row_max_unit           ← 5-level binary max tree             [Jainil]
    ├── row_sum_unit           ← exp LUT scan + row_sum accumulator  [Jainil]
    │   └── exp_lut            ← 9-entry LUT, pipelined (internal)
    ├── reciprocal_lut         ← 65536/row_sum, pipelined            [Jainil]
    ├── normalizer             ← 2-stage mul/div probability row     [Jainil]
    ├── weighted_sum_engine    ← BC parallel MACs + accumulator      [Jainil]
    ├── output_buffer          ← O_mem row writer                    [Parin]
    └── perf_counters          ← Cycle-accurate observability        [Parin]
```

**Ownership:**
- **Parin** — `attention_pkg`, `addr_gen`, `q_buffer`, `kv_buffer`, `output_buffer`, `perf_counters`, `attention_controller`, `attention_top`
- **Amogha** — `score_engine`, `mask_unit`, `row_score_store`
- **Jainil** — `row_max_unit`, `exp_lut`, `row_sum_unit`, `reciprocal_lut`, `normalizer`, `weighted_sum_engine`

---

## 4. Tiled Algorithm — Two-Pass Flow

The accelerator implements **IO-aware blocked tiling**: Q is processed in `BR`-row tiles; K and V are processed in `BC`-column tiles. At default parameters (N=32, D=16, BR=4, BC=4) there are 8 Q tiles and 8 KV tiles.

### High-Level Loop

```
for q_tile_idx = 0 .. NUM_Q_TILES-1:          // 8 Q tiles
  load q_tile [BR][D] from Q_mem              // 64 cycles

  for q_local_row_idx = 0 .. BR-1:            // 4 query rows per Q tile

    ── PASS A: Score collection ──────────────────────────────────
    clear row_score_store

    for kv_tile_idx = 0 .. NUM_KV_TILES-1:    // 8 KV tiles
      load kv_tile [BC][D] from K_mem+V_mem   // 64 cycles
      score_tile = (q_tile[row] · k_tile^T) >>> 2   // 258 cycles
      mask_unit: apply causal mask (NEG_INF if k > q)
      row_score_store: append BC scores at kv_tile_base

    ── SOFTMAX ───────────────────────────────────────────────────
    row_max  = max(row_scores[N])              // 5 cycles
    exp_row  = exp(row_scores - row_max)       // N+2 = 34 cycles
    row_sum  = sum(exp_row)                    //   (computed simultaneously)
    prob_row = (exp_row × 0xFFFF) / row_sum   // N+2 = 34 cycles

    ── PASS B: Output accumulation ───────────────────────────────
    for kv_tile_idx = 0 .. NUM_KV_TILES-1:    // 8 KV tiles
      reload kv_tile [BC][D] from K_mem+V_mem // 64 cycles
      prob_slice = prob_row[kv_tile_base : +BC]   (glue in attention_top)
      weighted_sum_engine:
        acc[d] += Σ_c prob_slice[c] × v_tile[c][d]
        (clear acc only on kv_tile_idx == 0)  // D+2 = 18 cycles

    output_buffer: write acc → O_mem[global_q_idx][:]  // D = 16 cycles
```

### Data Flow Diagram

```
Q_mem ──────────► q_buffer ──────────────────────────► score_engine ──┐
                  [BR][D]                                              │
K_mem ──────────► kv_buffer ─────────── k_tile ────────► score_engine  │
V_mem ──────────► kv_buffer ─┐          (Pass A)                       │
                              │                                        ▼
                              │                               mask_unit (causal)
                              │                                        │
                              │                               row_score_store[N]
                              │                                        │
                              │                               row_max_unit ────► row_max
                              │                                        │
                              │                               row_sum_unit ────► exp_row[N]
                              │                                        │         row_sum
                              │                               normalizer ──────► prob_row[N]
                              │                                        │
                              │          v_tile        prob_slice[BC]  │
                              └────────────────────────────────────────┤
                                         (Pass B)                      ▼
                                                           weighted_sum_engine
                                                                       │
                                                                out_row[D]
                                                                       │
                                                           output_buffer ──► O_mem[N][D]
```

---

## 5. Controller FSM

The `attention_controller` implements a **27-state Mealy FSM**. All state transitions are event-driven on one-cycle `done` pulses from compute submodules. All outputs default to zero; only the active state drives its output high.

```
IDLE
  │ start
  ▼
LOAD_Q_TILE ──► WAIT_Q_TILE
                    │ q_load_done
                    ▼
              ┌─► INIT_ROW ◄───────────────────────────────────────────────┐
              │      │ (row_store_start_row pulse)                         │
              │      ▼                                                     │
              │  LOAD_KV_FOR_SCORE ──► WAIT_KV_FOR_SCORE                  │
              │                            │ kv_load_done                  │
              │                            ▼                               │
              │                       START_SCORE ──► WAIT_SCORE           │
              │                                          │ score_done       │
              │                                          ▼                  │
              │                                   STORE_ROW_FRAGMENT        │
              │                                          │                  │
              │                                   NEXT_KV_FOR_SCORE         │
              │                                    │         │              │
              │                          more KV ◄─┘    all done           │
              │                                               │             │
              │                                    START_ROW_MAX            │
              │                                          │ row_max_done     │
              │                                    START_ROW_SUM            │
              │                                          │ row_sum_done     │
              │                                    START_NORMALIZE          │
              │                                          │ norm_done        │
              │                                    LOAD_KV_FOR_V ──► WAIT_KV_FOR_V
              │                                                          │ kv_load_done
              │                                                    START_WSUM
              │                                                     (clear_acc if idx=0)
              │                                                          │ wsum_done
              │                                                    NEXT_KV_FOR_V
              │                                                     │        │
              │                               more V tiles ◄────────┘   all done
              │                                                              │
              │                                                       WRITE_OUTPUT
              │                                                             │ out_write_done
              │                                                       NEXT_Q_ROW
              │                                more rows ──────────────────►┘
              │                                    │ all rows done
              │                               NEXT_Q_TILE
              └── more Q tiles ◄──────────────────┘
                                   │ all Q tiles done
                                   ▼
                                  DONE
```

**FSM State Encoding** (from `attention_pkg.sv`):

| State | Encoding | Action |
|---|---|---|
| `IDLE` | `5'd0` | Await `start` pulse |
| `LOAD_Q_TILE` | `5'd1` | Assert `q_load_start` |
| `WAIT_Q_TILE` | `5'd2` | Block on `q_load_done` |
| `INIT_ROW` | `5'd3` | Assert `row_store_start_row` (clear score buffer) |
| `LOAD_KV_FOR_SCORE` | `5'd4` | Assert `kv_load_start` |
| `WAIT_KV_FOR_SCORE` | `5'd5` | Block on `kv_load_done` |
| `START_SCORE` | `5'd6` | Assert `score_start` |
| `WAIT_SCORE` | `5'd7` | Block on `score_done` |
| `STORE_ROW_FRAGMENT` | `5'd8` | Assert `row_store_en` |
| `NEXT_KV_FOR_SCORE` | `5'd9` | Advance `kv_tile_idx`; loop or proceed |
| `START_ROW_MAX` | `5'd10` | Assert `row_max_start` |
| `WAIT_ROW_MAX` | `5'd11` | Block on `row_max_done` |
| `START_ROW_SUM` | `5'd12` | Assert `row_sum_start` |
| `WAIT_ROW_SUM` | `5'd13` | Block on `row_sum_done` |
| `START_NORMALIZE` | `5'd14` | Assert `norm_start` |
| `WAIT_NORMALIZE` | `5'd15` | Block on `norm_done` |
| `INIT_WSUM` | `5'd16` | *(Dead state — OPT-6 eliminated)* |
| `LOAD_KV_FOR_V` | `5'd17` | Assert `kv_load_start` (Pass B reload) |
| `WAIT_KV_FOR_V` | `5'd18` | Block on `kv_load_done` |
| `START_WSUM` | `5'd19` | Assert `wsum_start`; `wsum_clear_acc` if `kv_tile_idx==0` |
| `WAIT_WSUM` | `5'd20` | Block on `wsum_done` |
| `NEXT_KV_FOR_V` | `5'd21` | Advance `kv_tile_idx`; loop or write output |
| `WRITE_OUTPUT` | `5'd22` | Assert `out_write_en` |
| `WAIT_WRITE` | `5'd23` | Block on `out_write_done` |
| `NEXT_Q_ROW` | `5'd24` | Advance `q_local_row_idx`; loop or next tile |
| `NEXT_Q_TILE` | `5'd25` | Advance `q_tile_idx`; loop or done |
| `DONE` | `5'd26` | Assert `done`; hold until reset |

---

## 6. Module Reference

### 6.1 `attention_pkg` — Shared Parameter Package

Single source of truth for all constants. Every module imports this package.

```systemverilog
import attention_pkg::*;
```

| Parameter | Value | Description |
|---|---|---|
| `N` | 32 | Sequence length |
| `D` | 16 | Head dimension |
| `BR` | 4 | Query tile rows |
| `BC` | 4 | KV tile rows |
| `DATA_W` | 8 | INT8 input width |
| `SCORE_W` | 32 | Signed score accumulator width |
| `EXP_W` | 16 | exp LUT output width (Q0.16) |
| `SUM_W` | 32 | Softmax denominator width |
| `OUT_W` | 32 | Output accumulator width |
| `NUM_Q_TILES` | 8 | `N / BR` |
| `NUM_KV_TILES` | 8 | `N / BC` |
| `NEG_INF` | `-(1 <<< 30)` | Causal mask sentinel (INT32) |

---

### 6.2 `addr_gen` — Address Generator

Purely combinational. Converts tile indices from the controller into global memory addresses.

| Port | Dir | Width | Description |
|---|---|---|---|
| `q_tile_idx` | in | 3 | Current Q tile (0..7) |
| `kv_tile_idx` | in | 3 | Current KV tile (0..7) |
| `q_local_row_idx` | in | 2 | Row within Q tile (0..3) |
| `local_kv_row_idx` | in | 2 | Row within KV tile (tied to 0 in top) |
| `q_tile_base` | out | 5 | `q_tile_idx × BR` |
| `kv_tile_base` | out | 5 | `kv_tile_idx × BC` |
| `global_q_idx` | out | 5 | `q_tile_base + q_local_row_idx` |
| `global_k_idx` | out | 5 | `kv_tile_base + local_kv_row_idx` |

---

### 6.3 `q_buffer` — Query Tile Buffer

Loads one `[BR][D]` INT8 tile from `Q_mem` at address `q_tile_base`. Holds tile stable while the controller iterates all KV tiles.

- **Load time:** `BR × D = 64` cycles (one element per cycle, row-major)
- **Outputs:** `q_tile[BR][D]` → `score_engine`; `load_done` (pulse); `valid` (level)

---

### 6.4 `kv_buffer` — Key/Value Tile Buffer

Loads one `[BC][D]` K tile and one `[BC][D]` V tile simultaneously from `K_mem` and `V_mem`.

- **Load time:** `BC × D = 64` cycles
- **K tile** → `score_engine` (Pass A)
- **V tile** → `weighted_sum_engine` (Pass B)
- Both tiles are always loaded together regardless of which pass is active.

---

### 6.5 `score_engine` — Dot-Product Score Tile *(Amogha)*

Computes `score_tile[BR][BC]` = `(q_tile · k_tile^T) >>> SCALE_SHIFT`.

**Datapath:**
```
Stage 1 (comb):  q_ext × k_ext  ──► mac_product_reg  (pipeline register)
Stage 2 (comb):  acc += mac_product_reg               (or finalise score_tile entry)
```

| Signal | Dir | Description |
|---|---|---|
| `start` | in | One-cycle pulse from controller |
| `q_tile [BR][D]` | in | INT8 signed query vectors |
| `k_tile [BC][D]` | in | INT8 signed key vectors |
| `score_tile [BR][BC]` | out | INT32 signed scaled scores |
| `busy` | out | High during COMPUTE and DRAIN |
| `done` | out | One-cycle pulse at end of tile |

- **FSM:** `IDLE → COMPUTE → DRAIN → IDLE`
- **Loop order:** r (slowest) → c (mid) → d (fastest)
- **Latency:** `BR × BC × D + 2 = 258` cycles
- **Pipeline:** `comp_first` flag suppresses Stage 2 on first cycle; `r_cnt_p/c_cnt_p/d_cnt_p` track delayed counter values

---

### 6.6 `mask_unit` — Causal Mask *(Amogha)*

Applies causal mask after `score_engine`. Sets `score_tile[r][c] = NEG_INF` wherever `global_k > global_q`.

- **Implementation:** Combinational comparator array + **1-cycle pipeline register** on output
- **Latency overhead:** Zero net — absorbed by the FSM gap between `WAIT_SCORE` and `STORE_ROW_FRAGMENT`
- **NEG_INF value:** `-(1 <<< 30)` — chosen so `exp(NEG_INF - row_max)` maps to zero in the exp LUT

---

### 6.7 `row_score_store` — Score Row Buffer *(Amogha)*

Collects one full row of `N` masked scores across all KV tiles.

- **Protocol:**
  1. `start_row` pulse → clears `row_scores[N]` and `row_valid`
  2. `store_en` pulse → writes `BC` entries at `kv_tile_idx × BC`
  3. After `NUM_KV_TILES` stores → `row_valid` asserts; `store_done` pulses

---

### 6.8 `row_max_unit` — Row Maximum *(Jainil)*

Finds `max(row_scores[0:N-1])` using a **5-level pipelined binary comparison tree** (OPT-2).

```
Level 1 (S_IDLE start):  32 → 16 pairwise maxima  →  s1[16]
Level 2 (S_PIPE cnt=1):  16 →  8                  →  s2[8]
Level 3 (S_PIPE cnt=2):   8 →  4                  →  s3[4]
Level 4 (S_PIPE cnt=3):   4 →  2                  →  s4[2]
Level 5 (S_DONE):          2 →  1                  →  row_max
```

- **Latency:** 5 cycles (was 33 cycles sequential scan)
- **Saving:** 28 cycles/row × 32 rows = **896 cycles** over sequential baseline

---

### 6.9 `row_sum_unit` + `exp_lut` — Exponential Accumulator *(Jainil)*

Computes `exp_row[i] = exp(row_scores[i] - row_max)` and `row_sum = Σ exp_row[i]`.

**Internal `exp_lut`:** 9-entry priority-encoded LUT with 1-cycle pipeline register output.

**Two-pointer pipeline scheme:**
- `idx` drives `shifted_score` to `exp_lut` combinationally
- `idx_prev` (1-cycle delayed) is the index `exp_value` belongs to
- `scan_first` flag suppresses capture on the first S_SCAN cycle (pipeline fill)
- `S_DRAIN` flushes the final `exp_value` for `idx_prev = N-1`

- **Latency:** `N + 2 = 34` cycles
- **NEG_INF handling:** `exp(NEG_INF - row_max)` is clamped to `0x0000` by the LUT

---

### 6.10 `reciprocal_lut` — Reciprocal Approximation *(Jainil)*

Computes `row_sum_recip = 65536 / row_sum` (Q0.16 approximation).

- **Implementation:** Combinational integer divide `32'h0001_0000 / row_sum` + **1-cycle pipeline register**
- **Guard:** `row_sum == 0` → output `0xFFFF` (safe maximum)
- **Saturation:** result clipped to 16-bit max if overflow

> **Note:** `reciprocal_lut` is instantiated but `normalizer` uses direct division (`exp_row[i] × 0xFFFF / row_sum`) to avoid precision collapse on single-attended-token rows.

---

### 6.11 `normalizer` — Probability Row Generator *(Jainil)*

Converts `exp_row[N]` + `row_sum` into `prob_row[N]` (Q0.16 probabilities).

```
prob_row[i] = (exp_row[i] × 0xFFFF) / row_sum
```

**2-stage pipeline:**
```
Stage 1 (comb + reg):  exp_row[idx] × 0xFFFF  →  num_reg
Stage 2 (comb + reg):  num_reg / row_sum       →  prob_row[idx_p]
```

- `norm_first` suppresses Stage 2 write on first cycle
- `S_DRAIN` flushes the final `num_reg` for `idx_p = N-1`
- **Latency:** `N + 2 = 34` cycles
- `row_sum == 0` guard: `prob_next = 0`

---

### 6.12 `weighted_sum_engine` — Output Accumulator *(Jainil)*

Accumulates `acc[d] += Σ_c prob_slice[c] × v_tile[c][d]` across all KV tiles for one query row.

**Datapath (OPT-5 + OPT-ADDER-PIPE):**
```
Stage 1 (comb):  4 parallel MACs + 2-level adder tree  →  adder_sum_reg
Stage 2 (comb):  acc[d_idx_p] += adder_sum_reg
```

| Signal | Dir | Driver | Description |
|---|---|---|---|
| `clear_acc` | in | controller | Assert on `kv_tile_idx == 0` only; zeroes `acc[D]` |
| `start` | in | controller | One-cycle pulse per KV tile |
| `prob_slice [BC]` | in | attention_top glue | `prob_row[kv_tile_base + c]` |
| `v_tile [BC][D]` | in | kv_buffer | Value vectors for current KV tile |
| `out_row [D]` | out | — | Combinational mirror of `acc[]`; valid after last tile done |
| `busy` | out | — | High during S_CLEAR/S_ACCUM/S_DRAIN/S_DONE |
| `done` | out | — | One-cycle pulse per tile completion |

- **FSM:** `S_IDLE → S_CLEAR (if clear_acc) → S_ACCUM → S_DRAIN → S_DONE → S_IDLE`
- **Latency per tile:** `D + 2 = 18` cycles (first tile: `D + 3` with S_CLEAR)
- **Throughput vs baseline:** 4× improvement (BC=4 parallel MACs instead of sequential loop)

**`prob_slice` selection (attention_top glue logic):**
```systemverilog
always_comb begin
  for (int c = 0; c < BC; c++)
    prob_slice[c] = prob_row[kv_tile_base + c];
end
```

---

### 6.13 `output_buffer` — Output Row Writer *(Parin)*

Writes `out_row[D]` → `O_mem[global_q_idx][:]` one element per cycle.

- **Write time:** `D = 16` cycles
- Latches `global_q_idx` at `write_en` to protect against controller advancing the index mid-write
- Uses `global_q_idx` (absolute), not `q_local_row_idx` (tile-local), to correctly address `O_mem`

---

### 6.14 `perf_counters` — Performance Observability *(Parin)*

Counts six metrics between `start` and `done`. All counters freeze when `done` asserts.

See [Section 9 — Performance Counters](#9-performance-counters) for full definitions.

---

## 7. Parameters & Data Formats

### Primary Parameters

| Parameter | Value | Configurable? | Notes |
|---|---|---|---|
| `N` | 32 | Elaboration-time only | Sequence length |
| `D` | 16 | Elaboration-time only | Head dimension |
| `BR` | 4 | Elaboration-time only | Q tile rows; must divide N |
| `BC` | 4 | Elaboration-time only | KV tile rows; must divide N |
| `SCALE_SHIFT` | 2 | Localparam in score_engine | Post-dot-product right shift |

### Data Format Summary

| Data | Format | Width | Range |
|---|---|---|---|
| Q, K, V inputs | INT8 signed | 8-bit | −128 .. 127 |
| Score accumulator | INT32 signed | 32-bit | ~±2.1 × 10⁹ |
| exp LUT output | Q0.16 unsigned | 16-bit | 0 .. 65535 |
| Softmax denominator | Unsigned | 32-bit | 0 .. ~2³² |
| prob_row entries | Q0.16 unsigned | 16-bit | 0 .. 65535 (≈ 0.0 .. 1.0) |
| Output accumulator | INT32 signed | 32-bit | ~±2.1 × 10⁹ |
| `NEG_INF` sentinel | INT32 signed | 32-bit | `-(1 << 30) = -1073741824` |

---

## 8. AXI-Lite Register Map

The `attention_axi_wrapper` exposes the core over a 16-bit AXI4-Lite slave interface at 32-bit data width. Q/K/V matrices are written before asserting start; O matrix is read after done.

### Control & Status Registers

| Offset | Name | Access | Description |
|---|---|---|---|
| `0x0000` | `CTRL` | R/W | Bit 0: Write 1 to start (ignored if busy). Bit 1: Write 1 to clear done/IRQ. Bit 8: IRQ enable. Bit 9: IRQ status (read). |
| `0x0004` | `STATUS` | R | Bit 0: busy. Bit 1: done_sticky. Bit 2: core_done (pulse). |

### Performance Counter Registers (read-only)

| Offset | Name | Description |
|---|---|---|
| `0x0010` | `CYCLE_COUNT` | Total cycles from start to done |
| `0x0014` | `SCORE_CYCLES` | Cycles `score_engine.busy` was high |
| `0x0018` | `SOFTMAX_CYCLES` | Cycles any softmax unit was busy |
| `0x001C` | `WSUM_CYCLES` | Cycles `weighted_sum_engine.busy` was high |
| `0x0020` | `LOAD_EVENTS` | Total Q + KV tile load requests |
| `0x0024` | `STALL_CYCLES` | Stall cycles (reserved; tied to 0 in baseline) |

### Memory Windows

Each memory is stored row-major. Element `[row][col]` is at offset `(row × D + col) × 4` bytes from the window base.

| Offset | Size | Memory | Access |
|---|---|---|---|
| `0x0100` | 2 KB | `Q_mem [N][D]` | R/W (write before start only) |
| `0x0900` | 2 KB | `K_mem [N][D]` | R/W (write before start only) |
| `0x1100` | 2 KB | `V_mem [N][D]` | R/W (write before start only) |
| `0x1900` | 2 KB | `O_mem [N][D]` | R (valid after done) |

**Memory size:** `N × D × 4 bytes = 32 × 16 × 4 = 2048 bytes` per array.

### Software Usage Sequence

```c
// 1. Write Q, K, V matrices element by element
for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++)
        axi_write(Q_BASE + (i*D + j)*4, q_data[i][j] & 0xFF);

// 2. Start computation
axi_write(CTRL_ADDR, 0x101);   // bit0=start, bit8=irq_enable

// 3. Poll STATUS until done_sticky (bit1)
while (!(axi_read(STATUS_ADDR) & 0x2)) { /* wait */ }

// 4. Read output matrix
for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++)
        o_data[i][j] = (int32_t)axi_read(O_BASE + (i*D + j)*4);

// 5. Clear done
axi_write(CTRL_ADDR, 0x2);     // bit1=clear done
```

---

## 9. Performance Counters

All counters reset on `start` and freeze on `done`. Accessible via AXI or directly on the `attention_top` output ports.

| Counter | Port | Description | Expected (default params) |
|---|---|---|---|
| `cycle_count` | `cycle_count` | Total wall-clock cycles | Design-dependent |
| `score_cycles` | `score_cycles` | Cycles `score_engine` busy | `NUM_Q_TILES × BR × NUM_KV_TILES × 258` |
| `softmax_cycles` | `softmax_cycles` | Cycles `row_max ∥ row_sum ∥ norm` busy | Per row: `5 + 34 + 34 = 73` cycles |
| `wsum_cycles` | `wsum_cycles` | Cycles `weighted_sum_engine` busy | Per row: `19 + 7×18 = 145` cycles |
| `load_events` | `load_events` | Count of `q_load_start + kv_load_start` | Q: 8; KV: `8×4×8×2 = 512` total |
| `stall_cycles` | `stall_cycles` | Cycles `stall_flag` asserted | 0 (tied to 0 in baseline) |

---

## 10. Pipeline Optimisations

All optimisations are tagged in source with `OPT-n` comments.

| Tag | Module | Description | Saving |
|---|---|---|---|
| **OPT-1** | `score_engine` | Eliminated separate SCALE and DONE states | −2 states |
| **OPT-2** | `row_max_unit` | 5-level binary tree replaces 33-cycle sequential scan | 28 cy/row × 32 rows = **896 cy** |
| **OPT-5** | `weighted_sum_engine` | Unrolled inner BC=4 c-loop (4 parallel MACs) | 4× throughput; 48 cy/tile → 16 cy/tile |
| **OPT-6** | `attention_controller` | Eliminated `INIT_WSUM` dead state | 1 cy/row × 32 rows = **32 cy** |
| **OPT-MAC-PIPE** | `score_engine` | `mac_product_reg` register breaks multiply→accumulate critical path | fmax improvement |
| **OPT-ADDER-PIPE** | `weighted_sum_engine` | `adder_sum_reg` register breaks MAC tree→accumulate critical path | fmax improvement |
| *(mask pipeline)* | `mask_unit` | 1-cycle output register breaks BR×BC comparator path | fmax improvement; 0 net latency |
| *(norm pipeline)* | `normalizer` | 2-stage mul/div split; both divider inputs registered | fmax improvement |
| *(recip pipeline)* | `reciprocal_lut` | Output register breaks combinational divide path | fmax improvement |
| *(exp pipeline)* | `exp_lut` | 1-cycle output register; absorbed by `S_DRAIN` in row_sum_unit | fmax improvement |

---

## 11. Latency Analysis

### Per Query Row

| Phase | Cycles | Notes |
|---|---|---|
| Q tile load (amortised) | 64 / BR = **16** | Load once per tile; amortised over BR=4 rows |
| Pass A per KV tile | 64 (load) + 258 (score) + 1 (mask) + 1 (store) | = **324** cycles |
| Pass A total (× 8 KV tiles) | **2592** cycles | |
| `row_max_unit` | **5** | 5-level binary tree |
| `row_sum_unit` | **34** | N + 2 |
| `normalizer` | **34** | N + 2 |
| Pass B per KV tile (first) | 64 (load) + 19 (wsum+clear) | = **83** cycles |
| Pass B per KV tile (subsequent × 7) | 64 (load) + 18 (wsum) | = **82** cycles × 7 = **574** |
| Pass B total | 83 + 574 = **657** cycles | |
| Output write | **16** | D cycles |
| **Total per query row** | **~3354** cycles | + FSM transition overhead |

### Full Computation

| Scope | Cycles |
|---|---|
| Per Q tile (BR=4 rows) | ~3354 × 4 = ~13,416 |
| Full N=32 output (8 Q tiles) | ~13,416 × 8 = **~107,328** |

---

## 12. File Listing & Compile Order

From `DEPS.yml` — compile in this order for correct package elaboration:

```
attention_pkg.sv           # Shared package — MUST be first

# Control infrastructure (Parin)
addr_gen.sv
q_buffer.sv
kv_buffer.sv
output_buffer.sv
perf_counters.sv
attention_controller.sv

# Score path (Amogha)
score_engine.sv
mask_unit.sv
row_score_store.sv

# Softmax + output accumulation (Jainil)
row_max_unit.sv
exp_lut.sv
row_sum_unit.sv
reciprocal_lut.sv
normalizer.sv
weighted_sum_engine.sv

# Top-level integration (Parin)
attention_top.sv

# Optional: AXI wrapper (FPGA integration)
attention_axi_wrapper.sv    # or attention_axi_wrapper.v (BD wrapper)

# Testbench (not included in synthesis)
tb_attention_top.sv
```

### Deliverables

| File | Description |
|---|---|
| `attention_accel.bit` | Vivado bitstream for Xilinx 7-series |
| `attention_accel.xsa` | Xilinx Support Archive (hardware handoff for Vitis/PetaLinux) |

---

## 13. Simulation & Build

### Prerequisites

- SystemVerilog simulator (VCS, Questa, Xcelium, or Vivado Simulator)
- Vivado 2022.x+ (for synthesis and bitstream)

### Simulation

```bash
# Compile (VCS example)
vcs -sverilog -full64 \
  attention_pkg.sv \
  addr_gen.sv q_buffer.sv kv_buffer.sv output_buffer.sv \
  perf_counters.sv attention_controller.sv \
  score_engine.sv mask_unit.sv row_score_store.sv \
  row_max_unit.sv exp_lut.sv row_sum_unit.sv \
  reciprocal_lut.sv normalizer.sv weighted_sum_engine.sv \
  attention_top.sv \
  tb_attention_top.sv \
  -top tb_attention_top -o sim_out

# Run
./sim_out +vcs+dumpvars
```

### Vivado Synthesis (CLI)

```tcl
# In Vivado Tcl console or script
read_verilog -sv attention_pkg.sv
read_verilog -sv {addr_gen.sv q_buffer.sv kv_buffer.sv output_buffer.sv
                  perf_counters.sv attention_controller.sv
                  score_engine.sv mask_unit.sv row_score_store.sv
                  row_max_unit.sv exp_lut.sv row_sum_unit.sv
                  reciprocal_lut.sv normalizer.sv weighted_sum_engine.sv
                  attention_top.sv attention_axi_wrapper.sv}
synth_design -top attention_axi_wrapper -part xc7z020clg484-1
```

### Verification Checklist

| Check | Method |
|---|---|
| Correct output matrix | Compare `O_mem[N][D]` against Python golden model |
| Causal masking | Confirm `O_mem[q][:]` depends only on V[0..q], not V[q+1..N-1] |
| Softmax correctness | Confirm `prob_row` sums to ≈ `0xFFFF` per row |
| `done` exactly once | Assert `done` is a one-cycle pulse per computation |
| No accumulator bleed | Run 2+ consecutive computations with different inputs |
| Performance counters | Cross-check `cycle_count` against waveform cycle count |

---

## 14. Signal Ownership Reference

| Signal | Produced by | Consumed by |
|---|---|---|
| `start` (top-level) | testbench / AXI wrapper | `attention_controller`, `perf_counters` |
| `q_load_start` | `attention_controller` | `q_buffer`, `perf_counters` |
| `kv_load_start` | `attention_controller` | `kv_buffer`, `perf_counters` |
| `score_start` | `attention_controller` | `score_engine` |
| `row_store_start_row` | `attention_controller` | `row_score_store` |
| `row_store_en` | `attention_controller` | `row_score_store` |
| `row_max_start` | `attention_controller` | `row_max_unit` |
| `row_sum_start` | `attention_controller` | `row_sum_unit` |
| `norm_start` | `attention_controller` | `normalizer` |
| `wsum_start` | `attention_controller` | `weighted_sum_engine` |
| `wsum_clear_acc` | `attention_controller` | `weighted_sum_engine` |
| `out_write_en` | `attention_controller` | `output_buffer` |
| `q_tile_base` | `addr_gen` | `q_buffer`, `mask_unit` |
| `kv_tile_base` | `addr_gen` | `kv_buffer`, `mask_unit`, `attention_top` (prob_slice) |
| `global_q_idx` | `addr_gen` | `output_buffer` |
| `q_tile [BR][D]` | `q_buffer` | `score_engine` |
| `k_tile [BC][D]` | `kv_buffer` | `score_engine` |
| `v_tile [BC][D]` | `kv_buffer` | `weighted_sum_engine` |
| `score_tile [BR][BC]` | `score_engine` | `mask_unit` |
| `masked_score_tile [BR][BC]` | `mask_unit` | `row_score_store` |
| `row_scores [N]` | `row_score_store` | `row_max_unit`, `row_sum_unit` |
| `row_max` | `row_max_unit` | `row_sum_unit` |
| `exp_row [N]` | `row_sum_unit` | `normalizer` |
| `row_sum` | `row_sum_unit` | `reciprocal_lut`, `normalizer` |
| `row_sum_recip` | `reciprocal_lut` | *(kept; not wired to normalizer in v1.0)* |
| `prob_row [N]` | `normalizer` | `attention_top` (prob_slice glue) |
| `prob_slice [BC]` | `attention_top` (glue) | `weighted_sum_engine` |
| `out_row [D]` | `weighted_sum_engine` | `output_buffer` |
| `O_mem [N][D]` | `output_buffer` | testbench / AXI wrapper |
| `*.busy` | each compute module | `perf_counters` |
| `*.done` | each compute module | `attention_controller` |
| `cycle_count` etc. | `perf_counters` | AXI wrapper, testbench |

---

## Appendix A — Handshake Protocol

All submodule interfaces follow a uniform one-cycle pulse protocol:

```
              ┌─┐
start  ───────┘ └───────────────────────────
               ↑
              busy asserts

busy   ─────────────────────────────────┐
                                        └──
                                    ↑
                                 done pulse

              ┌──────────────────────┐
busy   ────────                      └──────

                                      ┌─┐
done   ───────────────────────────────┘ └───
```

**Rules:**
1. `start` must be a one-cycle pulse; not asserted while `busy == 1`
2. `done` is exactly one cycle; consumers must sample it in the same clock cycle it is high
3. `busy` is high from the cycle `start` is sampled until `done` fires
4. Data outputs (e.g. `score_tile`, `out_row`) are valid from `done` onward until the next `start`

---

## Appendix B — Known Limitations

| ID | Description | Workaround / Plan |
|---|---|---|
| L-001 | N, D, BR, BC are elaboration-time constants only | Re-synthesise for different configurations |
| L-002 | `reciprocal_lut` is instantiated but `normalizer` uses direct divide | Planned refactor: normalizer to use reciprocal path |
| L-003 | `INIT_WSUM` state is dead (OPT-6 bypasses it) | State encoding retained for compatibility; safe to remove in next revision |
| L-004 | No runtime reconfiguration via AXI | Future: AXI-writeable N, D, BR, BC registers |
| L-005 | Q/K/V arrays must fit in on-chip registers (N×D×DATA_W per array) | Future: DMA streaming from DDR with double-buffering |

---

*FlashAttention Inspired Tiled Causal Self Attention Accelerator· v1.0 · Parin · Amogha · Jainil*
