// ============================================================
// attention_top.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Top-level integration module. Instantiates and wires all
//   functional submodules. Implements the prob_slice selection
//   glue logic. Exposes Q/K/V memory inputs, the O_mem output,
//   and all performance counter outputs.
//
// MODULES INSTANTIATED HERE:
//   (owned by Parin)
//   - attention_controller
//   - addr_gen
//   - q_buffer
//   - kv_buffer
//   - output_buffer
//   - perf_counters
//
//   (owned by teammates)
//   - score_engine
//   - mask_unit
//   - row_score_store
//   - row_max_unit
//   - row_sum_unit
//   - reciprocal_lut
//   - normalizer
//   - weighted_sum_engine
//
// NOT INSTANTIATED HERE:
//   - attention_pkg  (imported as a package)
//   - tb_attention_top (testbench is a separate top-level)
//   - dot_product_pe   (optional, internal to score_engine)
//
// KEY GLUE LOGIC:
//   prob_slice[c] = prob_row[kv_tile_base + c]  for c = 0..BC-1
//   This is the only combinational logic that lives in this file.
// ============================================================

import attention_pkg::*;

module attention_top (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      start,

  // ---- Input memory arrays (preloaded by testbench) ----
  input  logic signed [DATA_W-1:0] Q_mem [0:N-1][0:D-1],
  input  logic signed [DATA_W-1:0] K_mem [0:N-1][0:D-1],
  input  logic signed [DATA_W-1:0] V_mem [0:N-1][0:D-1],

  // ---- Completion flag ----
  output logic                      done,

  // ---- Output memory (written by output_buffer) ----
  output logic signed [OUT_W-1:0]  O_mem [0:N-1][0:D-1],

  // ---- Performance counters ----
  output logic [31:0]               cycle_count,
  output logic [31:0]               score_cycles,
  output logic [31:0]               softmax_cycles,
  output logic [31:0]               wsum_cycles,
  output logic [31:0]               load_events,
  output logic [31:0]               stall_cycles
);

  // ==========================================================
  // SECTION 1: CONTROLLER OUTPUTS (control signals + indices)
  // ==========================================================

  // ---- Controller start/enable outputs ----
  logic q_load_start;
  logic kv_load_start;
  logic score_start;
  logic row_store_start_row;
  logic row_store_en;
  logic row_max_start;
  logic row_sum_start;
  logic norm_start;
  logic wsum_clear_acc;
  logic wsum_start;
  logic out_write_en;

  // ---- Loop indices from controller ----
  logic [Q_TILE_IDX_W-1:0]  q_tile_idx;
  logic [KV_TILE_IDX_W-1:0] kv_tile_idx;
  logic [BR_IDX_W-1:0]      q_local_row_idx;

  // ==========================================================
  // SECTION 2: ADDRESS GENERATOR OUTPUTS
  // ==========================================================

  logic [SEQ_IDX_W-1:0] q_tile_base;
  logic [SEQ_IDX_W-1:0] kv_tile_base;
  logic [SEQ_IDX_W-1:0] global_q_idx;
  logic [SEQ_IDX_W-1:0] global_k_idx;  // available for debug

  // ==========================================================
  // SECTION 3: BUFFER OUTPUTS
  // ==========================================================

  // ---- Q buffer ----
  logic signed [DATA_W-1:0] q_tile [BR][D];
  logic                      q_load_done;
  logic                      q_valid;

  // ---- KV buffer ----
  logic signed [DATA_W-1:0] k_tile [BC][D];
  logic signed [DATA_W-1:0] v_tile [BC][D];
  logic                      kv_load_done;
  logic                      kv_valid;

  // ==========================================================
  // SECTION 4: COMPUTE MODULE OUTPUTS
  // ==========================================================

  // ---- score_engine ----
  logic signed [SCORE_W-1:0] score_tile     [BR][BC];
  logic                       score_busy;
  logic                       score_done;

  // ---- mask_unit (sequential: 1-cycle pipeline register) ----
  logic signed [SCORE_W-1:0] masked_score_tile [BR][BC];

  // ---- row_score_store ----
  logic signed [SCORE_W-1:0] row_scores [0:N-1];
  logic                       row_valid;
  logic                       row_store_done;

  // ---- row_max_unit ----
  logic signed [SCORE_W-1:0] row_max;
  logic                       row_max_busy;
  logic                       row_max_done;

  // ---- row_sum_unit ----
  logic [EXP_W-1:0]          exp_row [0:N-1];
  logic [SUM_W-1:0]           row_sum;
  logic                       row_sum_busy;
  logic                       row_sum_done;

  // ---- reciprocal_lut (sequential: 1-cycle pipeline register) ----
  logic [EXP_W-1:0]          row_sum_recip;

  // ---- normalizer ----
  logic [EXP_W-1:0]          prob_row [0:N-1];
  logic                       norm_busy;
  logic                       norm_done;

  // ---- prob_slice glue (attention_top owns this) ----
  // prob_slice[c] = prob_row[kv_tile_base + c] for c = 0..BC-1
  // kv_tile_base is always a multiple of BC so no out-of-bounds risk.
  logic [EXP_W-1:0]          prob_slice [0:BC-1];

  // ---- weighted_sum_engine ----
  logic signed [OUT_W-1:0]   out_row [0:D-1];
  logic                       wsum_busy;
  logic                       wsum_done;

  // ---- output_buffer done ----
  logic                       out_write_done;

  // ---- softmax aggregate busy (for perf_counters) ----
  logic softmax_busy;
  assign softmax_busy = row_max_busy | row_sum_busy | norm_busy;

  // ==========================================================
  // SECTION 5: PROB_SLICE SELECTION GLUE
  // prob_slice is selected combinationally from prob_row using
  // kv_tile_base. This is the ONLY place this slicing occurs.
  // weighted_sum_engine receives a BC-wide window of prob_row
  // corresponding to the current KV tile.
  // ==========================================================
  always_comb begin
    for (int c = 0; c < BC; c++) begin
      prob_slice[c] = prob_row[kv_tile_base + SEQ_IDX_W'(c)];
    end
  end

  // ==========================================================
  // SECTION 6: MODULE INSTANTIATIONS
  // ==========================================================

  // ----------------------------------------------------------
  // attention_controller
  // Drives all start/enable signals and loop indices.
  // ----------------------------------------------------------
  attention_controller u_controller (
    .clk                (clk),
    .rst_n              (rst_n),
    .start              (start),
    // done inputs from datapath
    .q_load_done        (q_load_done),
    .kv_load_done       (kv_load_done),
    .score_done         (score_done),
    .row_store_done     (row_store_done),
    .row_valid          (row_valid),
    .row_max_done       (row_max_done),
    .row_sum_done       (row_sum_done),
    .norm_done          (norm_done),
    .wsum_done          (wsum_done),
    .out_write_done     (out_write_done),
    // control outputs
    .q_load_start       (q_load_start),
    .kv_load_start      (kv_load_start),
    .score_start        (score_start),
    .row_store_start_row(row_store_start_row),
    .row_store_en       (row_store_en),
    .row_max_start      (row_max_start),
    .row_sum_start      (row_sum_start),
    .norm_start         (norm_start),
    .wsum_clear_acc     (wsum_clear_acc),
    .wsum_start         (wsum_start),
    .out_write_en       (out_write_en),
    // index outputs
    .q_tile_idx         (q_tile_idx),
    .kv_tile_idx        (kv_tile_idx),
    .q_local_row_idx    (q_local_row_idx),
    .done               (done)
  );

  // ----------------------------------------------------------
  // addr_gen
  // Converts indices to tile bases and global row indices.
  // local_kv_row_idx tied to 0 (global_k_idx used for debug only).
  // ----------------------------------------------------------
  addr_gen u_addr_gen (
    .q_tile_idx        (q_tile_idx),
    .kv_tile_idx       (kv_tile_idx),
    .q_local_row_idx   (q_local_row_idx),
    .local_kv_row_idx  ({BC_IDX_W{1'b0}}),
    .q_tile_base       (q_tile_base),
    .kv_tile_base      (kv_tile_base),
    .global_q_idx      (global_q_idx),
    .global_k_idx      (global_k_idx)
  );

  // ----------------------------------------------------------
  // q_buffer
  // Loads one Q tile [BR][D] on load_start, holds until next load.
  // q_tile -> score_engine
  // ----------------------------------------------------------
  q_buffer u_q_buffer (
    .clk         (clk),
    .rst_n       (rst_n),
    .load_start  (q_load_start),
    .q_tile_base (q_tile_base),
    .Q_mem       (Q_mem),
    .q_tile      (q_tile),
    .load_done   (q_load_done),
    .valid       (q_valid)
  );

  // ----------------------------------------------------------
  // kv_buffer
  // Loads one K+V tile [BC][D] on load_start.
  // k_tile -> score_engine (Pass A)
  // v_tile -> weighted_sum_engine (Pass B)
  // ----------------------------------------------------------
  kv_buffer u_kv_buffer (
    .clk          (clk),
    .rst_n        (rst_n),
    .load_start   (kv_load_start),
    .kv_tile_base (kv_tile_base),
    .K_mem        (K_mem),
    .V_mem        (V_mem),
    .k_tile       (k_tile),
    .v_tile       (v_tile),
    .load_done    (kv_load_done),
    .valid        (kv_valid)
  );

  // ----------------------------------------------------------
  // score_engine  (owned by Amogha)
  // Computes score_tile[BR][BC] = (q_tile * k_tile^T) >>> 2
  // ----------------------------------------------------------
  score_engine #(
    .BR     (BR),
    .BC     (BC),
    .D      (D),
    .DATA_W (DATA_W),
    .SCORE_W(SCORE_W)
  ) u_score_engine (
    .clk        (clk),
    .rst_n      (rst_n),
    .start      (score_start),
    .q_tile     (q_tile),
    .k_tile     (k_tile),
    .score_tile (score_tile),
    .busy       (score_busy),
    .done       (score_done)
  );

  // ----------------------------------------------------------
  // mask_unit  (owned by Amogha)
  // Applies causal mask: score[r][c] = NEG_INF if global_k > global_q
  // Sequential: 1-cycle pipeline register breaks comparator path.
  // Latency absorbed by FSM gap: WAIT_SCORE -> STORE_ROW_FRAGMENT.
  // ----------------------------------------------------------
  mask_unit #(
    .BR     (BR),
    .BC     (BC),
    .SCORE_W(SCORE_W),
    .IDX_W  (SEQ_IDX_W)
  ) u_mask_unit (
    .clk            (clk),
    .rst_n          (rst_n),
    .score_tile_in  (score_tile),
    .q_tile_base    (q_tile_base),
    .kv_tile_base   (kv_tile_base),
    .score_tile_out (masked_score_tile)
  );

  // ----------------------------------------------------------
  // row_score_store  (owned by Amogha)
  // Collects one full row of N masked scores across all KV tiles.
  // On start_row: clears the row buffer.
  // On store_en:  writes BC-wide fragment at kv_tile_idx*BC.
  // ----------------------------------------------------------
  row_score_store #(
    .N      (N),
    .BR     (BR),
    .BC     (BC),
    .SCORE_W(SCORE_W),
    .IDX_W  (SEQ_IDX_W)
  ) u_row_score_store (
    .clk             (clk),
    .rst_n           (rst_n),
    .start_row       (row_store_start_row),
    .store_en        (row_store_en),
    .q_local_row_idx (q_local_row_idx),
    .kv_tile_idx     (kv_tile_idx),
    .score_tile_in   (masked_score_tile),
    .row_scores      (row_scores),
    .row_valid       (row_valid),
    .store_done      (row_store_done)
  );

  // ----------------------------------------------------------
  // row_max_unit  (owned by Jainil)
  // Scans row_scores[N] to find the maximum for softmax stability.
  // ----------------------------------------------------------
  row_max_unit u_row_max_unit (
    .clk        (clk),
    .rst_n      (rst_n),
    .start      (row_max_start),
    .row_scores (row_scores),
    .row_max    (row_max),
    .busy       (row_max_busy),
    .done       (row_max_done)
  );

  // ----------------------------------------------------------
  // row_sum_unit  (owned by Jainil)
  // Computes exp_row[N] = exp(row_scores - row_max) and row_sum.
  // ----------------------------------------------------------
  row_sum_unit u_row_sum_unit (
    .clk        (clk),
    .rst_n      (rst_n),
    .start      (row_sum_start),
    .row_scores (row_scores),
    .row_max    (row_max),
    .exp_row    (exp_row),
    .row_sum    (row_sum),
    .busy       (row_sum_busy),
    .done       (row_sum_done)
  );

  // ----------------------------------------------------------
  // reciprocal_lut  (owned by Jainil)
  // Sequential: 1-cycle pipeline register breaks the divide path.
  // row_sum_recip valid one cycle after row_sum_done.
  // ----------------------------------------------------------
  reciprocal_lut u_reciprocal_lut (
    .clk           (clk),
    .rst_n         (rst_n),
    .row_sum       (row_sum),
    .row_sum_recip (row_sum_recip)
  );

  // ----------------------------------------------------------
  // normalizer  (owned by Jainil)
  // Computes prob_row[N] = (exp_row * row_sum_recip) >> 16
  // ----------------------------------------------------------
  normalizer u_normalizer (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (norm_start),
    .exp_row      (exp_row),
    .row_sum      (row_sum),      // direct division; reciprocal_lut kept but unused by normalizer
    .prob_row     (prob_row),
    .busy         (norm_busy),
    .done         (norm_done)
  );

  // ----------------------------------------------------------
  // weighted_sum_engine  (owned by Jainil)
  // Accumulates out_row[D] = sum over BC of prob_slice * v_tile.
  // clear_acc is high only on the first KV tile of Pass B.
  // prob_slice is selected from prob_row in this module (above).
  // ----------------------------------------------------------
  weighted_sum_engine u_wsum_engine (
    .clk       (clk),
    .rst_n     (rst_n),
    .clear_acc (wsum_clear_acc),
    .start     (wsum_start),
    .prob_slice(prob_slice),
    .v_tile    (v_tile),
    .out_row   (out_row),
    .busy      (wsum_busy),
    .done      (wsum_done)
  );

  // ----------------------------------------------------------
  // output_buffer
  // Writes out_row into O_mem[global_q_idx][:].
  // global_q_idx is the absolute sequence position (not tile-local).
  // ----------------------------------------------------------
  output_buffer u_output_buffer (
    .clk          (clk),
    .rst_n        (rst_n),
    .write_en     (out_write_en),
    .global_q_idx (global_q_idx),
    .out_row      (out_row),
    .O_mem        (O_mem),
    .write_done   (out_write_done)
  );

  // ----------------------------------------------------------
  // perf_counters
  // Tracks cycle counts and activity metrics for demo reporting.
  // stall_flag tied to 0 in the baseline design.
  // ----------------------------------------------------------
  perf_counters u_perf_counters (
    .clk           (clk),
    .rst_n         (rst_n),
    .start         (start),
    .done          (done),
    .score_busy    (score_busy),
    .row_max_busy  (row_max_busy),
    .row_sum_busy  (row_sum_busy),
    .norm_busy     (norm_busy),
    .wsum_busy     (wsum_busy),
    .q_load_start  (q_load_start),
    .kv_load_start (kv_load_start),
    .stall_flag    (1'b0),
    .cycle_count   (cycle_count),
    .score_cycles  (score_cycles),
    .softmax_cycles(softmax_cycles),
    .wsum_cycles   (wsum_cycles),
    .load_events   (load_events),
    .stall_cycles  (stall_cycles)
  );

endmodule
