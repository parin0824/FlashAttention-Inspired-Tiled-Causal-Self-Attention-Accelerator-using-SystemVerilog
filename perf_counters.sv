// ============================================================
// perf_counters.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Tracks six performance metrics during one complete attention
//   inference pass. All counters are active only between start
//   and done so repeated simulations produce clean, comparable
//   readings.
//
// METRIC DEFINITIONS:
//
//   cycle_count:
//     Total wall-clock cycles from start to done.
//     Useful for end-to-end latency measurement.
//
//   score_cycles:
//     Cycles during which score_engine is actively computing.
//     High utilization here means the MAC array is the bottleneck.
//
//   softmax_cycles:
//     Cycles during which any of row_max_unit, row_sum_unit, or
//     normalizer is active. Because these run sequentially in the
//     baseline, this measures the total softmax pipeline cost
//     per query row.
//
//   wsum_cycles:
//     Cycles during which weighted_sum_engine is accumulating.
//     Measures output accumulation cost across all V tiles.
//
//   load_events:
//     Number of individual tile load requests (q_load_start or
//     kv_load_start pulses). Each event transfers BR*D or BC*D
//     elements. Useful for estimating memory bandwidth demand.
//
//   stall_cycles:
//     Cycles during which an external stall_flag is asserted.
//     In the baseline this is unused (tie to 0); reserved for
//     future memory latency or backpressure modeling.
//
// SIGNAL OWNERSHIP:
//   start, done              <- attention_controller
//   score_busy               <- score_engine
//   row_max_busy             <- row_max_unit
//   row_sum_busy             <- row_sum_unit
//   norm_busy                <- normalizer
//   wsum_busy                <- weighted_sum_engine
//   q_load_start             <- attention_controller
//   kv_load_start            <- attention_controller
//   stall_flag               <- attention_top (tie to 0 for baseline)
//   all outputs              -> attention_top output ports, testbench
// ============================================================

import attention_pkg::*;

module perf_counters (
  input  logic        clk,
  input  logic        rst_n,

  // ---- Run boundary signals (from controller) ----
  input  logic        start,          // rising edge starts all counters
  input  logic        done,           // when asserted, counters freeze

  // ---- Compute busy signals ----
  input  logic        score_busy,     // score_engine is computing
  input  logic        row_max_busy,   // row_max_unit is scanning
  input  logic        row_sum_busy,   // row_sum_unit is accumulating
  input  logic        norm_busy,      // normalizer is generating prob_row
  input  logic        wsum_busy,      // weighted_sum_engine is accumulating

  // ---- Memory load pulse signals (from controller) ----
  input  logic        q_load_start,   // pulse: Q tile load initiated
  input  logic        kv_load_start,  // pulse: KV tile load initiated

  // ---- Optional stall indicator ----
  input  logic        stall_flag,     // tie to 0 in baseline

  // ---- Counter outputs (to attention_top / testbench) ----
  output logic [31:0] cycle_count,    // total active cycles
  output logic [31:0] score_cycles,   // cycles score_engine was busy
  output logic [31:0] softmax_cycles, // cycles any softmax unit was busy
  output logic [31:0] wsum_cycles,    // cycles wsum engine was busy
  output logic [31:0] load_events,    // total tile load requests issued
  output logic [31:0] stall_cycles    // cycles stall_flag was asserted
);

  // ----------------------------------------------------------
  // Internal active flag.
  // Counters count only while active = 1, which is set by
  // start and cleared when done asserts.
  // ----------------------------------------------------------
  logic active;

  // ----------------------------------------------------------
  // Sequential logic: active flag and all counters
  // ----------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Clear everything on reset
      active         <= 1'b0;
      cycle_count    <= 32'd0;
      score_cycles   <= 32'd0;
      softmax_cycles <= 32'd0;
      wsum_cycles    <= 32'd0;
      load_events    <= 32'd0;
      stall_cycles   <= 32'd0;

    end else begin

      // --------------------------------------------------------
      // Active flag control:
      //   - Set when start pulses (and not yet done)
      //   - Cleared the cycle done asserts
      // done takes priority so the final done cycle is not counted
      // --------------------------------------------------------
      if (done) begin
        active <= 1'b0;
      end else if (start) begin
        active <= 1'b1;
      end

      // --------------------------------------------------------
      // cycle_count:
      //   Increment every cycle the design is active.
      //   Measures total latency from start to done.
      // --------------------------------------------------------
      if (active && !done) begin
        cycle_count <= cycle_count + 32'd1;
      end

      // --------------------------------------------------------
      // score_cycles:
      //   Increment while score_engine.busy is asserted.
      //   Indicates how many cycles are spent on QK^T dot products.
      //   In the tiled design: NUM_Q_TILES * BR * NUM_KV_TILES
      //   score tile computations occur.
      // --------------------------------------------------------
      if (active && !done && score_busy) begin
        score_cycles <= score_cycles + 32'd1;
      end

      // --------------------------------------------------------
      // softmax_cycles:
      //   Increment while any softmax-path unit is active.
      //   row_max_unit, row_sum_unit, and normalizer run sequentially
      //   per query row, so their busy signals do not overlap.
      //   Summing them via OR gives total softmax cost per row.
      // --------------------------------------------------------
      if (active && !done && (row_max_busy || row_sum_busy || norm_busy)) begin
        softmax_cycles <= softmax_cycles + 32'd1;
      end

      // --------------------------------------------------------
      // wsum_cycles:
      //   Increment while weighted_sum_engine is accumulating.
      //   Measures output accumulation cost across all V tiles.
      //   In the tiled design: BR * NUM_KV_TILES accumulations
      //   occur per Q tile.
      // --------------------------------------------------------
      if (active && !done && wsum_busy) begin
        wsum_cycles <= wsum_cycles + 32'd1;
      end

      // --------------------------------------------------------
      // load_events:
      //   Increment on each tile load request pulse.
      //   Counts Q tile loads and KV tile loads separately then
      //   combined. Expected total:
      //     Q loads  = NUM_Q_TILES = 8
      //     KV loads = NUM_Q_TILES * BR * NUM_KV_TILES * 2
      //              = 8 * 4 * 8 * 2 = 512  (score pass + V pass)
      // --------------------------------------------------------
      if (active && !done) begin
        if (q_load_start && kv_load_start) begin
          // Simultaneous Q and KV load (unlikely in baseline but handled)
          load_events <= load_events + 32'd2;
        end else if (q_load_start || kv_load_start) begin
          load_events <= load_events + 32'd1;
        end
      end

      // --------------------------------------------------------
      // stall_cycles:
      //   Increment while stall_flag is asserted.
      //   Tied to 0 in baseline; reserved for future memory
      //   latency or backpressure modeling.
      // --------------------------------------------------------
      if (active && !done && stall_flag) begin
        stall_cycles <= stall_cycles + 32'd1;
      end

    end
  end

endmodule
