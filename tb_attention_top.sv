// ============================================================
// tb_attention_top.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Top-level simulation testbench for attention_top.
//   Drives clock, reset, and deterministic Q/K/V input data.
//   Pulses start, waits for done, then:
//     - Checks all performance counters are nonzero
//     - Runs a software reference model
//     - Compares every O_mem element against reference
//     - Reports TEST PASSED only if all outputs match
//
// REFERENCE MODEL ALGORITHM:
//   For each query row q:
//     1. score[q][k] = (sum_d Q[q][d]*K[k][d]) >>> 2  (scaled dot product)
//     2. Causal mask: score[q][k] = NEG_INF if k > q
//     3. row_max = max(score[q][:])
//     4. exp_row[k] = exp_lut_ref(score[q][k] - row_max)
//     5. row_sum = sum(exp_row[:])
//     6. prob[k] = (exp_row[k] * 0xFFFF) / row_sum  [normalizer direct division]
//     7. O[q][d] = sum_k( OUT_W'(prob[k] * V[k][d]) >>> EXP_W )
//
//   Step 6 mirrors normalizer.sv's direct division, NOT the two-step
//   reciprocal_lut approach. Direct division avoids catastrophic underflow
//   when a single token dominates (row_sum=0xFFFF):
//     recip=65536/65535=1, prob=(65535*1)>>16=0 -- WRONG (old approach)
//     prob=65535*65535/65535=65535              -- CORRECT (current RTL)
//   exp_lut_ref replicates exp_lut.sv exactly (same LUT values).
//
// THIS FILE IS NOT SYNTHESIZABLE.
// ============================================================

`timescale 1ns/1ps
import attention_pkg::*;

module tb_attention_top;

  // ==========================================================
  // SECTION 1: CLOCK AND RESET
  // ==========================================================

  localparam CLK_PERIOD    = 10;        // 10 ns -> 100 MHz
  localparam TIMEOUT_CYCLES = 500_000;  // generous upper bound

  logic clk;
  logic rst_n;
  logic start;

  initial clk = 1'b0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // ==========================================================
  // SECTION 2: DUT SIGNAL DECLARATIONS
  // ==========================================================

  logic signed [DATA_W-1:0] Q_mem [0:N-1][0:D-1];
  logic signed [DATA_W-1:0] K_mem [0:N-1][0:D-1];
  logic signed [DATA_W-1:0] V_mem [0:N-1][0:D-1];

  logic                     done;
  logic signed [OUT_W-1:0]  O_mem [0:N-1][0:D-1];

  logic [31:0] cycle_count;
  logic [31:0] score_cycles;
  logic [31:0] softmax_cycles;
  logic [31:0] wsum_cycles;
  logic [31:0] load_events;
  logic [31:0] stall_cycles;

  // ==========================================================
  // SECTION 3: DUT INSTANTIATION
  // ==========================================================

  attention_top dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .start         (start),
    .Q_mem         (Q_mem),
    .K_mem         (K_mem),
    .V_mem         (V_mem),
    .done          (done),
    .O_mem         (O_mem),
    .cycle_count   (cycle_count),
    .score_cycles  (score_cycles),
    .softmax_cycles(softmax_cycles),
    .wsum_cycles   (wsum_cycles),
    .load_events   (load_events),
    .stall_cycles  (stall_cycles)
  );

  // ==========================================================
  // SECTION 4: WAVEFORM DUMP
  // ==========================================================

  initial begin
    $dumpfile("tb_attention_top.vcd");
    $dumpvars(0, tb_attention_top);
  end

  // ==========================================================
  // SECTION 5: TIMEOUT WATCHDOG
  // ==========================================================

  int unsigned timeout_cnt;
  logic        run_started;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      run_started <= 1'b0;
    end else if (start) begin
      run_started <= 1'b1;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      timeout_cnt <= 0;
    end else if (run_started && !done) begin
      timeout_cnt <= timeout_cnt + 1;
      if (timeout_cnt >= TIMEOUT_CYCLES) begin
        $display("LOG: %0t : ERROR : tb_attention_top : dut.done : expected_value: 1'b1 actual_value: 1'b0", $time);
        $display("ERROR");
        $fatal(1, "TIMEOUT: done did not assert within %0d cycles", TIMEOUT_CYCLES);
      end
    end
  end

  // ==========================================================
  // SECTION 6: REFERENCE MODEL DECLARATIONS
  // ==========================================================

  // Storage for reference outputs and intermediate values
  logic signed [OUT_W-1:0]   ref_O      [0:N-1][0:D-1];
  logic signed [SCORE_W-1:0] ref_score  [0:N-1][0:N-1];
  logic signed [SCORE_W-1:0] ref_row_max;
  logic signed [SCORE_W-1:0] ref_shifted;
  logic        [EXP_W-1:0]   ref_exp_row[0:N-1];
  logic        [EXP_W-1:0]   ref_prob   [0:N-1];
  longint signed              ref_score_acc;
  longint unsigned            ref_row_sum;
  longint signed              ref_out_acc;
  integer i, d, q, k;
  int     err_count;

  // ----------------------------------------------------------
  // exp_lut_ref: mirrors exp_lut.sv LUT exactly.
  // Used by the reference model to produce identical exp values
  // to the hardware. Any change to exp_lut.sv must be reflected
  // here to keep the testbench aligned.
  // ----------------------------------------------------------
  function automatic logic [EXP_W-1:0] exp_lut_ref(
    input logic signed [SCORE_W-1:0] x
  );
    if      (x >= 0)  return 16'hFFFF;   // exp(0)  ~1.000
    else if (x >= -1) return 16'h5E2D;   // exp(-1) ~0.368
    else if (x >= -2) return 16'h22A9;   // exp(-2) ~0.135
    else if (x >= -3) return 16'h0CC0;   // exp(-3) ~0.050
    else if (x >= -4) return 16'h04B1;   // exp(-4) ~0.018
    else if (x >= -5) return 16'h01B9;   // exp(-5) ~0.007
    else if (x >= -6) return 16'h00A2;   // exp(-6) ~0.002
    else if (x >= -7) return 16'h003C;   // exp(-7) ~0.001
    else if (x >= -8) return 16'h0016;   // exp(-8) ~0.000
    else              return 16'h0000;   // masked or negligible
  endfunction

  // ==========================================================
  // SECTION 7: MAIN STIMULUS AND CHECKING
  // ==========================================================

  initial begin
    $display("TEST START");

    // --------------------------------------------------------
    // Phase 1: Assert reset
    // --------------------------------------------------------
    rst_n = 1'b0;
    start = 1'b0;
    repeat (5) @(posedge clk);

    // --------------------------------------------------------
    // Phase 2: Initialize Q, K, V memories
    //   Q_mem[i][d] = i + d          (positive diagonal ramp)
    //   K_mem[i][d] = i - d          (anti-diagonal ramp)
    //   V_mem[i][d] = (2*i + d) % 64 (wrapped positive ramp)
    //
    //   All values fit in DATA_W=8 signed bits (-128..127).
    //   Q max = 31+15 = 46,  K min = 0-15 = -15,  V max = 63.
    // --------------------------------------------------------
    for (i = 0; i < N; i++) begin
      for (d = 0; d < D; d++) begin
        Q_mem[i][d] = DATA_W'(signed'(i + d));
        K_mem[i][d] = DATA_W'(signed'(i - d));
        V_mem[i][d] = DATA_W'(signed'((2*i + d) % 64));
      end
    end

    // --------------------------------------------------------
    // Phase 3: Release reset and pulse start
    // --------------------------------------------------------
    @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    start = 1'b1;
    @(posedge clk);
    start = 1'b0;

    // --------------------------------------------------------
    // Phase 4: Wait for done
    // --------------------------------------------------------
    @(posedge clk);
    while (!done) begin
      @(posedge clk);
    end

    // --------------------------------------------------------
    // Phase 5: Print and sanity-check performance counters
    // --------------------------------------------------------
    $display("----------------------------------------------------");
    $display("PERFORMANCE COUNTERS");
    $display("----------------------------------------------------");
    $display("  cycle_count    = %0d", cycle_count);
    $display("  score_cycles   = %0d", score_cycles);
    $display("  softmax_cycles = %0d", softmax_cycles);
    $display("  wsum_cycles    = %0d", wsum_cycles);
    $display("  load_events    = %0d", load_events);
    $display("  stall_cycles   = %0d", stall_cycles);
    $display("----------------------------------------------------");

    if (cycle_count == 0) begin
      $display("LOG: %0t : ERROR : tb_attention_top : dut.cycle_count : expected_value: >0 actual_value: 0", $time);
      $display("ERROR"); $fatal(1, "cycle_count should be > 0");
    end
    if (score_cycles == 0) begin
      $display("LOG: %0t : ERROR : tb_attention_top : dut.score_cycles : expected_value: >0 actual_value: 0", $time);
      $display("ERROR"); $fatal(1, "score_cycles should be > 0");
    end
    if (softmax_cycles == 0) begin
      $display("LOG: %0t : ERROR : tb_attention_top : dut.softmax_cycles : expected_value: >0 actual_value: 0", $time);
      $display("ERROR"); $fatal(1, "softmax_cycles should be > 0");
    end
    if (wsum_cycles == 0) begin
      $display("LOG: %0t : ERROR : tb_attention_top : dut.wsum_cycles : expected_value: >0 actual_value: 0", $time);
      $display("ERROR"); $fatal(1, "wsum_cycles should be > 0");
    end
    if (load_events == 0) begin
      $display("LOG: %0t : ERROR : tb_attention_top : dut.load_events : expected_value: >0 actual_value: 0", $time);
      $display("ERROR"); $fatal(1, "load_events should be > 0");
    end

    $display("LOG: %0t : INFO : tb_attention_top : dut.cycle_count : expected_value: >0 actual_value: %0d", $time, cycle_count);
    $display("LOG: %0t : INFO : tb_attention_top : dut.load_events : expected_value: >0 actual_value: %0d", $time, load_events);

    // --------------------------------------------------------
    // Phase 6: Print O_mem output rows
    // --------------------------------------------------------
    $display("----------------------------------------------------");
    $display("OUTPUT MATRIX O_mem [%0d rows x %0d cols]", N, D);
    $display("----------------------------------------------------");
    for (i = 0; i < N; i++) begin
      $write("  O_mem[%02d] = [", i);
      for (d = 0; d < D; d++) begin
        if (d < D-1) $write("%0d, ", O_mem[i][d]);
        else         $write("%0d",   O_mem[i][d]);
      end
      $display("]");
    end
    $display("----------------------------------------------------");

    // --------------------------------------------------------
    // Phase 7: Software reference model
    //
    // Mirrors the hardware pipeline exactly:
    //   score  -> causal mask -> row_max -> exp_lut ->
    //   row_sum -> normalize (direct div) -> weighted sum
    //
    // All intermediate arithmetic uses longint (64-bit) to
    // avoid overflow before final truncation to OUT_W bits.
    // --------------------------------------------------------
    $display("----------------------------------------------------");
    $display("RUNNING REFERENCE MODEL...");
    $display("----------------------------------------------------");

    err_count = 0;

    for (q = 0; q < N; q++) begin

      // Step 1: Scaled dot-product scores + causal mask
      for (k = 0; k < N; k++) begin
        ref_score_acc = 0;
        for (d = 0; d < D; d++) begin
          ref_score_acc += longint'($signed(Q_mem[q][d])) *
                           longint'($signed(K_mem[k][d]));
        end
        // Arithmetic right-shift by 2 (matches score_engine SCALE_SHIFT)
        ref_score[q][k] = SCORE_W'(ref_score_acc >>> 2);
        // Causal mask: future keys are set to NEG_INF
        if (k > q) begin
          ref_score[q][k] = NEG_INF;
        end
      end

      // Step 2: Find row maximum (for softmax stability)
      ref_row_max = {1'b1, {(SCORE_W-1){1'b0}}};  // most-negative signed
      for (k = 0; k < N; k++) begin
        if (ref_score[q][k] > ref_row_max) begin
          ref_row_max = ref_score[q][k];
        end
      end

      // Step 3: Compute exp_row and accumulate row_sum
      ref_row_sum = 0;
      for (k = 0; k < N; k++) begin
        ref_shifted     = ref_score[q][k] - ref_row_max;
        ref_exp_row[k]  = exp_lut_ref(ref_shifted);
        ref_row_sum    += longint'(ref_exp_row[k]);
      end

      // Step 4: Normalize — mirrors normalizer.sv direct division exactly:
      //   prob[k] = (exp_row[k] * 0xFFFF) / row_sum
      //
      // Direct division avoids the two-step reciprocal underflow problem
      // when a single token dominates (row_sum=0xFFFF):
      //   recip=65536/65535=1, prob=(65535*1)>>16=0  -- WRONG (old approach)
      //   prob=65535*65535/65535=65535               -- CORRECT (this approach)
      for (k = 0; k < N; k++) begin
        if (ref_row_sum == 0) begin
          ref_prob[k] = '0;
        end else begin
          ref_prob[k] = EXP_W'(
            (longint'(ref_exp_row[k]) * longint'(32'hFFFF)) / ref_row_sum
          );
        end
      end

      // Step 5: Weighted sum over V  (matches weighted_sum_engine)
      // Hardware per-term: term = OUT_W'(prob * v) >>> EXP_W, then accumulate.
      // IMPORTANT: truncate each term to OUT_W BEFORE accumulating to match
      // the hardware's per-term right-shift (not end-of-sum shift).
      // sum(floor(x/k)) != floor(sum(x)/k) due to per-term truncation.
      for (d = 0; d < D; d++) begin
        ref_out_acc = 0;
        for (k = 0; k < N; k++) begin
          // Replicate: OUT_W'($signed(32b_prob) * $signed(32b_v)) >>> EXP_W
          // Product fits in 32 bits (max 65535*127=8,322,945 < 2^31).
          // int'() truncates to 32 bits (no loss), then >>> 16 is arithmetic.
          ref_out_acc += longint'(int'(longint'(ref_prob[k]) *
                                       longint'($signed(V_mem[k][d])))) >>> EXP_W;
        end
        ref_O[q][d] = OUT_W'(ref_out_acc);
      end
    end

    // --------------------------------------------------------
    // Phase 8: Element-wise comparison of O_mem vs reference
    // --------------------------------------------------------
    $display("COMPARING O_mem vs reference model...");
    for (q = 0; q < N; q++) begin
      for (d = 0; d < D; d++) begin
        if (O_mem[q][d] !== ref_O[q][d]) begin
          $display("LOG: %0t : ERROR : tb_attention_top : dut.O_mem[%0d][%0d] : expected_value: %0d actual_value: %0d",
                   $time, q, d, ref_O[q][d], O_mem[q][d]);
          err_count++;
        end else begin
          $display("LOG: %0t : INFO : tb_attention_top : dut.O_mem[%0d][%0d] : expected_value: %0d actual_value: %0d",
                   $time, q, d, ref_O[q][d], O_mem[q][d]);
        end
      end
    end

    if (err_count > 0) begin
      $display("----------------------------------------------------");
      $display("ERROR: %0d / %0d output elements MISMATCHED", err_count, N*D);
      $display("----------------------------------------------------");
      $display("ERROR");
      $fatal(1, "OUTPUT MISMATCH: %0d elements differ from reference", err_count);
    end

    $display("----------------------------------------------------");
    $display("All %0d output elements matched reference model", N*D);
    $display("----------------------------------------------------");
    $display("LOG: %0t : INFO : tb_attention_top : dut.done : expected_value: 1'b1 actual_value: 1'b1", $time);
    $display("TEST PASSED");
    $finish;
  end

endmodule
