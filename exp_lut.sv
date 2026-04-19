// ============================================================
// exp_lut.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Approximate exp(x) for shifted attention scores used inside
//   softmax. This is a hardware-friendly approximation, not an
//   exact floating-point exponential.
//
// INPUT RANGE ASSUMPTION:
//   shifted_score = row_scores[i] - row_max, computed inside
//   row_sum_unit. Because row_max is the maximum value of the
//   row, the shifted value is always <= 0. Any positive input
//   is clamped to exp(0). Inputs below the supported negative
//   range (-8) are treated as negligible and return 0.
//
// APPROXIMATION CHOICE:
//   The input is quantized to integer steps over the range
//   [-8, 0]. Each integer bin maps to a precomputed constant:
//     exp_value = round(exp(x) * 65535)
//   This is a coarse 9-entry LUT with integer-bin resolution.
//
// SEQUENTIAL CONVERSION (pipeline register on output):
//   The 9-level priority if-else chain synthesizes as a deep
//   comparator tree that limits fmax when in the critical path.
//   A single pipeline register on exp_value breaks this path,
//   allowing the synthesizer to meet timing at a higher clock
//   frequency.
//
//   Latency: 1 cycle from valid shifted_score to valid exp_value.
//   Callers (row_sum_unit) must account for this 1-cycle delay.
//
// FIXED-POINT MEANING OF exp_value:
//   exp_value is an unsigned EXP_W=16-bit value in Q0.16 format.
//     exp(0)  → 0xFFFF = 65535  ≈ 1.000
//     exp(-1) → 0x5E2D = 24109  ≈ 0.368
//     exp(-8) → 0x0016 =    22  ≈ 0.000
//
// SIGNAL OWNERSHIP:
//   shifted_score  <- row_sum_unit (row_scores[i] - row_max)
//   exp_value      -> row_sum_unit (stored in exp_row[i] and
//                     accumulated into row_sum)
// ============================================================

import attention_pkg::*;

module exp_lut (
  input  logic                          clk,
  input  logic                          rst_n,

  // Shifted attention score: row_scores[i] - row_max
  // Expected to be <= 0 in normal operation.
  input  logic signed [SCORE_W-1:0]     shifted_score,

  // Approximate exp(shifted_score) in Q0.16 unsigned fixed-point.
  // Valid one cycle after shifted_score is presented.
  // exp(0) = 0xFFFF (~1.0), exp(very_negative) = 0x0000 (~0.0).
  output logic        [EXP_W-1:0]       exp_value
);

  // ----------------------------------------------------------
  // Combinational LUT: case-based ROM style.
  //
  // Precomputed values: round(exp(x) * 65535) for x = 0..-8
  // -------------------------------------------------------
  //  x  |  exp(x)   | * 65535 | hex
  // ----+-----------+---------+-------
  //   0 | 1.000000  |  65535  | 0xFFFF
  //  -1 | 0.367879  |  24109  | 0x5E2D
  //  -2 | 0.135335  |   8873  | 0x22A9
  //  -3 | 0.049787  |   3264  | 0x0CC0
  //  -4 | 0.018316  |   1201  | 0x04B1
  //  -5 | 0.006738  |    441  | 0x01B9
  //  -6 | 0.002479  |    162  | 0x00A2
  //  -7 | 0.000912  |     60  | 0x003C
  //  -8 | 0.000335  |     22  | 0x0016
  // < -8|  ~0        |      0  | 0x0000
  // ----------------------------------------------------------
  logic [EXP_W-1:0] exp_next;

  always_comb begin
    // Clamp positive inputs: shifted_score > 0 should not occur
    // in correct operation but is handled safely as exp(0).
    if (shifted_score >= 0)      exp_next = 16'hFFFF;  // exp(0)  ~ 1.000
    else if (shifted_score >= -1) exp_next = 16'h5E2D;  // exp(-1) ~ 0.368
    else if (shifted_score >= -2) exp_next = 16'h22A9;  // exp(-2) ~ 0.135
    else if (shifted_score >= -3) exp_next = 16'h0CC0;  // exp(-3) ~ 0.050
    else if (shifted_score >= -4) exp_next = 16'h04B1;  // exp(-4) ~ 0.018
    else if (shifted_score >= -5) exp_next = 16'h01B9;  // exp(-5) ~ 0.007
    else if (shifted_score >= -6) exp_next = 16'h00A2;  // exp(-6) ~ 0.002
    else if (shifted_score >= -7) exp_next = 16'h003C;  // exp(-7) ~ 0.001
    else if (shifted_score >= -8) exp_next = 16'h0016;  // exp(-8) ~ 0.000
    else                          exp_next = 16'h0000;  // < -8: negligible
  end

  // ----------------------------------------------------------
  // Pipeline register: breaks the comparator chain critical path.
  // exp_value is valid one cycle after shifted_score is presented.
  // Reset to 0x0000 (safe neutral value — contributes zero to sum).
  // ----------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      exp_value <= '0;
    end else begin
      exp_value <= exp_next;
    end
  end

endmodule