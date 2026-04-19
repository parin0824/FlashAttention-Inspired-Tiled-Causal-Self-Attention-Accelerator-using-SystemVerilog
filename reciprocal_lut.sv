// ============================================================
// reciprocal_lut.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Compute approximate 1 / row_sum in Q0.16 fixed-point format.
//   Output is: row_sum_recip ≈ 65536 / row_sum
//
//   For normalizer use: prob_row[i] = (exp_row[i] * row_sum_recip) >> 16
//
//   Implementation: integer division 2^16 / row_sum.
//
// SEQUENTIAL CONVERSION (pipeline register on output):
//   The combinational divider synthesizes as a deep logic chain
//   that limits fmax when placed in the timing path. Adding a
//   single pipeline register stage isolates the divide operation
//   to its own clock period, allowing the synthesizer to meet
//   timing at a higher frequency.
//
//   Latency: 1 cycle from input row_sum to valid row_sum_recip.
//
//   row_sum == 0 is guarded safely with output 0xFFFF.
//
// SIGNAL OWNERSHIP:
//   row_sum       <- row_sum_unit
//   row_sum_recip -> normalizer
// ============================================================

import attention_pkg::*;

module reciprocal_lut (
  input  logic              clk,
  input  logic              rst_n,

  input  logic [SUM_W-1:0]  row_sum,
  output logic [EXP_W-1:0]  row_sum_recip
);

  // ----------------------------------------------------------
  // Combinational reciprocal approximation.
  // Numerator is 2^16 = 65536 in a wider intermediate to avoid
  // truncation before the divide. The result is then clipped to
  // EXP_W bits with saturation at 0xFFFF.
  //
  // This combinational result is captured into a pipeline
  // register on the next rising clock edge.
  // ----------------------------------------------------------
  logic [SUM_W-1:0] recip_wide_comb;
  logic [EXP_W-1:0] recip_next;

  always_comb begin
    recip_wide_comb = '0;
    recip_next      = {EXP_W{1'b1}};   // default: zero-sum guard (0xFFFF)
    if (row_sum == '0) begin
      // Guard: if sum is zero (all scores were NEG_INF), return
      // maximum reciprocal so downstream stays numerically safe.
      recip_next = {EXP_W{1'b1}};      // 0xFFFF
    end else begin
      // Compute 65536 / row_sum; clip to 16 bits with saturation
      recip_wide_comb = (32'h0001_0000) / row_sum;
      recip_next      = (recip_wide_comb > {EXP_W{1'b1}}) ?
                        {EXP_W{1'b1}} :
                        EXP_W'(recip_wide_comb);
    end
  end

  // ----------------------------------------------------------
  // Pipeline register: breaks the divide combinational path.
  // row_sum_recip is valid one cycle after row_sum is presented.
  // Reset to 0xFFFF (safe reciprocal for zero-sum guard state).
  // ----------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      row_sum_recip <= {EXP_W{1'b1}};  // 0xFFFF — safe default
    end else begin
      row_sum_recip <= recip_next;
    end
  end

endmodule
