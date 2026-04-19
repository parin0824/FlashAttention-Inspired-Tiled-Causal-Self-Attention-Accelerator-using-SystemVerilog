// ============================================================
// normalizer.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Normalize one row of exponentials into a probability row.
//   For each entry i:
//     prob_row[i] = (exp_row[i] * 0xFFFF) / row_sum
//
// PIPELINE CONVERSION (2-stage: separate multiply and divide):
//   Original critical path (all combinational in one cycle):
//     exp_row[idx] -> multiply by 0xFFFF -> divide by row_sum
//                  -> prob_row[idx] register
//
//   Split into two registered pipeline stages:
//   Stage 1 (comb + reg): exp_row[idx] * 0xFFFF  ->  num_reg
//   Stage 2 (comb + reg): num_reg / row_sum        ->  prob_row[idx_p]
//
//   Both inputs to the divider are now registers, giving a full
//   clock period budget for just the division — the biggest fmax win.
//
//   idx_p (1-cycle delayed counter) tracks which entry num_reg is for.
//   norm_first suppresses the first Stage 2 write (pipeline fill).
//   S_DRAIN flushes the final num_reg value.
//
// FSM: S_IDLE -> S_NORM -> S_DRAIN -> S_DONE -> S_IDLE
// Latency: N+2 cycles to done (+1 vs original).
//
// WHAT prob_row REPRESENTS:
//   prob_row[N] is the normalized attention weight row for one
//   query position. Each entry represents the relative attention
//   probability that the current query attends to key position i.
//   Entries corresponding to causally masked future positions
//   will be approximately zero because exp_lut already mapped
//   their NEG_INF-shifted scores to 0 in exp_row.
//   attention_top slices prob_row into prob_slice[BC] for each
//   KV tile before passing it to weighted_sum_engine.
//
// FIXED-POINT SCALING AND TRUNCATION:
//   Both exp_row[i] and row_sum_recip are EXP_W=16-bit unsigned
//   values in an approximately Q0.16 format, where the full
//   scale value 0xFFFF represents roughly 1.0.
//
//   The multiplication produces a 2*EXP_W = 32-bit product in
//   an approximately Q0.32 format (two fractional parts stacked).
//   Right-shifting by EXP_W=16 bits returns the result to Q0.16,
//   giving prob_row[i] in the same format as the inputs.
//
//   Shift amount is parameterized via EXP_W so that if the
//   data width changes, the scaling remains consistent.
//
//   Truncation (not rounding) is used for hardware simplicity.
//   This introduces at most 1 LSB error per entry, which is
//   acceptable for demo-quality hardware softmax.
//
// SIGNAL OWNERSHIP:
//   start         <- attention_controller
//   exp_row       <- row_sum_unit
//   row_sum_recip <- reciprocal_lut (combinational, stable at start)
//   prob_row      -> attention_top (prob_slice selection glue)
//   busy          -> attention_controller, perf_counters
//   done          -> attention_controller
// ============================================================

import attention_pkg::*;

module normalizer (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    start,

  // N exponential values from row_sum_unit, one per score position.
  // Entries for causally masked positions are approximately zero.
  input  logic [EXP_W-1:0]       exp_row [0:N-1],

  // Softmax denominator from row_sum_unit (direct, not reciprocal).
  // Using direct division avoids the two-step reciprocal approach
  // that loses all precision when row_sum ~ 2^EXP_W (single-token
  // case), where recip = 65536/65535 = 1 and (exp*1)>>16 = 0.
  input  logic [SUM_W-1:0]       row_sum,

  // Normalized attention weights for the current query row.
  // Values sum approximately to 1.0 in Q0.16-like representation.
  output logic [EXP_W-1:0]       prob_row [0:N-1],

  output logic                    busy,   // high while computing
  output logic                    done    // one-cycle pulse when complete
);

  // ----------------------------------------------------------
  // Index counters
  //   idx   : entry currently presented to Stage 1 (multiply)
  //   idx_p : one-cycle delayed — Stage 2 (divide) valid for this
  // ----------------------------------------------------------
  logic [$clog2(N)-1:0] idx;
  logic [$clog2(N)-1:0] idx_p;

  // Pipeline fill flag: suppress Stage 2 write on first S_NORM cycle
  logic norm_first;

  // ----------------------------------------------------------
  // Stage 1 pipeline register: captures multiply result.
  // num_reg at cycle T = exp_row[idx_p] * 0xFFFF from cycle T-1.
  // ----------------------------------------------------------
  logic [SUM_W-1:0] num_reg;

  // ----------------------------------------------------------
  // Stage 1 combinational: multiply only.
  // ----------------------------------------------------------
  logic [SUM_W-1:0] num_next;
  always_comb begin
    num_next = SUM_W'(exp_row[idx]) * 32'hFFFF;
  end

  // ----------------------------------------------------------
  // Stage 2 combinational: divide only.
  // Both inputs (num_reg, row_sum) are registered — full clock
  // period available for the division critical path.
  // ----------------------------------------------------------
  logic [SUM_W-1:0] prob_next;
  always_comb begin
    if (row_sum == '0)
      prob_next = '0;
    else
      prob_next = num_reg / row_sum;
  end

  typedef enum logic [2:0] {
    S_IDLE  = 3'd0,
    S_NORM  = 3'd1,
    S_DRAIN = 3'd2,
    S_DONE  = 3'd3
  } state_t;

  state_t state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      idx        <= '0;
      idx_p      <= '0;
      norm_first <= 1'b0;
      num_reg    <= '0;
      busy       <= 1'b0;
      done       <= 1'b0;
      for (int i = 0; i < N; i++) prob_row[i] <= '0;

    end else begin
      done    <= 1'b0;      // default: one-cycle pulse
      num_reg <= num_next;  // Stage 1 register: always active

      case (state)

        // --------------------------------------------------------
        // S_IDLE: wait for start.
        // --------------------------------------------------------
        S_IDLE: begin
          if (start && !busy) begin
            idx        <= '0;
            idx_p      <= '0;
            norm_first <= 1'b1;
            busy       <= 1'b1;
            state      <= S_NORM;
          end
        end

        // --------------------------------------------------------
        // S_NORM:
        //   Each cycle presents exp_row[idx] to Stage 1 (multiply).
        //   num_reg holds Stage 1 result for idx_p (previous cycle).
        //   Stage 2 divides num_reg by row_sum -> prob_next.
        //
        //   norm_first=1: pipeline filling, skip Stage 2 write.
        //   norm_first=0: write prob_row[idx_p] = EXP_W'(prob_next).
        //
        //   When idx=N-1: go to S_DRAIN to flush last num_reg.
        // --------------------------------------------------------
        S_NORM: begin
          if (!norm_first) begin
            prob_row[idx_p] <= EXP_W'(prob_next);
          end
          norm_first <= 1'b0;
          idx_p      <= idx;
          if (idx == $clog2(N)'(N - 1)) begin
            state <= S_DRAIN;
          end else begin
            idx <= idx + 1'b1;
          end
        end

        // --------------------------------------------------------
        // S_DRAIN:
        //   num_reg = exp_row[N-1] * 0xFFFF (last entry valid).
        //   Write prob_row[idx_p=N-1] and proceed to S_DONE.
        // --------------------------------------------------------
        S_DRAIN: begin
          prob_row[idx_p] <= EXP_W'(prob_next);
          state           <= S_DONE;
        end

        // --------------------------------------------------------
        // S_DONE: pulse done, deassert busy, return to IDLE.
        // --------------------------------------------------------
        S_DONE: begin
          done  <= 1'b1;
          busy  <= 1'b0;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule
