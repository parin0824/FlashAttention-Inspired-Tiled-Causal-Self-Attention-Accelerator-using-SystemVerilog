// row_sum_unit.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Softmax denominator generator. For one completed attention
//   score row, computes:
//     shifted_score = row_scores[i] - row_max
//     exp_row[i]    = exp_lut(shifted_score)
//     row_sum      += exp_row[i]
//
// EXP_LUT PIPELINE ADJUSTMENT:
//   exp_lut now has a 1-cycle registered output (sequential).
//   The shifted_score driven in cycle T produces a valid
//   exp_value in cycle T+1. To absorb this latency with minimal
//   overhead, S_SCAN uses a two-pointer scheme:
//
//     idx      : current index being presented to exp_lut
//     idx_prev : one-cycle delayed index (which exp_value belongs to)
//     scan_first: high only on the first S_SCAN cycle; suppresses
//                 capture since exp_value is not yet valid
//
//   Flow:
//     Cycle 0  (scan_first=1): present idx=0 → exp_lut; skip capture
//     Cycle 1..N-1 (scan_first=0, idx=1..N-1):
//       capture exp_value for idx_prev, advance idx
//     At idx=N-1: transition to S_DRAIN (present nothing new)
//     S_DRAIN: capture exp_value for idx_prev=N-1 (last entry)
//     S_DONE : pulse done
//
//   Total cycles: N+2 per row (was N). Overhead: 2 cycles only.
//
// ROLE OF row_max SUBTRACTION:
//   Subtracting row_max before exponentiation is the standard
//   numerically stable softmax trick. Without it, large positive
//   scores cause fixed-point overflow in exp. After subtraction,
//   all shifted values are <= 0, so exp_lut outputs stay in
//   the range (0, 1], which fits cleanly in Q0.16 format.
//   The final softmax probabilities are mathematically unchanged
//   because the max term cancels in numerator and denominator.
//
// WHY exp_row IS STORED AND NOT JUST ACCUMULATED:
//   row_sum alone is insufficient. The normalizer needs all N
//   individual exp_row[i] values to compute:
//     prob_row[i] = (exp_row[i] * row_sum_recip) >> 16
//   If only the sum were kept, each probability could not be
//   recovered. Storing exp_row[N] allows the normalizer to make
//   one sequential pass over the array to generate prob_row[N].
//
// NEG_INF HANDLING:
//   Causally masked positions in row_scores contain NEG_INF.
//   After subtracting row_max, those entries become very large
//   negative numbers. exp_lut clamps them to 0x0000, so they
//   contribute zero to both exp_row and row_sum, correctly
//   producing zero attention weight on future tokens.
//
// SIGNAL OWNERSHIP:
//   start       <- attention_controller
//   row_scores  <- row_score_store
//   row_max     <- row_max_unit (must be valid before start)
//   exp_row     -> normalizer
//   row_sum     -> reciprocal_lut
//   busy        -> attention_controller, perf_counters
//   done        -> attention_controller
// ============================================================

import attention_pkg::*;

module row_sum_unit (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          start,

  // Full row of N masked scaled scores from row_score_store.
  // May contain NEG_INF for causally masked future-key positions.
  input  logic signed [SCORE_W-1:0]     row_scores [0:N-1],

  // Row maximum from row_max_unit; must be stable before start.
  input  logic signed [SCORE_W-1:0]     row_max,

  // Shifted exponentials, one per score entry (Q0.16 format).
  // Stored for later use by normalizer to produce prob_row[N].
  output logic        [EXP_W-1:0]       exp_row [0:N-1],

  // Softmax denominator: sum of all exp_row entries.
  // Consumed by reciprocal_lut to compute 1/row_sum.
  output logic        [SUM_W-1:0]       row_sum,

  output logic                          busy,   // high while computing
  output logic                          done    // one-cycle pulse when complete
);

  // ----------------------------------------------------------
  // exp_lut instantiation (now sequential: 1-cycle latency)
  // shifted_score is driven combinationally from idx so that
  // exp_value arrives on the following cycle for idx_prev.
  // ----------------------------------------------------------
  logic signed [SCORE_W-1:0] shifted_score;
  logic        [EXP_W-1:0]   exp_value;

  exp_lut u_exp_lut (
    .clk           (clk),
    .rst_n         (rst_n),
    .shifted_score (shifted_score),
    .exp_value     (exp_value)
  );

  // ----------------------------------------------------------
  // Index pointers
  //   idx      : index being presented to exp_lut this cycle
  //   idx_prev : one-cycle delayed idx (exp_value is valid for this)
  //   scan_first: suppresses capture on the first S_SCAN cycle
  // ----------------------------------------------------------
  logic [$clog2(N)-1:0] idx;
  logic [$clog2(N)-1:0] idx_prev;
  logic                  scan_first;

  // Combinational: feed shifted score for current idx to exp_lut
  always_comb begin
    shifted_score = row_scores[idx] - row_max;
  end

  // ----------------------------------------------------------
  // FSM
  // ----------------------------------------------------------
  typedef enum logic [1:0] {
    S_IDLE  = 2'd0,
    S_SCAN  = 2'd1,
    S_DRAIN = 2'd2,   // absorb last exp_value after idx reaches N-1
    S_DONE  = 2'd3
  } state_t;

  state_t state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      idx        <= '0;
      idx_prev   <= '0;
      scan_first <= 1'b0;
      row_sum    <= '0;
      busy       <= 1'b0;
      done       <= 1'b0;
      for (int i = 0; i < N; i++) exp_row[i] <= '0;

    end else begin
      // done is a one-cycle pulse; deassert by default
      done <= 1'b0;

      case (state)

        // --------------------------------------------------------
        // S_IDLE:
        //   Wait for start. Ignore repeated start while busy.
        //   Reset row_sum and idx at the beginning of each new
        //   run so stale values from a previous row do not
        //   corrupt the new accumulation.
        // --------------------------------------------------------
        S_IDLE: begin
          if (start && !busy) begin
            idx        <= '0;
            idx_prev   <= '0;
            row_sum    <= '0;   // must reset for every new start
            scan_first <= 1'b1; // first cycle: exp_value not valid yet
            busy       <= 1'b1;
            state      <= S_SCAN;
          end
        end

        // --------------------------------------------------------
        // S_SCAN:
        //   Presents shifted_score for idx to exp_lut each cycle.
        //   exp_value is valid for idx_prev (one cycle behind).
        //
        //   Cycle 0 (scan_first=1):
        //     - shifted_score[0] driven to exp_lut
        //     - No capture (exp_value not valid yet)
        //     - idx advances to 1; idx_prev <= 0; scan_first <= 0
        //
        //   Cycle 1..N-1 (scan_first=0):
        //     - exp_value = exp(shifted_score[idx_prev]) is valid
        //     - exp_row[idx_prev] and row_sum updated
        //     - idx advances; idx_prev tracks one cycle behind
        //
        //   When idx reaches N-1: last shifted_score presented,
        //   transition to S_DRAIN to capture the final exp_value.
        // --------------------------------------------------------
        S_SCAN: begin
          if (!scan_first) begin
            // exp_value is valid for idx_prev — capture it
            exp_row[idx_prev] <= exp_value;
            row_sum           <= row_sum + SUM_W'(exp_value);
          end
          scan_first <= 1'b0;
          idx_prev   <= idx;

          if (idx == $clog2(N)'(N - 1)) begin
            // Last entry presented to exp_lut; drain next cycle
            state <= S_DRAIN;
          end else begin
            idx <= idx + 1'b1;
          end
        end

        // --------------------------------------------------------
        // S_DRAIN:
        //   exp_value for idx_prev = N-1 is now valid (1-cycle
        //   pipeline flush). Capture the final entry.
        //   exp_row[0:N-1] and row_sum are both stable after this.
        //   Proceed to S_DONE.
        // --------------------------------------------------------
        S_DRAIN: begin
          exp_row[idx_prev] <= exp_value;
          row_sum           <= row_sum + SUM_W'(exp_value);
          state             <= S_DONE;
        end

        // --------------------------------------------------------
        // S_DONE:
        //   Pulse done for exactly one cycle.
        //   exp_row[0:N-1] and row_sum are both stable and valid.
        //   Deassert busy. Return to IDLE.
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
