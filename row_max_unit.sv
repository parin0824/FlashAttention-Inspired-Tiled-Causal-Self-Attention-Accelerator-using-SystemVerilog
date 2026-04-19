// ============================================================
// row_max_unit.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// OPT-2: Replaced sequential N-cycle scan with a 5-level
//   binary comparison tree (log2(32) = 5 levels).
//
//   Level 1 (S_IDLE, start cycle): 16 pairwise maxima from 32 inputs
//   Level 2 (S_PIPE, cnt=1):        8 pairwise maxima
//   Level 3 (S_PIPE, cnt=2):        4 pairwise maxima
//   Level 4 (S_PIPE, cnt=3):        2 pairwise maxima → s4[0..1]
//   Level 5 + output (S_DONE):     final max(s4[0], s4[1]) → row_max
//
//   Latency: 5 cycles from start to done (was 33 cycles).
//   Saves 28 cycles per query row × 32 rows = 896 cycles total.
//
//   Tradeoff: Uses 16+8+4+2 = 30 extra SCORE_W-bit FFs for
//   intermediate stages vs 1 FF for the sequential version.
//   Justified when throughput matters more than area.
// ============================================================

import attention_pkg::*;

module row_max_unit (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          start,

  input  logic signed [SCORE_W-1:0]     row_scores [0:N-1],

  output logic signed [SCORE_W-1:0]     row_max,

  output logic                          busy,
  output logic                          done
);

  // ----------------------------------------------------------
  // Intermediate pipeline stage registers
  // OPT-2: 5-level binary tree; each level halves element count
  // ----------------------------------------------------------
  logic signed [SCORE_W-1:0] s1 [0:15];  // 16 values after level 1
  logic signed [SCORE_W-1:0] s2 [0:7];   // 8 values  after level 2
  logic signed [SCORE_W-1:0] s3 [0:3];   // 4 values  after level 3
  logic signed [SCORE_W-1:0] s4 [0:1];   // 2 values  after level 4
  // Final max computed in S_DONE directly into row_max

  // Pipeline stage counter (1..3 for S_PIPE)
  logic [1:0] pipe_cnt;

  typedef enum logic [1:0] {
    S_IDLE = 2'd0,
    S_PIPE = 2'd1,   // runs for pipe_cnt = 1, 2, 3
    S_DONE = 2'd2
  } state_t;

  state_t state;

  // ----------------------------------------------------------
  // Combinational max helper (avoids repeated ternary chains)
  // ----------------------------------------------------------
  function automatic logic signed [SCORE_W-1:0] smax(
    input logic signed [SCORE_W-1:0] a,
    input logic signed [SCORE_W-1:0] b
  );
    return (a > b) ? a : b;
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= S_IDLE;
      pipe_cnt <= '0;
      busy     <= 1'b0;
      done     <= 1'b0;
      row_max  <= '0;
      for (int i = 0; i < 16; i++) s1[i] <= '0;
      for (int i = 0; i < 8;  i++) s2[i] <= '0;
      for (int i = 0; i < 4;  i++) s3[i] <= '0;
      for (int i = 0; i < 2;  i++) s4[i] <= '0;

    end else begin
      // done is a one-cycle pulse; deassert by default
      done <= 1'b0;

      case (state)

        // --------------------------------------------------------
        // S_IDLE:
        //   On start: immediately compute level-1 comparisons
        //   from row_scores into s1[0..15].  This uses the first
        //   cycle productively (no wasted "setup" cycle).
        //   Transition to S_PIPE with pipe_cnt = 1 to indicate
        //   that s1 has been written and level 2 is next.
        // --------------------------------------------------------
        S_IDLE: begin
          if (start && !busy) begin
            // OPT-2: Level 1 — 16 pairwise maxima from 32 inputs
            for (int i = 0; i < 16; i++) begin
              s1[i] <= smax(row_scores[2*i], row_scores[2*i + 1]);
            end
            pipe_cnt <= 2'd1;
            busy     <= 1'b1;
            state    <= S_PIPE;
          end
        end

        // --------------------------------------------------------
        // S_PIPE:
        //   Each cycle advances one tree level.
        //   pipe_cnt tracks which level to compute next.
        //   s1 is valid at pipe_cnt=1 (written in S_IDLE start cycle).
        //   s2 valid at pipe_cnt=2, s3 at pipe_cnt=3.
        //   After s4 is written (pipe_cnt=3 → pipe_cnt becomes 3),
        //   we move to S_DONE.
        // --------------------------------------------------------
        S_PIPE: begin
          case (pipe_cnt)

            2'd1: begin
              // OPT-2: Level 2 — 8 pairwise maxima from s1
              for (int i = 0; i < 8; i++) begin
                s2[i] <= smax(s1[2*i], s1[2*i + 1]);
              end
              pipe_cnt <= 2'd2;
            end

            2'd2: begin
              // OPT-2: Level 3 — 4 pairwise maxima from s2
              for (int i = 0; i < 4; i++) begin
                s3[i] <= smax(s2[2*i], s2[2*i + 1]);
              end
              pipe_cnt <= 2'd3;
            end

            2'd3: begin
              // OPT-2: Level 4 — 2 pairwise maxima from s3
              for (int i = 0; i < 2; i++) begin
                s4[i] <= smax(s3[2*i], s3[2*i + 1]);
              end
              state <= S_DONE;
            end

            default: state <= S_IDLE;
          endcase
        end

        // --------------------------------------------------------
        // S_DONE:
        //   OPT-2: Level 5 — final comparison of s4[0] vs s4[1].
        //   Written directly into row_max.
        //   Pulse done for exactly one cycle. Deassert busy.
        //   row_max is stable from this cycle onward.
        //
        //   Total latency from start:
        //     Cycle 0 (S_IDLE, start): s1 written
        //     Cycle 1 (S_PIPE, cnt=1): s2 written
        //     Cycle 2 (S_PIPE, cnt=2): s3 written
        //     Cycle 3 (S_PIPE, cnt=3): s4 written, → S_DONE
        //     Cycle 4 (S_DONE):        row_max written, done=1
        //   done fires 4 clock cycles after start. WAIT_ROW_MAX = 5 cy.
        // --------------------------------------------------------
        S_DONE: begin
          row_max <= smax(s4[0], s4[1]);
          done    <= 1'b1;
          busy    <= 1'b0;
          state   <= S_IDLE;
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule
