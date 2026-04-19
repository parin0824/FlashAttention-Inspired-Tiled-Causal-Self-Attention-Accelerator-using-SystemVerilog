// ============================================================
// weighted_sum_engine.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// OPT-5 (preserved): Unrolled inner c-loop (BC=4 parallel multipliers).
//   Outer d-loop (D=16 cycles) remains sequential.
//   Baseline: BC*D=64 cy/tile. Optimized: D=16 cy/tile.
//
// OPT-ADDER-PIPE: Registered adder_sum to break multiply-tree critical path.
//   Original critical path in S_ACCUM (all combinational in one cycle):
//     prob_slice[c] * v_tile[c][d] -> shift -> term[c] ->
//     adder tree -> adder_sum -> acc[d] + adder_sum -> acc[d]
//
//   Adding a pipeline register after the adder tree splits this into:
//     Stage 1 (comb): BC multiplies + shifts + adder tree -> adder_sum_reg
//     Stage 2 (comb): acc[d_idx_p] + adder_sum_reg -> acc[d_idx_p]
//
//   d_idx_p (1-cycle delayed) tracks which acc[d] adder_sum_reg belongs to.
//   accum_first suppresses the first Stage 2 update (pipeline fill).
//   S_DRAIN flushes the final adder_sum_reg value.
//
// FSM: S_IDLE -> S_CLEAR/S_ACCUM -> S_DRAIN -> S_DONE -> S_IDLE
// Latency: D+2 cycles per tile (+1 vs original D+1).
//
// PURPOSE:
//   Accumulate one final attention output row across all V tiles
//   of one query row using a probability slice and the current
//   V tile.
//
// PARTIAL ACCUMULATION ACROSS V TILES:
//   This module is invoked once per KV tile during Pass B of the
//   controller loop. Each invocation adds the contribution of BC
//   key-value positions to the running accumulator acc[D]:
//     acc[d] += prob_slice[c] * v_tile[c][d]  for all c, d
//   After NUM_KV_TILES invocations, acc holds the complete
//   weighted sum across all N attended positions.
//
// WHY prob_slice IS TILE-LOCAL (NOT FULL prob_row):
//   prob_row[N] contains probabilities for all N key positions.
//   During Pass B, only the BC entries corresponding to the
//   current KV tile are needed. attention_top selects:
//     prob_slice[c] = prob_row[kv_tile_base + c]
//   Passing only BC values keeps the port width small and
//   decouples the engine from the full sequence length N.
//   The engine does not need to know which tile is active;
//   it simply trusts that prob_slice is correctly pre-selected.
//
// ROLE OF clear_acc IN SEPARATING QUERY ROWS:
//   The same accumulator is reused across multiple query rows
//   (one per q_local_row_idx per Q tile). The controller asserts
//   clear_acc only on the very first KV tile of each query row's
//   Pass B, zeroing acc before accumulation begins. On all
//   subsequent KV tiles for the same query row, clear_acc is
//   deasserted so the partial sums are preserved and built upon.
//   Without this mechanism, output rows would bleed into each
//   other across query row boundaries.
//
// SIGNAL OWNERSHIP:
//   clear_acc, start  <- attention_controller
//   prob_slice        <- attention_top glue
//                        (prob_row[kv_tile_base + c])
//   v_tile            <- kv_buffer
//   out_row           -> output_buffer (consumed after last tile)
//   busy              -> attention_controller, perf_counters
//   done              -> attention_controller
// ============================================================

import attention_pkg::*;

module weighted_sum_engine (
  input  logic                          clk,
  input  logic                          rst_n,

  // Asserted by controller on the first KV tile of each query row.
  // Clears acc before accumulation starts for that row.
  // Deasserted on all subsequent KV tiles of the same query row.
  input  logic                          clear_acc,

  input  logic                          start,

  // BC-wide probability slice for the current KV tile.
  // Selected by attention_top: prob_slice[c] = prob_row[kv_tile_base+c]
  input  logic [EXP_W-1:0]             prob_slice [0:BC-1],

  // Current KV tile values from kv_buffer.
  input  logic signed [DATA_W-1:0]      v_tile [0:BC-1][0:D-1],

  // Accumulated output row. Reflects the internal accumulator at
  // all times. Valid as the final result after the last KV tile.
  output logic signed [OUT_W-1:0]       out_row [0:D-1],

  output logic                          busy,   // high while accumulating
  output logic                          done    // one-cycle pulse per tile
);

  // ----------------------------------------------------------
  // Internal accumulator: OUT_W signed bits per output dimension.
  // Retains partial sums across multiple KV tile invocations.
  // Only cleared when clear_acc is asserted at start.
  // ----------------------------------------------------------
  logic signed [OUT_W-1:0] acc [0:D-1];

  // out_row continuously mirrors the accumulator so the final
  // result is available to output_buffer without an extra copy.
  always_comb begin
    for (int d = 0; d < D; d++) begin
      out_row[d] = acc[d];
    end
  end

  // ----------------------------------------------------------
  // d-dimension counters
  //   d_idx   : current dimension presented to Stage 1
  //   d_idx_p : one-cycle delayed — adder_sum_reg valid for this
  // ----------------------------------------------------------
  logic [$clog2(D)-1:0] d_idx;
  logic [$clog2(D)-1:0] d_idx_p;

  // Pipeline fill flag: suppress Stage 2 update on first S_ACCUM cycle
  logic accum_first;

  // ----------------------------------------------------------
  // Stage 1 combinational: BC parallel MACs + adder tree.
  //   term[c]    = (prob_slice[c] * v_tile[c][d_idx]) >>> EXP_W
  //   adder_next = (term[0]+term[1]) + (term[2]+term[3])
  // adder_next is registered into adder_sum_reg every cycle.
  // ----------------------------------------------------------
  logic signed [OUT_W-1:0] term       [0:BC-1];
  logic signed [OUT_W-1:0] adder_next;
  logic signed [OUT_W-1:0] adder_sum_reg;

  always_comb begin
    for (int c = 0; c < BC; c++) begin
      term[c] = OUT_W'(
        $signed({{(OUT_W - EXP_W){1'b0}}, prob_slice[c]}) *
        $signed({{(OUT_W - DATA_W){v_tile[c][d_idx][DATA_W-1]}},
                  v_tile[c][d_idx]})
      ) >>> EXP_W;
    end
    // Two-level adder tree for BC=4
    adder_next = (term[0] + term[1]) + (term[2] + term[3]);
  end

  typedef enum logic [2:0] {
    S_IDLE  = 3'd0,
    S_CLEAR = 3'd1,
    S_ACCUM = 3'd2,
    S_DRAIN = 3'd3,
    S_DONE  = 3'd4
  } state_t;

  state_t state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state         <= S_IDLE;
      d_idx         <= '0;
      d_idx_p       <= '0;
      accum_first   <= 1'b0;
      adder_sum_reg <= '0;
      busy          <= 1'b0;
      done          <= 1'b0;
      for (int i = 0; i < D; i++) acc[i] <= '0;

    end else begin
      done          <= 1'b0;        // default: one-cycle pulse
      adder_sum_reg <= adder_next;  // Stage 1 register: always active

      case (state)

        // --------------------------------------------------------
        // S_IDLE:
        //   Wait for start. Initialise pipeline state.
        //   If clear_acc: zero acc in S_CLEAR first.
        //   Otherwise go directly to S_ACCUM (retain partial sums).
        // --------------------------------------------------------
        S_IDLE: begin
          if (start && !busy) begin
            d_idx       <= '0;
            d_idx_p     <= '0;
            accum_first <= 1'b1;
            busy        <= 1'b1;
            if (clear_acc) begin
              state <= S_CLEAR;
            end else begin
              state <= S_ACCUM;
            end
          end
        end

        // --------------------------------------------------------
        // S_CLEAR:
        //   Zero all D accumulator entries in one cycle.
        //   Proceeds immediately to S_ACCUM.
        // --------------------------------------------------------
        S_CLEAR: begin
          for (int i = 0; i < D; i++) begin
            acc[i] <= '0;
          end
          d_idx <= '0;
          state <= S_ACCUM;
        end

        // --------------------------------------------------------
        // S_ACCUM:
        //   Each cycle presents d_idx to Stage 1 (multiply + tree).
        //   adder_sum_reg holds the Stage 1 result for d_idx_p.
        //
        //   accum_first=1: pipeline filling, skip Stage 2 update.
        //   accum_first=0: acc[d_idx_p] += adder_sum_reg.
        //
        //   When d_idx = D-1: go to S_DRAIN to flush last result.
        // --------------------------------------------------------
        S_ACCUM: begin
          if (!accum_first) begin
            acc[d_idx_p] <= acc[d_idx_p] + adder_sum_reg;
          end

          accum_first <= 1'b0;
          d_idx_p     <= d_idx;

          if (d_idx == $clog2(D)'(D - 1)) begin
            state <= S_DRAIN;
          end else begin
            d_idx <= d_idx + 1'b1;
          end
        end

        // --------------------------------------------------------
        // S_DRAIN:
        //   adder_sum_reg holds the Stage 1 result for d_idx_p=D-1.
        //   Apply the final accumulator update and go to S_DONE.
        // --------------------------------------------------------
        S_DRAIN: begin
          acc[d_idx_p] <= acc[d_idx_p] + adder_sum_reg;
          state        <= S_DONE;
        end

        // --------------------------------------------------------
        // S_DONE:
        //   One tile contribution fully absorbed into acc.
        //   Pulse done. out_row mirrors acc (combinational).
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
