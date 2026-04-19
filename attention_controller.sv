// ============================================================
// attention_controller.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Central FSM that sequences the complete tiled causal
//   self-attention algorithm. Drives all start/enable signals,
//   maintains tile and row loop counters, and asserts done
//   after all N output rows have been written.
//
// ALGORITHM LOOP STRUCTURE (canonical truth):
//
//   for q_tile_idx = 0..NUM_Q_TILES-1:       // 8 Q tiles
//     load Q tile
//     for q_local_row_idx = 0..BR-1:         // 4 query rows per tile
//       clear row_score_store
//       // Pass A: score collection
//       for kv_tile_idx = 0..NUM_KV_TILES-1: // 8 KV tiles
//         load KV tile
//         compute score_tile
//         store row fragment
//       // softmax
//       compute row_max
//       compute exp_row + row_sum
//       normalize -> prob_row
//       // Pass B: output accumulation
//       for kv_tile_idx = 0..NUM_KV_TILES-1: // 8 KV tiles
//         load KV tile
//         accumulate prob_slice * v_tile (clear acc on first tile only)
//       write output row
//
// SIGNAL OWNERSHIP:
//   all done inputs  <- datapath submodules
//   all start/en outputs -> datapath submodules
//   q_tile_idx, kv_tile_idx, q_local_row_idx -> addr_gen
// ============================================================

import attention_pkg::*;

module attention_controller (
  input  logic                        clk,
  input  logic                        rst_n,

  // ---- Top-level run control ----
  input  logic                        start,

  // ---- Done inputs from datapath modules ----
  input  logic                        q_load_done,
  input  logic                        kv_load_done,
  input  logic                        score_done,
  input  logic                        row_store_done,
  input  logic                        row_valid,
  input  logic                        row_max_done,
  input  logic                        row_sum_done,
  input  logic                        norm_done,
  input  logic                        wsum_done,
  input  logic                        out_write_done,

  // ---- Control outputs to datapath modules ----
  output logic                        q_load_start,
  output logic                        kv_load_start,
  output logic                        score_start,
  output logic                        row_store_start_row,
  output logic                        row_store_en,
  output logic                        row_max_start,
  output logic                        row_sum_start,
  output logic                        norm_start,
  output logic                        wsum_clear_acc,
  output logic                        wsum_start,
  output logic                        out_write_en,

  // ---- Loop index outputs to addr_gen ----
  output logic [Q_TILE_IDX_W-1:0]    q_tile_idx,
  output logic [KV_TILE_IDX_W-1:0]   kv_tile_idx,
  output logic [BR_IDX_W-1:0]        q_local_row_idx,

  // ---- Completion flag ----
  output logic                        done
);

  // ----------------------------------------------------------
  // State register
  // ----------------------------------------------------------
  ctrl_state_t state, next_state;

  // ----------------------------------------------------------
  // Index registers
  // Driven exclusively in always_ff; read in always_comb
  // ----------------------------------------------------------
  logic [Q_TILE_IDX_W-1:0]  q_tile_idx_r;
  logic [KV_TILE_IDX_W-1:0] kv_tile_idx_r;
  logic [BR_IDX_W-1:0]      q_local_row_idx_r;

  // Connect internal registers to output ports
  assign q_tile_idx      = q_tile_idx_r;
  assign kv_tile_idx     = kv_tile_idx_r;
  assign q_local_row_idx = q_local_row_idx_r;

  // ----------------------------------------------------------
  // State register: synchronous update, async active-low reset
  // ----------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state             <= IDLE;
      q_tile_idx_r      <= '0;
      kv_tile_idx_r     <= '0;
      q_local_row_idx_r <= '0;
    end else begin
      state <= next_state;

      // --------------------------------------------------------
      // Index updates: each index increments at the transition
      // out of its corresponding NEXT_* state so that the new
      // index is stable before the following state reads it.
      // --------------------------------------------------------

      case (state)

        // -------------------------------------------------------
        // NEXT_KV_FOR_SCORE:
        //   kv_tile_idx advances to the next KV tile in Pass A.
        //   If all KV tiles done, kv_tile_idx resets for reuse.
        // -------------------------------------------------------
        NEXT_KV_FOR_SCORE: begin
          if (kv_tile_idx_r == KV_TILE_IDX_W'(NUM_KV_TILES - 1)) begin
            kv_tile_idx_r <= '0;  // reset for Pass B
          end else begin
            kv_tile_idx_r <= kv_tile_idx_r + KV_TILE_IDX_W'(1);
          end
        end

        // -------------------------------------------------------
        // NEXT_KV_FOR_V:
        //   kv_tile_idx advances to the next KV tile in Pass B.
        //   If all V tiles done, kv_tile_idx resets for next row.
        // -------------------------------------------------------
        NEXT_KV_FOR_V: begin
          if (kv_tile_idx_r == KV_TILE_IDX_W'(NUM_KV_TILES - 1)) begin
            kv_tile_idx_r <= '0;  // reset for next query row
          end else begin
            kv_tile_idx_r <= kv_tile_idx_r + KV_TILE_IDX_W'(1);
          end
        end

        // -------------------------------------------------------
        // NEXT_Q_ROW:
        //   q_local_row_idx advances to the next row within the
        //   current Q tile. kv_tile_idx was already reset in
        //   NEXT_KV_FOR_V so no reset needed here.
        // -------------------------------------------------------
        NEXT_Q_ROW: begin
          if (q_local_row_idx_r == BR_IDX_W'(BR - 1)) begin
            q_local_row_idx_r <= '0;  // reset for next Q tile
          end else begin
            q_local_row_idx_r <= q_local_row_idx_r + BR_IDX_W'(1);
          end
        end

        // -------------------------------------------------------
        // NEXT_Q_TILE:
        //   q_tile_idx advances to the next Q tile.
        //   q_local_row_idx was already reset in NEXT_Q_ROW.
        // -------------------------------------------------------
        NEXT_Q_TILE: begin
          if (q_tile_idx_r == Q_TILE_IDX_W'(NUM_Q_TILES - 1)) begin
            q_tile_idx_r <= '0;  // all done, reset for safety
          end else begin
            q_tile_idx_r <= q_tile_idx_r + Q_TILE_IDX_W'(1);
          end
        end

        default: ; // no index change in other states

      endcase
    end
  end

  // ----------------------------------------------------------
  // Next-state logic and output decode
  // All outputs default to 0; only the active state asserts them.
  // ----------------------------------------------------------
  always_comb begin
    // ---- Safe defaults for all outputs ----
    next_state           = state;
    q_load_start         = 1'b0;
    kv_load_start        = 1'b0;
    score_start          = 1'b0;
    row_store_start_row  = 1'b0;
    row_store_en         = 1'b0;
    row_max_start        = 1'b0;
    row_sum_start        = 1'b0;
    norm_start           = 1'b0;
    wsum_clear_acc       = 1'b0;
    wsum_start           = 1'b0;
    out_write_en         = 1'b0;
    done                 = 1'b0;

    case (state)

      // --------------------------------------------------------
      // IDLE:
      //   Wait for top-level start pulse.
      //   Indices are held at 0 from reset.
      // --------------------------------------------------------
      IDLE: begin
        if (start) begin
          next_state = LOAD_Q_TILE;
        end
      end

      // --------------------------------------------------------
      // LOAD_Q_TILE:
      //   Assert q_load_start for one cycle to initiate Q tile
      //   copy from Q_mem into q_buffer.
      //   q_tile_idx is already set to the correct value.
      // --------------------------------------------------------
      LOAD_Q_TILE: begin
        q_load_start = 1'b1;
        next_state   = WAIT_Q_TILE;
      end

      // --------------------------------------------------------
      // WAIT_Q_TILE:
      //   Hold until q_buffer asserts q_load_done.
      //   q_tile is stable and valid after this.
      // --------------------------------------------------------
      WAIT_Q_TILE: begin
        if (q_load_done) begin
          next_state = INIT_ROW;
        end
      end

      // --------------------------------------------------------
      // INIT_ROW:
      //   Assert row_store_start_row for one cycle to clear the
      //   row_score_store for the upcoming score collection pass.
      //   q_local_row_idx is the active query row within this tile.
      //   kv_tile_idx is 0 (reset from previous pass or from reset).
      // --------------------------------------------------------
      INIT_ROW: begin
        row_store_start_row = 1'b1;
        next_state          = LOAD_KV_FOR_SCORE;
      end

      // --------------------------------------------------------
      // LOAD_KV_FOR_SCORE (Pass A):
      //   Assert kv_load_start to load the next K/V tile into
      //   kv_buffer. kv_tile_idx selects the tile.
      // --------------------------------------------------------
      LOAD_KV_FOR_SCORE: begin
        kv_load_start = 1'b1;
        next_state    = WAIT_KV_FOR_SCORE;
      end

      // --------------------------------------------------------
      // WAIT_KV_FOR_SCORE (Pass A):
      //   Hold until kv_buffer asserts kv_load_done.
      //   k_tile and v_tile are stable and valid after this.
      // --------------------------------------------------------
      WAIT_KV_FOR_SCORE: begin
        if (kv_load_done) begin
          next_state = START_SCORE;
        end
      end

      // --------------------------------------------------------
      // START_SCORE:
      //   Assert score_start for one cycle.
      //   score_engine will compute score_tile = q_tile * k_tile^T
      //   and scale by >>> 2.
      // --------------------------------------------------------
      START_SCORE: begin
        score_start = 1'b1;
        next_state  = WAIT_SCORE;
      end

      // --------------------------------------------------------
      // WAIT_SCORE:
      //   Hold until score_engine asserts score_done.
      //   score_tile is stable and has been masked by mask_unit.
      // --------------------------------------------------------
      WAIT_SCORE: begin
        if (score_done) begin
          next_state = STORE_ROW_FRAGMENT;
        end
      end

      // --------------------------------------------------------
      // STORE_ROW_FRAGMENT:
      //   Assert row_store_en for one cycle.
      //   row_score_store writes the BC-wide row fragment for
      //   q_local_row_idx into row_scores[kv_tile_idx*BC : +BC-1].
      // --------------------------------------------------------
      STORE_ROW_FRAGMENT: begin
        row_store_en = 1'b1;
        next_state   = NEXT_KV_FOR_SCORE;
      end

      // --------------------------------------------------------
      // NEXT_KV_FOR_SCORE:
      //   row_store_done is waited on implicitly via one-cycle
      //   row_store_en. The index update happens in always_ff.
      //   If this was the last KV tile, move to softmax.
      //   Otherwise loop back to load the next KV tile.
      // --------------------------------------------------------
      NEXT_KV_FOR_SCORE: begin
        if (row_store_done) begin
          if (kv_tile_idx_r == KV_TILE_IDX_W'(NUM_KV_TILES - 1)) begin
            // All KV tiles processed for this query row —
            // row_scores[N] is now complete. Move to softmax.
            next_state = START_ROW_MAX;
          end else begin
            // More KV tiles remain in Pass A
            next_state = LOAD_KV_FOR_SCORE;
          end
        end
      end

      // --------------------------------------------------------
      // START_ROW_MAX:
      //   Assert row_max_start for one cycle.
      //   row_max_unit will scan row_scores[N] and find the max.
      // --------------------------------------------------------
      START_ROW_MAX: begin
        row_max_start = 1'b1;
        next_state    = WAIT_ROW_MAX;
      end

      // --------------------------------------------------------
      // WAIT_ROW_MAX:
      //   Hold until row_max_unit asserts row_max_done.
      //   row_max is stable after this.
      // --------------------------------------------------------
      WAIT_ROW_MAX: begin
        if (row_max_done) begin
          next_state = START_ROW_SUM;
        end
      end

      // --------------------------------------------------------
      // START_ROW_SUM:
      //   Assert row_sum_start for one cycle.
      //   row_sum_unit will compute exp_row[N] and row_sum.
      // --------------------------------------------------------
      START_ROW_SUM: begin
        row_sum_start = 1'b1;
        next_state    = WAIT_ROW_SUM;
      end

      // --------------------------------------------------------
      // WAIT_ROW_SUM:
      //   Hold until row_sum_unit asserts row_sum_done.
      //   exp_row[N] and row_sum are stable after this.
      //   reciprocal_lut is combinational and updates immediately.
      // --------------------------------------------------------
      WAIT_ROW_SUM: begin
        if (row_sum_done) begin
          next_state = START_NORMALIZE;
        end
      end

      // --------------------------------------------------------
      // START_NORMALIZE:
      //   Assert norm_start for one cycle.
      //   normalizer will compute prob_row[N] = exp_row * recip.
      // --------------------------------------------------------
      START_NORMALIZE: begin
        norm_start = 1'b1;
        next_state = WAIT_NORMALIZE;
      end

      // --------------------------------------------------------
      // WAIT_NORMALIZE:
      //   Hold until normalizer asserts norm_done.
      //   prob_row[N] is stable and ready for weighted sum.
      //
      //   OPT-6: Eliminated INIT_WSUM state (was a 1-cycle idle).
      //   Transition directly to LOAD_KV_FOR_V here.
      //   kv_tile_idx was already reset to 0 in the last
      //   NEXT_KV_FOR_SCORE transition; no extra bookkeeping needed.
      //   Saves 1 cycle per query row × 32 rows = 32 cycles total.
      // --------------------------------------------------------
      WAIT_NORMALIZE: begin
        if (norm_done) begin
          next_state = LOAD_KV_FOR_V;  // OPT-6: was INIT_WSUM
        end
      end

      // INIT_WSUM state is retained in attention_pkg enum for
      // compatibility but is no longer reachable in this FSM.
      // The default branch handles it safely if ever entered.
      INIT_WSUM: begin
        // OPT-6: dead state — should never be reached
        next_state = LOAD_KV_FOR_V;
      end

      // --------------------------------------------------------
      // LOAD_KV_FOR_V (Pass B):
      //   Assert kv_load_start to reload K/V tile.
      //   Even though only v_tile is used in this pass, both K
      //   and V are loaded together to keep the interface simple.
      // --------------------------------------------------------
      LOAD_KV_FOR_V: begin
        kv_load_start = 1'b1;
        next_state    = WAIT_KV_FOR_V;
      end

      // --------------------------------------------------------
      // WAIT_KV_FOR_V (Pass B):
      //   Hold until kv_buffer asserts kv_load_done.
      //   v_tile is stable and ready for weighted_sum_engine.
      // --------------------------------------------------------
      WAIT_KV_FOR_V: begin
        if (kv_load_done) begin
          next_state = START_WSUM;
        end
      end

      // --------------------------------------------------------
      // START_WSUM:
      //   Assert wsum_start for one cycle.
      //   wsum_clear_acc is asserted ONLY on the first KV tile
      //   (kv_tile_idx == 0) to zero the accumulator before
      //   beginning a new output row accumulation.
      //   For all subsequent tiles, the accumulator retains its
      //   partial sum from the previous tile.
      // --------------------------------------------------------
      START_WSUM: begin
        wsum_start = 1'b1;
        if (kv_tile_idx_r == KV_TILE_IDX_W'(0)) begin
          wsum_clear_acc = 1'b1;  // first tile: clear accumulator
        end else begin
          wsum_clear_acc = 1'b0;  // subsequent tiles: accumulate
        end
        next_state = WAIT_WSUM;
      end

      // --------------------------------------------------------
      // WAIT_WSUM:
      //   Hold until weighted_sum_engine asserts wsum_done.
      //   acc[D] has been updated with prob_slice * v_tile.
      // --------------------------------------------------------
      WAIT_WSUM: begin
        if (wsum_done) begin
          next_state = NEXT_KV_FOR_V;
        end
      end

      // --------------------------------------------------------
      // NEXT_KV_FOR_V:
      //   Index update happens in always_ff.
      //   If this was the last V tile, out_row is complete.
      //   Move to WRITE_OUTPUT.
      //   Otherwise loop back for the next V tile.
      // --------------------------------------------------------
      NEXT_KV_FOR_V: begin
        if (kv_tile_idx_r == KV_TILE_IDX_W'(NUM_KV_TILES - 1)) begin
          // All V tiles accumulated — output row is complete
          next_state = WRITE_OUTPUT;
        end else begin
          next_state = LOAD_KV_FOR_V;
        end
      end

      // --------------------------------------------------------
      // WRITE_OUTPUT:
      //   Assert out_write_en for one cycle.
      //   output_buffer will write out_row into O_mem[global_q_idx].
      // --------------------------------------------------------
      WRITE_OUTPUT: begin
        out_write_en = 1'b1;
        next_state   = WAIT_WRITE;
      end

      // --------------------------------------------------------
      // WAIT_WRITE:
      //   Hold until output_buffer asserts out_write_done.
      //   O_mem[global_q_idx][:] is now committed.
      // --------------------------------------------------------
      WAIT_WRITE: begin
        if (out_write_done) begin
          next_state = NEXT_Q_ROW;
        end
      end

      // --------------------------------------------------------
      // NEXT_Q_ROW:
      //   Index update happens in always_ff.
      //   If all BR rows in this Q tile are complete, move to
      //   NEXT_Q_TILE. Otherwise load the next row in this tile.
      // --------------------------------------------------------
      NEXT_Q_ROW: begin
        if (q_local_row_idx_r == BR_IDX_W'(BR - 1)) begin
          // All rows in current Q tile are done
          next_state = NEXT_Q_TILE;
        end else begin
          // More rows remain in this Q tile
          next_state = INIT_ROW;
        end
      end

      // --------------------------------------------------------
      // NEXT_Q_TILE:
      //   Index update happens in always_ff.
      //   If all Q tiles are processed, the design is complete.
      //   Otherwise reload the next Q tile.
      // --------------------------------------------------------
      NEXT_Q_TILE: begin
        if (q_tile_idx_r == Q_TILE_IDX_W'(NUM_Q_TILES - 1)) begin
          // All Q tiles processed — entire output matrix is written
          next_state = DONE;
        end else begin
          next_state = LOAD_Q_TILE;
        end
      end

      // --------------------------------------------------------
      // DONE:
      //   Assert done continuously until reset or new start.
      //   All N output rows have been written to O_mem.
      // --------------------------------------------------------
      DONE: begin
        done = 1'b1;
        // Hold in DONE until external reset releases the design
      end

      // --------------------------------------------------------
      // Default: should never be reached with proper synthesis
      // --------------------------------------------------------
      default: begin
        next_state = IDLE;
      end

    endcase
  end

endmodule
