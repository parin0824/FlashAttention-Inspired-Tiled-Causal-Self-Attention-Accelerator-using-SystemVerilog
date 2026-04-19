// ============================================================
// q_buffer.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Loads one Q tile of shape [BR][D] from the global Q_mem array
//   and holds it stable in internal registers while the controller
//   iterates over all KV tiles for that Q tile.
//
//   Tile mapping:
//     q_tile[r][d]  <--  Q_mem[q_tile_base + r][d]
//     for r = 0..BR-1,  d = 0..D-1
//
//   Example (q_tile_base = 8, BR = 4, D = 16):
//     q_tile[0][:] = Q_mem[8][:]
//     q_tile[1][:] = Q_mem[9][:]
//     q_tile[2][:] = Q_mem[10][:]
//     q_tile[3][:] = Q_mem[11][:]
//
// SIGNAL OWNERSHIP:
//   load_start, q_tile_base  <- attention_controller / addr_gen
//   Q_mem                    <- attention_top (preloaded by testbench)
//   q_tile                   -> score_engine
//   load_done, valid         -> attention_controller
// ============================================================

import attention_pkg::*;

module q_buffer (
  input  logic                       clk,
  input  logic                       rst_n,

  // ---- Load control (from controller) ----
  input  logic                       load_start,   // pulse to begin tile load
  input  logic [SEQ_IDX_W-1:0]      q_tile_base,  // first row of tile in Q_mem

  // ---- Source memory (from attention_top / testbench) ----
  input  logic signed [DATA_W-1:0]  Q_mem [0:N-1][0:D-1],

  // ---- Tile output (to score_engine) ----
  output logic signed [DATA_W-1:0]  q_tile [0:BR-1][0:D-1],

  // ---- Handshake outputs (to controller) ----
  output logic                       load_done,    // one-cycle pulse when tile ready
  output logic                       valid         // level: tile contents are valid
);

  // ----------------------------------------------------------
  // Internal row and dimension counters used during tile copy.
  //   r_cnt : tracks which local tile row is being written (0..BR-1)
  //   d_cnt : tracks which dimension element is being written (0..D-1)
  //
  // Copy schedule: one element per cycle, row-major order.
  //   Total cycles to fill tile = BR * D = 4 * 16 = 64 cycles.
  // ----------------------------------------------------------
  logic [BR_IDX_W-1:0]  r_cnt;
  logic [DIM_IDX_W-1:0] d_cnt;
  logic                  loading;   // high while copy is in progress

  // ----------------------------------------------------------
  // Sequential logic: tile registers, counters, and status flags
  // ----------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // On reset: clear all status and counters.
      // q_tile storage is not explicitly cleared to save area;
      // valid = 0 guarantees consumers will not use stale data.
      loading   <= 1'b0;
      load_done <= 1'b0;
      valid     <= 1'b0;
      r_cnt     <= '0;
      d_cnt     <= '0;

    end else begin

      // Default: deassert one-cycle pulse
      load_done <= 1'b0;

      if (load_start && !loading) begin
        // -------------------------------------------------------
        // Rising edge of load_start: begin a new tile copy.
        // Clear valid so consumers see the tile as stale while
        // the new data is being written in.
        // -------------------------------------------------------
        loading <= 1'b1;
        valid   <= 1'b0;
        r_cnt   <= '0;
        d_cnt   <= '0;

      end else if (loading) begin
        // -------------------------------------------------------
        // Tile copy in progress: write one element per cycle.
        //
        // Global memory address for this element:
        //   Q_mem[ q_tile_base + r_cnt ][ d_cnt ]
        //
        // Local tile address:
        //   q_tile[ r_cnt ][ d_cnt ]
        // -------------------------------------------------------
        q_tile[r_cnt][d_cnt] <= Q_mem[q_tile_base + SEQ_IDX_W'(r_cnt)][d_cnt];

        if (d_cnt == DIM_IDX_W'(D - 1)) begin
          // Finished all D dimensions for this row
          d_cnt <= '0;

          if (r_cnt == BR_IDX_W'(BR - 1)) begin
            // Finished all BR rows: tile copy complete
            loading   <= 1'b0;
            load_done <= 1'b1;   // one-cycle pulse
            valid     <= 1'b1;   // tile is now valid and stable
            r_cnt     <= '0;
          end else begin
            // Advance to next tile row
            r_cnt <= r_cnt + BR_IDX_W'(1);
          end

        end else begin
          // Advance to next dimension within this row
          d_cnt <= d_cnt + DIM_IDX_W'(1);
        end
      end
    end
  end

endmodule