// ============================================================
// kv_buffer.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Loads one K tile and one V tile, each of shape [BC][D], from
//   K_mem and V_mem simultaneously. Both tiles are held stable in
//   internal registers after loading.
//
//   This buffer is reused in TWO separate phases per query row:
//
//   PHASE 1 — Score collection (inner KV loop, Pass A):
//     k_tile is consumed by score_engine to compute score_tile.
//     v_tile is loaded but not used in this phase.
//
//   PHASE 2 — Output accumulation (inner KV loop, Pass B):
//     v_tile is consumed by weighted_sum_engine to accumulate out_row.
//     k_tile is loaded but not used in this phase.
//
//   Both tiles are always loaded together to keep the interface
//   simple and symmetric. The controller decides which output is
//   consumed in each phase.
//
//   Tile mapping:
//     k_tile[r][d]  <--  K_mem[kv_tile_base + r][d]
//     v_tile[r][d]  <--  V_mem[kv_tile_base + r][d]
//     for r = 0..BC-1,  d = 0..D-1
//
//   Example (kv_tile_base = 12, BC = 4, D = 16):
//     k_tile[0][:] = K_mem[12][:]    v_tile[0][:] = V_mem[12][:]
//     k_tile[1][:] = K_mem[13][:]    v_tile[1][:] = V_mem[13][:]
//     k_tile[2][:] = K_mem[14][:]    v_tile[2][:] = V_mem[14][:]
//     k_tile[3][:] = K_mem[15][:]    v_tile[3][:] = V_mem[15][:]
//
// SIGNAL OWNERSHIP:
//   load_start, kv_tile_base  <- attention_controller / addr_gen
//   K_mem, V_mem              <- attention_top (preloaded by testbench)
//   k_tile                    -> score_engine      (score phase)
//   v_tile                    -> weighted_sum_engine (output phase)
//   load_done, valid          -> attention_controller
// ============================================================

import attention_pkg::*;

module kv_buffer (
  input  logic                      clk,
  input  logic                      rst_n,

  // ---- Load control (from controller / addr_gen) ----
  input  logic                      load_start,    // pulse to begin tile load
  input  logic [SEQ_IDX_W-1:0]     kv_tile_base,  // first row of tile in K/V mem

  // ---- Source memories (from attention_top / testbench) ----
  input  logic signed [DATA_W-1:0] K_mem [0:N-1][0:D-1],
  input  logic signed [DATA_W-1:0] V_mem [0:N-1][0:D-1],

  // ---- Tile outputs ----
  // k_tile: consumed by score_engine during score-collection phase
  output logic signed [DATA_W-1:0] k_tile [0:BC-1][0:D-1],

  // v_tile: consumed by weighted_sum_engine during output-accumulation phase
  output logic signed [DATA_W-1:0] v_tile [0:BC-1][0:D-1],

  // ---- Handshake outputs (to controller) ----
  output logic                      load_done,     // one-cycle pulse when both tiles ready
  output logic                      valid          // level: tile contents are valid
);

  // ----------------------------------------------------------
  // Internal counters for tile copy.
  //   r_cnt : local tile row being written (0..BC-1)
  //   d_cnt : dimension element being written (0..D-1)
  //
  // Both k_tile and v_tile are written in parallel each cycle.
  // Copy schedule: one element pair per cycle, row-major order.
  //   Total cycles = BC * D = 4 * 16 = 64 cycles.
  // ----------------------------------------------------------
  logic [BC_IDX_W-1:0]  r_cnt;
  logic [DIM_IDX_W-1:0] d_cnt;
  logic                  loading;   // high while copy is in progress

  // ----------------------------------------------------------
  // Sequential logic: tile registers, counters, and status flags
  // ----------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // On reset: clear status flags and counters.
      // Tile storage not explicitly zeroed; valid = 0 is the guard.
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
        // Start of new tile load.
        // Deassert valid immediately so score_engine and
        // weighted_sum_engine do not consume stale data.
        // -------------------------------------------------------
        loading <= 1'b1;
        valid   <= 1'b0;
        r_cnt   <= '0;
        d_cnt   <= '0;

      end else if (loading) begin
        // -------------------------------------------------------
        // Tile copy in progress: write one K and one V element
        // per cycle at the same [r_cnt][d_cnt] position.
        //
        // Global memory addresses:
        //   K_mem[ kv_tile_base + r_cnt ][ d_cnt ]
        //   V_mem[ kv_tile_base + r_cnt ][ d_cnt ]
        //
        // Local tile addresses:
        //   k_tile[ r_cnt ][ d_cnt ]
        //   v_tile[ r_cnt ][ d_cnt ]
        // -------------------------------------------------------
        k_tile[r_cnt][d_cnt] <= K_mem[kv_tile_base + SEQ_IDX_W'(r_cnt)][d_cnt];
        v_tile[r_cnt][d_cnt] <= V_mem[kv_tile_base + SEQ_IDX_W'(r_cnt)][d_cnt];

        if (d_cnt == DIM_IDX_W'(D - 1)) begin
          // Finished all D dimensions for this row
          d_cnt <= '0;

          if (r_cnt == BC_IDX_W'(BC - 1)) begin
            // Finished all BC rows: both tiles fully loaded
            loading   <= 1'b0;
            load_done <= 1'b1;   // one-cycle pulse
            valid     <= 1'b1;   // both k_tile and v_tile now valid
            r_cnt     <= '0;
          end else begin
            // Advance to next tile row
            r_cnt <= r_cnt + BC_IDX_W'(1);
          end

        end else begin
          // Advance to next dimension within this row
          d_cnt <= d_cnt + DIM_IDX_W'(1);
        end
      end
    end
  end

endmodule