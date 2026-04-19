// ============================================================
// output_buffer.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Stores completed output rows into the global output matrix
//   O_mem. Each write deposits one full row out_row[D] at the
//   absolute sequence position global_q_idx.
//
// WHY global_q_idx AND NOT q_local_row_idx:
//   q_local_row_idx is a tile-local index (0..BR-1) that resets
//   to zero for every new Q tile. Writing with q_local_row_idx
//   would overwrite the same BR rows of O_mem on every Q tile
//   iteration, destroying all previously written output.
//
//   global_q_idx = q_tile_base + q_local_row_idx, which is the
//   absolute row position in [0..N-1]. This is the correct write
//   address that places each output row exactly once into O_mem.
//
//   Example: Q tile 2 (q_tile_base=8), q_local_row_idx=1
//     global_q_idx = 9  -->  O_mem[9][:] is written
//     q_local_row_idx   -->  O_mem[1][:] would be written (WRONG)
//
// SIGNAL OWNERSHIP:
//   write_en, global_q_idx  <- attention_controller / addr_gen
//   out_row                 <- weighted_sum_engine
//   O_mem                   -> attention_top output port, testbench
//   write_done              -> attention_controller
// ============================================================

import attention_pkg::*;

module output_buffer (
  input  logic                       clk,
  input  logic                       rst_n,

  // ---- Write control (from controller / addr_gen) ----
  input  logic                       write_en,       // pulse to begin row write
  input  logic [SEQ_IDX_W-1:0]      global_q_idx,   // absolute O_mem row to write

  // ---- Data input (from weighted_sum_engine) ----
  input  logic signed [OUT_W-1:0]   out_row [0:D-1],

  // ---- Output memory (to attention_top / testbench) ----
  output logic signed [OUT_W-1:0]   O_mem [0:N-1][0:D-1],

  // ---- Handshake output (to controller) ----
  output logic                       write_done      // one-cycle pulse when write finishes
);

  // ----------------------------------------------------------
  // Dimension counter used during row write.
  //   d_cnt : tracks which dimension element is being written
  //
  // Write schedule: one element per cycle, d = 0..D-1.
  //   Total cycles to write one row = D = 16 cycles.
  // ----------------------------------------------------------
  logic [DIM_IDX_W-1:0] d_cnt;
  logic                  writing;        // high while write is in progress
  logic [SEQ_IDX_W-1:0] write_row_idx;  // latched global_q_idx at write_en

  // ----------------------------------------------------------
  // Sequential logic: O_mem write, counter, and status flags
  // ----------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // On reset: clear control state.
      // O_mem is not explicitly zeroed; testbench reads O_mem
      // only after write_done confirms a valid row was written.
      writing       <= 1'b0;
      write_done    <= 1'b0;
      d_cnt         <= '0;
      write_row_idx <= '0;

    end else begin

      // Default: deassert one-cycle pulse
      write_done <= 1'b0;

      if (write_en && !writing) begin
        // -------------------------------------------------------
        // Latch the target row index at the start of the write.
        // global_q_idx must be stable for the full D-cycle write
        // window; latching it here protects against controller
        // changing its index mid-write.
        // -------------------------------------------------------
        writing       <= 1'b1;
        write_row_idx <= global_q_idx;
        d_cnt         <= '0;

      end else if (writing) begin
        // -------------------------------------------------------
        // Write one output element per cycle.
        //
        // Target address:
        //   O_mem[ write_row_idx ][ d_cnt ]
        //
        // Source:
        //   out_row[ d_cnt ]
        //
        // write_row_idx is the global sequence index of this
        // query row, NOT the tile-local q_local_row_idx.
        // -------------------------------------------------------
        O_mem[write_row_idx][d_cnt] <= out_row[d_cnt];

        if (d_cnt == DIM_IDX_W'(D - 1)) begin
          // All D dimensions written: row is complete
          writing    <= 1'b0;
          write_done <= 1'b1;   // one-cycle pulse to controller
          d_cnt      <= '0;
        end else begin
          d_cnt <= d_cnt + DIM_IDX_W'(1);
        end
      end
    end
  end

endmodule
