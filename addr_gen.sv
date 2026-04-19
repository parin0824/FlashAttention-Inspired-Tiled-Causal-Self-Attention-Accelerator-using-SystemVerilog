// ============================================================
// addr_gen.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Purely combinational address/index generator.
//   Converts tile-level and local-row indices supplied by the
//   controller into tile base addresses and global sequence
//   indices consumed by buffers, masking logic, output writeback,
//   and probability-slice selection.
//
// INPUTS (from attention_controller):
//   q_tile_idx       : current Q tile being processed (0..NUM_Q_TILES-1)
//   kv_tile_idx      : current KV tile being processed (0..NUM_KV_TILES-1)
//   q_local_row_idx  : active query row within current Q tile (0..BR-1)
//   local_kv_row_idx : active KV row within current KV tile (0..BC-1)
//
// OUTPUTS:
//   q_tile_base  -> q_buffer (load address), mask_unit (causal compare)
//   kv_tile_base -> kv_buffer (load address), mask_unit (causal compare),
//                   attention_top (prob_slice index = kv_tile_base + c)
//   global_q_idx -> output_buffer (write address for O_mem row)
//   global_k_idx -> mask_unit / debug support
// ============================================================

import attention_pkg::*;

module addr_gen (
  // ---- Tile and row index inputs (driven by controller) ----
  input  logic [Q_TILE_IDX_W-1:0]  q_tile_idx,        // 0..NUM_Q_TILES-1
  input  logic [KV_TILE_IDX_W-1:0] kv_tile_idx,       // 0..NUM_KV_TILES-1
  input  logic [BR_IDX_W-1:0]      q_local_row_idx,   // 0..BR-1
  input  logic [BC_IDX_W-1:0]      local_kv_row_idx,  // 0..BC-1

  // ---- Tile base outputs ----
  // q_tile_base  : first global row index of the current Q tile
  //                consumed by q_buffer and mask_unit
  output logic [SEQ_IDX_W-1:0]     q_tile_base,

  // kv_tile_base : first global row index of the current KV tile
  //                consumed by kv_buffer, mask_unit, and
  //                prob_slice selection in attention_top
  output logic [SEQ_IDX_W-1:0]     kv_tile_base,

  // ---- Global row index outputs ----
  // global_q_idx : absolute sequence position of the active query row
  //                consumed by output_buffer to write O_mem[global_q_idx]
  output logic [SEQ_IDX_W-1:0]     global_q_idx,

  // global_k_idx : absolute sequence position of a specific KV row
  //                consumed by mask_unit and available for debug
  output logic [SEQ_IDX_W-1:0]     global_k_idx
);

  // ----------------------------------------------------------
  // All outputs are purely combinational — no registers needed.
  //
  // Formula summary:
  //   q_tile_base  = q_tile_idx  * BR
  //   kv_tile_base = kv_tile_idx * BC
  //   global_q_idx = q_tile_base + q_local_row_idx
  //   global_k_idx = kv_tile_base + local_kv_row_idx
  //
  // Widths:
  //   q_tile_idx  is Q_TILE_IDX_W  bits  (3 bits, max value 7)
  //   BR          is a localparam         (value 4)
  //   Product max = 7 * 4 = 28 < 32 = 2^SEQ_IDX_W  — no overflow
  //
  //   kv_tile_idx is KV_TILE_IDX_W bits  (3 bits, max value 7)
  //   BC          is a localparam         (value 4)
  //   Product max = 7 * 4 = 28 < 32 = 2^SEQ_IDX_W  — no overflow
  // ----------------------------------------------------------

  always_comb begin
    // Tile base: first global row of the current Q tile
    // q_tile_base in [0, 4, 8, 12, 16, 20, 24, 28]
    q_tile_base  = SEQ_IDX_W'(q_tile_idx)  * SEQ_IDX_W'(BR);

    // Tile base: first global row of the current KV tile
    // kv_tile_base in [0, 4, 8, 12, 16, 20, 24, 28]
    kv_tile_base = SEQ_IDX_W'(kv_tile_idx) * SEQ_IDX_W'(BC);

    // Global index of the active query row within the current Q tile
    // Range: 0..N-1 (0..31)
    global_q_idx = q_tile_base  + SEQ_IDX_W'(q_local_row_idx);

    // Global index of a specific row within the current KV tile
    // Range: 0..N-1 (0..31)
    global_k_idx = kv_tile_base + SEQ_IDX_W'(local_kv_row_idx);
  end

endmodule
