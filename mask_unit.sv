// =============================================================================
// mask_unit.sv
// Applies causal mask to a BR x BC score tile.
// SEQUENTIAL: 1-cycle pipeline register on output for higher fmax.
// =============================================================================

import attention_pkg::*;

module mask_unit #(
    parameter int BR      = 4,
    parameter int BC      = 4,
    parameter int SCORE_W = 32,
    parameter int IDX_W   = $clog2(32)
) (
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic signed [SCORE_W-1:0]     score_tile_in  [BR][BC],
    input  logic        [IDX_W-1:0]       q_tile_base,
    input  logic        [IDX_W-1:0]       kv_tile_base,
    output logic signed [SCORE_W-1:0]     score_tile_out [BR][BC]
);

    localparam int ADDR_W = IDX_W + 1;

    logic [ADDR_W-1:0]         gq        [BR][BC];
    logic [ADDR_W-1:0]         gk        [BR][BC];
    logic signed [SCORE_W-1:0] mask_next [BR][BC];

    always_comb begin
        for (int r = 0; r < BR; r++) begin
            for (int c = 0; c < BC; c++) begin
                gq[r][c] = {1'b0, q_tile_base}  + ADDR_W'(r);
                gk[r][c] = {1'b0, kv_tile_base} + ADDR_W'(c);
                mask_next[r][c] = (gk[r][c] > gq[r][c])
                                  ? NEG_INF
                                  : score_tile_in[r][c];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BC; c++)
                    score_tile_out[r][c] <= '0;
        end else begin
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BC; c++)
                    score_tile_out[r][c] <= mask_next[r][c];
        end
    end

endmodule
