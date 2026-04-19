// =============================================================================
// row_score_store.sv
// Collects one full row of N scores by writing BC-wide fragments from each
// KV tile.  Stores only one query row at a time.
// Protocol:
//   1. start_row  -> clear row_scores, row_valid, tile counter
//   2. store_en   -> write BC entries at kv_tile_idx*BC; increment counter
//   3. After NUM_KV_TILES stores: row_valid=1, store_done pulses 1 cycle
// =============================================================================

module row_score_store #(
    parameter int N       = 32,
    parameter int BR      = 4,
    parameter int BC      = 4,
    parameter int SCORE_W = 32,
    parameter int IDX_W   = $clog2(32)
) (
    input  logic                            clk,
    input  logic                            rst_n,
    input  logic                            start_row,
    input  logic                            store_en,
    input  logic [$clog2(BR)-1:0]           q_local_row_idx,
    input  logic [$clog2(N/BC)-1:0]         kv_tile_idx,
    input  logic signed [SCORE_W-1:0]       score_tile_in [BR][BC],
    output logic signed [SCORE_W-1:0]       row_scores [N],
    output logic                            row_valid,
    output logic                            store_done
);

    localparam int NUM_KV_TILES = N / BC;
    localparam int ADDR_W       = $clog2(N);
    localparam int CTR_W        = $clog2(NUM_KV_TILES + 1);

    logic [CTR_W-1:0]  tile_ctr;
    logic [ADDR_W-1:0] wr_base;

    always_comb begin
        wr_base = ADDR_W'(kv_tile_idx) * ADDR_W'(BC);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_ctr   <= '0;
            row_valid  <= 1'b0;
            store_done <= 1'b0;
            for (int i = 0; i < N; i++)
                row_scores[i] <= '0;
        end else begin
            store_done <= 1'b0;

            if (start_row) begin
                tile_ctr  <= '0;
                row_valid <= 1'b0;
                for (int i = 0; i < N; i++)
                    row_scores[i] <= '0;
            end else if (store_en) begin
                for (int c = 0; c < BC; c++) begin
                    row_scores[wr_base + ADDR_W'(c)] <=
                        score_tile_in[q_local_row_idx][c];
                end
                tile_ctr   <= tile_ctr + CTR_W'(1);
                store_done <= 1'b1;  // pulse after every store_en (per-store ACK)
                if (tile_ctr == CTR_W'(NUM_KV_TILES - 1)) begin
                    row_valid  <= 1'b1;  // only set when full row is complete
                end
            end
        end
    end

endmodule
