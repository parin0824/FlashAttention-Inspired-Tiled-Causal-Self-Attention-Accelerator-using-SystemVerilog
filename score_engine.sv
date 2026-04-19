// =============================================================================
// score_engine.sv
// Computes one BR x BC tile of scaled attention scores.
// Math: score_tile[r][c] = (SUM_d q_tile[r][d]*k_tile[c][d]) >>> SCALE_SHIFT
//
// OPT-1 (preserved): Eliminated separate SCALE state and DONE state.
//
// OPT-MAC-PIPE: Registered MAC output to break multiplier critical path.
//   Stage 1: sign-extend + multiply  ->  mac_product_reg  (registered)
//   Stage 2: acc + mac_product_reg   ->  acc              (registered)
//
//   Delayed counter copies (r_cnt_p, c_cnt_p, d_cnt_p) track which entry
//   mac_product_reg belongs to. comp_first skips the first accumulation
//   (pipeline fill). S_DRAIN flushes the last registered product.
//
// FSM    : IDLE -> COMPUTE -> DRAIN -> IDLE
// Latency: BR*BC*D + 2 cycles to done pulse (+1 vs original).
//          Default params (BR=BC=4, D=16): 258 cycles.
// =============================================================================

module score_engine #(
    parameter int BR      = 4,
    parameter int BC      = 4,
    parameter int D       = 16,
    parameter int DATA_W  = 8,
    parameter int SCORE_W = 32
) (
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,
    input  logic signed [DATA_W-1:0]      q_tile [BR][D],
    input  logic signed [DATA_W-1:0]      k_tile [BC][D],
    output logic signed [SCORE_W-1:0]     score_tile [BR][BC],
    output logic                          busy,
    output logic                          done
);

    localparam int SCALE_SHIFT = 2;
    localparam int R_W = $clog2(BR);
    localparam int C_W = $clog2(BC);
    localparam int D_W = $clog2(D);

    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        COMPUTE = 2'b01,
        DRAIN   = 2'b10
    } state_t;

    state_t state;

    // Current loop counters
    logic [R_W-1:0]            r_cnt;
    logic [C_W-1:0]            c_cnt;
    logic [D_W-1:0]            d_cnt;

    // One-cycle delayed counters: mac_product_reg corresponds to these
    logic [R_W-1:0]            r_cnt_p;
    logic [C_W-1:0]            c_cnt_p;
    logic [D_W-1:0]            d_cnt_p;

    logic signed [SCORE_W-1:0] acc;
    logic signed [SCORE_W-1:0] mac_product_reg;
    logic                      comp_first;

    // Combinational Stage 1: sign-extend and multiply
    logic signed [SCORE_W-1:0] q_ext;
    logic signed [SCORE_W-1:0] k_ext;
    logic signed [SCORE_W-1:0] mac_next;

    always_comb begin
        q_ext    = {{(SCORE_W-DATA_W){q_tile[r_cnt][d_cnt][DATA_W-1]}},
                     q_tile[r_cnt][d_cnt]};
        k_ext    = {{(SCORE_W-DATA_W){k_tile[c_cnt][d_cnt][DATA_W-1]}},
                     k_tile[c_cnt][d_cnt]};
        mac_next = q_ext * k_ext;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= IDLE;
            r_cnt           <= '0;
            c_cnt           <= '0;
            d_cnt           <= '0;
            r_cnt_p         <= '0;
            c_cnt_p         <= '0;
            d_cnt_p         <= '0;
            acc             <= '0;
            mac_product_reg <= '0;
            comp_first      <= 1'b0;
            busy            <= 1'b0;
            done            <= 1'b0;
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BC; c++)
                    score_tile[r][c] <= '0;
        end else begin
            done            <= 1'b0;          // default: pulse cleared each cycle
            mac_product_reg <= mac_next;       // always register the MAC result

            case (state)

                // ---------------------------------------------------------
                // IDLE: wait for start; hold counters at 0 so mac_product_reg
                // settles to q[0][0]*k[0][0] before COMPUTE begins.
                // comp_first ensures that settled value is not mis-used.
                // ---------------------------------------------------------
                IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        state      <= COMPUTE;
                        busy       <= 1'b1;
                        r_cnt      <= '0;
                        c_cnt      <= '0;
                        d_cnt      <= '0;
                        r_cnt_p    <= '0;
                        c_cnt_p    <= '0;
                        d_cnt_p    <= '0;
                        acc        <= '0;
                        comp_first <= 1'b1;
                    end
                end

                // ---------------------------------------------------------
                // COMPUTE: one entry per cycle through the BR*BC*D loop.
                //
                // mac_product_reg = product for (r_cnt_p, c_cnt_p, d_cnt_p).
                //
                // comp_first=1 (first cycle): pipeline filling, skip Stage 2.
                // comp_first=0: apply Stage 2 using mac_product_reg:
                //   d_cnt_p != D-1 : acc  += mac_product_reg
                //   d_cnt_p == D-1 : finalize score_tile[r_p][c_p], reset acc
                //
                // When current (r,c,d) reaches (BR-1,BC-1,D-1): go to DRAIN.
                // ---------------------------------------------------------
                COMPUTE: begin
                    if (!comp_first) begin
                        if (d_cnt_p == D_W'(D - 1)) begin
                            score_tile[r_cnt_p][c_cnt_p] <=
                                (acc + mac_product_reg) >>> SCALE_SHIFT;
                            acc <= '0;
                        end else begin
                            acc <= acc + mac_product_reg;
                        end
                    end

                    comp_first <= 1'b0;
                    r_cnt_p    <= r_cnt;
                    c_cnt_p    <= c_cnt;
                    d_cnt_p    <= d_cnt;

                    if (r_cnt == R_W'(BR - 1) &&
                        c_cnt == C_W'(BC - 1) &&
                        d_cnt == D_W'(D  - 1)) begin
                        state <= DRAIN;
                    end else if (d_cnt == D_W'(D - 1)) begin
                        d_cnt <= '0;
                        if (c_cnt == C_W'(BC - 1)) begin
                            c_cnt <= '0;
                            r_cnt <= r_cnt + R_W'(1);
                        end else begin
                            c_cnt <= c_cnt + C_W'(1);
                        end
                    end else begin
                        d_cnt <= d_cnt + D_W'(1);
                    end
                end

                // ---------------------------------------------------------
                // DRAIN: flush final mac_product_reg for (BR-1, BC-1, D-1).
                // d_cnt_p = D-1, so finalize score_tile[r_p][c_p].
                // Assert done (one-cycle pulse fires next cycle).
                // ---------------------------------------------------------
                DRAIN: begin
                    score_tile[r_cnt_p][c_cnt_p] <=
                        (acc + mac_product_reg) >>> SCALE_SHIFT;
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end

endmodule