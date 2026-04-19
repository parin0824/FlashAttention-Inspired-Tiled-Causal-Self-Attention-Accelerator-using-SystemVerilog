// ============================================================
// attention_pkg.sv
// IO-Aware Tiled Causal Self-Attention Accelerator
//
// PURPOSE:
//   Single source of truth for all shared constants, data widths,
//   index sizes, derived parameters, and the controller FSM state
//   encoding. Every module in this project must import this package.
//
// USAGE:
//   import attention_pkg::*;
// ============================================================

package attention_pkg;

  // ----------------------------------------------------------
  // 1. PRIMARY DESIGN PARAMETERS
  //    Core dimensions of the attention engine.
  //    N  : total sequence length
  //    D  : head dimension (embedding width per head)
  //    BR : query tile row count (rows of Q processed per tile)
  //    BC : key/value tile row count (rows of K/V per tile)
  // ----------------------------------------------------------
  localparam int unsigned N    = 32;
  localparam int unsigned D    = 16;
  localparam int unsigned BR   = 4;
  localparam int unsigned BC   = 4;

  // ----------------------------------------------------------
  // 2. DATA WIDTH PARAMETERS
  //    DATA_W  : signed width of Q, K, V elements (INT8)
  //    SCORE_W : signed accumulator width for dot-product scores
  //    EXP_W   : unsigned fixed-point width for exp LUT output
  //    SUM_W   : unsigned accumulator width for softmax denominator
  //    OUT_W   : signed accumulator width for output row elements
  // ----------------------------------------------------------
  localparam int unsigned DATA_W  = 8;
  localparam int unsigned SCORE_W = 32;
  localparam int unsigned EXP_W   = 16;
  localparam int unsigned SUM_W   = 32;
  localparam int unsigned OUT_W   = 32;

  // ----------------------------------------------------------
  // 3. DERIVED TILE COUNT PARAMETERS
  //    NUM_Q_TILES  : number of Q tile iterations  = N / BR
  //    NUM_KV_TILES : number of KV tile iterations = N / BC
  // ----------------------------------------------------------
  localparam int unsigned NUM_Q_TILES  = N / BR;   // 8
  localparam int unsigned NUM_KV_TILES = N / BC;   // 8

  // ----------------------------------------------------------
  // 4. INDEX BIT WIDTHS
  //    Minimum bit widths needed to represent tile and element
  //    indices cleanly in hardware registers and ports.
  //
  //    Q_TILE_IDX_W  : bits to index q_tile_idx  (0..NUM_Q_TILES-1)
  //    KV_TILE_IDX_W : bits to index kv_tile_idx (0..NUM_KV_TILES-1)
  //    SEQ_IDX_W     : bits to index any row in [0..N-1]
  //    BR_IDX_W      : bits to index a local query row [0..BR-1]
  //    BC_IDX_W      : bits to index a local KV row    [0..BC-1]
  //    DIM_IDX_W     : bits to index a dimension element [0..D-1]
  // ----------------------------------------------------------
  localparam int unsigned Q_TILE_IDX_W  = $clog2(NUM_Q_TILES);   // 3
  localparam int unsigned KV_TILE_IDX_W = $clog2(NUM_KV_TILES);  // 3
  localparam int unsigned SEQ_IDX_W     = $clog2(N);              // 5
  localparam int unsigned BR_IDX_W      = $clog2(BR);             // 2
  localparam int unsigned BC_IDX_W      = $clog2(BC);             // 2
  localparam int unsigned DIM_IDX_W     = $clog2(D);              // 4

  // ----------------------------------------------------------
  // 5. CAUSAL MASK CONSTANT
  //    NEG_INF is written into masked-out score positions so that
  //    exp(NEG_INF) rounds to zero inside the exp LUT, effectively
  //    zeroing out future-token attention weights.
  //
  //    Value: -(1 << (SCORE_W-2))
  //    For SCORE_W=32 this is -1073741824, safely representable
  //    as a signed 32-bit integer and well clear of overflow.
  // ----------------------------------------------------------
  localparam logic signed [SCORE_W-1:0] NEG_INF =
      -(1 <<< (SCORE_W - 2));

  // ----------------------------------------------------------
  // 6. CONTROLLER FSM STATE ENCODING
  //    One-hot or binary encoded (tool chooses); define as enum
  //    so state names are readable in simulation and synthesis.
  //
  //    State progression follows the two-pass-per-query-row loop:
  //
  //    IDLE
  //    └─> LOAD_Q_TILE / WAIT_Q_TILE
  //        └─> INIT_ROW                          (per query row)
  //            ├─> LOAD_KV_FOR_SCORE / WAIT_KV_FOR_SCORE
  //            │   └─> START_SCORE / WAIT_SCORE
  //            │       └─> STORE_ROW_FRAGMENT
  //            │           └─> NEXT_KV_FOR_SCORE (loop 8 KV tiles)
  //            ├─> START_ROW_MAX  / WAIT_ROW_MAX
  //            ├─> START_ROW_SUM  / WAIT_ROW_SUM
  //            ├─> START_NORMALIZE / WAIT_NORMALIZE
  //            ├─> INIT_WSUM
  //            │   └─> LOAD_KV_FOR_V / WAIT_KV_FOR_V
  //            │       └─> START_WSUM / WAIT_WSUM
  //            │           └─> NEXT_KV_FOR_V     (loop 8 KV tiles)
  //            └─> WRITE_OUTPUT / WAIT_WRITE
  //                └─> NEXT_Q_ROW  (loop BR rows)
  //                    └─> NEXT_Q_TILE (loop NUM_Q_TILES tiles)
  //    DONE
  // ----------------------------------------------------------
  typedef enum logic [4:0] {
    IDLE                = 5'd0,
    LOAD_Q_TILE         = 5'd1,
    WAIT_Q_TILE         = 5'd2,
    INIT_ROW            = 5'd3,
    LOAD_KV_FOR_SCORE   = 5'd4,
    WAIT_KV_FOR_SCORE   = 5'd5,
    START_SCORE         = 5'd6,
    WAIT_SCORE          = 5'd7,
    STORE_ROW_FRAGMENT  = 5'd8,
    NEXT_KV_FOR_SCORE   = 5'd9,
    START_ROW_MAX       = 5'd10,
    WAIT_ROW_MAX        = 5'd11,
    START_ROW_SUM       = 5'd12,
    WAIT_ROW_SUM        = 5'd13,
    START_NORMALIZE     = 5'd14,
    WAIT_NORMALIZE      = 5'd15,
    INIT_WSUM           = 5'd16,
    LOAD_KV_FOR_V       = 5'd17,
    WAIT_KV_FOR_V       = 5'd18,
    START_WSUM          = 5'd19,
    WAIT_WSUM           = 5'd20,
    NEXT_KV_FOR_V       = 5'd21,
    WRITE_OUTPUT        = 5'd22,
    WAIT_WRITE          = 5'd23,
    NEXT_Q_ROW          = 5'd24,
    NEXT_Q_TILE         = 5'd25,
    DONE                = 5'd26
  } ctrl_state_t;

endpackage
