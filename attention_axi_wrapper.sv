import attention_pkg::*;

module attention_axi_wrapper #(
  parameter integer C_S_AXI_DATA_WIDTH = 32,
  parameter integer C_S_AXI_ADDR_WIDTH = 16
) (
  (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 16, FREQ_HZ 100000000, HAS_BURST 0, HAS_LOCK 0, HAS_PROT 1, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, SUPPORTS_NARROW_BURST 0, MAX_BURST_LENGTH 1" *)
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *) input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *) input  wire [2:0]                  S_AXI_AWPROT,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input  wire                        S_AXI_AWVALID,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output wire                        S_AXI_AWREADY,

  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)  input  wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)  input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input  wire                        S_AXI_WVALID,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output wire                        S_AXI_WREADY,

  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)  output reg  [1:0]                  S_AXI_BRESP,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output reg                         S_AXI_BVALID,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input  wire                        S_AXI_BREADY,

  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *) input  wire [2:0]                  S_AXI_ARPROT,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input  wire                        S_AXI_ARVALID,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output reg                         S_AXI_ARREADY,

  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)  output reg  [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)  output reg  [1:0]                  S_AXI_RRESP,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output reg                         S_AXI_RVALID,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input  wire                        S_AXI_RREADY,

  (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI_ACLK, ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET S_AXI_ARESETN" *)
  input  wire                        S_AXI_ACLK,
  (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI_ARESETN, POLARITY ACTIVE_LOW" *)
  input  wire                        S_AXI_ARESETN,

  output wire                        irq
);

  localparam integer CTRL_ADDR      = 16'h0000;
  localparam integer STATUS_ADDR    = 16'h0004;
  localparam integer CYCLE_ADDR     = 16'h0010;
  localparam integer SCORE_ADDR     = 16'h0014;
  localparam integer SOFTMAX_ADDR   = 16'h0018;
  localparam integer WSUM_ADDR      = 16'h001C;
  localparam integer LOAD_ADDR      = 16'h0020;
  localparam integer STALL_ADDR     = 16'h0024;
  localparam integer Q_BASE_ADDR    = 16'h0100;
  localparam integer K_BASE_ADDR    = 16'h0900;
  localparam integer V_BASE_ADDR    = 16'h1100;
  localparam integer O_BASE_ADDR    = 16'h1900;
  localparam integer MEM_WORDS      = N*D; // 512
  localparam integer MEM_BYTES      = MEM_WORDS*4; // 2048 bytes

  reg start_pulse;
  reg busy_reg;
  reg done_sticky;
  reg irq_enable;
  reg irq_status;

  logic signed [DATA_W-1:0] q_mem [0:N-1][0:D-1];
  logic signed [DATA_W-1:0] k_mem [0:N-1][0:D-1];
  logic signed [DATA_W-1:0] v_mem [0:N-1][0:D-1];
  logic signed [OUT_W-1:0]  o_mem [0:N-1][0:D-1];

  logic core_done;
  logic [31:0] cycle_count;
  logic [31:0] score_cycles;
  logic [31:0] softmax_cycles;
  logic [31:0] wsum_cycles;
  logic [31:0] load_events;
  logic [31:0] stall_cycles;

  attention_top u_core (
    .clk           (S_AXI_ACLK),
    .rst_n         (S_AXI_ARESETN),
    .start         (start_pulse),
    .Q_mem         (q_mem),
    .K_mem         (k_mem),
    .V_mem         (v_mem),
    .done          (core_done),
    .O_mem         (o_mem),
    .cycle_count   (cycle_count),
    .score_cycles  (score_cycles),
    .softmax_cycles(softmax_cycles),
    .wsum_cycles   (wsum_cycles),
    .load_events   (load_events),
    .stall_cycles  (stall_cycles)
  );

  assign S_AXI_AWREADY = 1'b1;
  assign S_AXI_WREADY  = 1'b1;
  assign irq = irq_enable & irq_status;

  wire write_fire = S_AXI_AWVALID & S_AXI_WVALID & ~S_AXI_BVALID;
  wire read_fire  = S_AXI_ARVALID & ~S_AXI_RVALID;

  integer i, j;
  integer word_idx;
  integer row_idx;
  integer col_idx;
  integer mem_off;

  always_ff @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
    if (!S_AXI_ARESETN) begin
      S_AXI_BVALID <= 1'b0;
      S_AXI_BRESP  <= 2'b00;
      S_AXI_ARREADY <= 1'b0;
      S_AXI_RVALID <= 1'b0;
      S_AXI_RRESP  <= 2'b00;
      S_AXI_RDATA  <= '0;
      start_pulse  <= 1'b0;
      busy_reg     <= 1'b0;
      done_sticky  <= 1'b0;
      irq_enable   <= 1'b0;
      irq_status   <= 1'b0;
      for (i = 0; i < N; i = i + 1) begin
        for (j = 0; j < D; j = j + 1) begin
          q_mem[i][j] <= '0;
          k_mem[i][j] <= '0;
          v_mem[i][j] <= '0;
        end
      end
    end else begin
      start_pulse <= 1'b0;
      S_AXI_ARREADY <= 1'b0;

      if (core_done) begin
        busy_reg    <= 1'b0;
        done_sticky <= 1'b1;
        irq_status  <= 1'b1;
      end

      if (S_AXI_BVALID && S_AXI_BREADY)
        S_AXI_BVALID <= 1'b0;

      if (write_fire) begin
        S_AXI_BVALID <= 1'b1;
        S_AXI_BRESP  <= 2'b00;
        if (S_AXI_AWADDR == CTRL_ADDR) begin
          if (S_AXI_WSTRB[0]) begin
            if (S_AXI_WDATA[0] && !busy_reg) begin
              start_pulse <= 1'b1;
              busy_reg    <= 1'b1;
              done_sticky <= 1'b0;
              irq_status  <= 1'b0;
            end
            if (S_AXI_WDATA[1]) begin
              done_sticky <= 1'b0;
              irq_status  <= 1'b0;
            end
            irq_enable <= S_AXI_WDATA[8];
          end
        end else if (!busy_reg && (S_AXI_AWADDR >= Q_BASE_ADDR) && (S_AXI_AWADDR < (Q_BASE_ADDR + MEM_BYTES))) begin
          mem_off = (S_AXI_AWADDR - Q_BASE_ADDR) >> 2;
          row_idx = mem_off / D;
          col_idx = mem_off % D;
          if (row_idx < N && col_idx < D)
            q_mem[row_idx][col_idx] <= S_AXI_WDATA[DATA_W-1:0];
        end else if (!busy_reg && (S_AXI_AWADDR >= K_BASE_ADDR) && (S_AXI_AWADDR < (K_BASE_ADDR + MEM_BYTES))) begin
          mem_off = (S_AXI_AWADDR - K_BASE_ADDR) >> 2;
          row_idx = mem_off / D;
          col_idx = mem_off % D;
          if (row_idx < N && col_idx < D)
            k_mem[row_idx][col_idx] <= S_AXI_WDATA[DATA_W-1:0];
        end else if (!busy_reg && (S_AXI_AWADDR >= V_BASE_ADDR) && (S_AXI_AWADDR < (V_BASE_ADDR + MEM_BYTES))) begin
          mem_off = (S_AXI_AWADDR - V_BASE_ADDR) >> 2;
          row_idx = mem_off / D;
          col_idx = mem_off % D;
          if (row_idx < N && col_idx < D)
            v_mem[row_idx][col_idx] <= S_AXI_WDATA[DATA_W-1:0];
        end
      end

      if (read_fire) begin
        S_AXI_ARREADY <= 1'b1;
        S_AXI_RVALID <= 1'b1;
        S_AXI_RRESP  <= 2'b00;
        S_AXI_RDATA  <= '0;

        case (S_AXI_ARADDR)
          CTRL_ADDR: begin
            S_AXI_RDATA[0] <= busy_reg;
            S_AXI_RDATA[1] <= done_sticky;
            S_AXI_RDATA[8] <= irq_enable;
            S_AXI_RDATA[9] <= irq_status;
          end
          STATUS_ADDR: begin
            S_AXI_RDATA[0] <= busy_reg;
            S_AXI_RDATA[1] <= done_sticky;
            S_AXI_RDATA[2] <= core_done;
          end
          CYCLE_ADDR:   S_AXI_RDATA <= cycle_count;
          SCORE_ADDR:   S_AXI_RDATA <= score_cycles;
          SOFTMAX_ADDR: S_AXI_RDATA <= softmax_cycles;
          WSUM_ADDR:    S_AXI_RDATA <= wsum_cycles;
          LOAD_ADDR:    S_AXI_RDATA <= load_events;
          STALL_ADDR:   S_AXI_RDATA <= stall_cycles;
          default: begin
            if ((S_AXI_ARADDR >= Q_BASE_ADDR) && (S_AXI_ARADDR < (Q_BASE_ADDR + MEM_BYTES))) begin
              mem_off = (S_AXI_ARADDR - Q_BASE_ADDR) >> 2;
              row_idx = mem_off / D;
              col_idx = mem_off % D;
              if (row_idx < N && col_idx < D)
                S_AXI_RDATA <= {{(32-DATA_W){q_mem[row_idx][col_idx][DATA_W-1]}}, q_mem[row_idx][col_idx]};
            end else if ((S_AXI_ARADDR >= K_BASE_ADDR) && (S_AXI_ARADDR < (K_BASE_ADDR + MEM_BYTES))) begin
              mem_off = (S_AXI_ARADDR - K_BASE_ADDR) >> 2;
              row_idx = mem_off / D;
              col_idx = mem_off % D;
              if (row_idx < N && col_idx < D)
                S_AXI_RDATA <= {{(32-DATA_W){k_mem[row_idx][col_idx][DATA_W-1]}}, k_mem[row_idx][col_idx]};
            end else if ((S_AXI_ARADDR >= V_BASE_ADDR) && (S_AXI_ARADDR < (V_BASE_ADDR + MEM_BYTES))) begin
              mem_off = (S_AXI_ARADDR - V_BASE_ADDR) >> 2;
              row_idx = mem_off / D;
              col_idx = mem_off % D;
              if (row_idx < N && col_idx < D)
                S_AXI_RDATA <= {{(32-DATA_W){v_mem[row_idx][col_idx][DATA_W-1]}}, v_mem[row_idx][col_idx]};
            end else if ((S_AXI_ARADDR >= O_BASE_ADDR) && (S_AXI_ARADDR < (O_BASE_ADDR + MEM_BYTES))) begin
              mem_off = (S_AXI_ARADDR - O_BASE_ADDR) >> 2;
              row_idx = mem_off / D;
              col_idx = mem_off % D;
              if (row_idx < N && col_idx < D)
                S_AXI_RDATA <= o_mem[row_idx][col_idx];
            end
          end
        endcase
      end

      if (S_AXI_RVALID && S_AXI_RREADY)
        S_AXI_RVALID <= 1'b0;
    end
  end
endmodule
