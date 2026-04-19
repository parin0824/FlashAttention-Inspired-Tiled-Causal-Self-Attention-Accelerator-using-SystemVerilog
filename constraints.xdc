# ============================================================
# constraints.xdc
# IO-Aware Tiled Causal Self-Attention Accelerator
# Target: Xilinx xc7z010 (Zynq-7010, e.g. Pynq-Z1/Z2)
#
# SYN-FIX-2: Added clock and I/O constraints so Vivado can
# perform proper timing analysis. Without this file, WNS = NA.
# ============================================================

# ------------------------------------------------------------
# 1. Primary clock definition
#    100 MHz = 10 ns period on the clk port.
#    Adjust -period to match your board's clock input.
#    Pynq-Z1/Z2: 125 MHz = 8 ns.  ZedBoard: 100 MHz = 10 ns.
# ------------------------------------------------------------
create_clock -period 10.000 -name clk [get_ports clk]

# ------------------------------------------------------------
# 2. Input delays for control ports (relative to clk)
#    2 ns max setup margin assumed for board-level routing.
# ------------------------------------------------------------
set_input_delay -clock clk -max 2.000 [get_ports rst_n]
set_input_delay -clock clk -max 2.000 [get_ports start]

# ------------------------------------------------------------
# 3. Output delay for the done flag
# ------------------------------------------------------------
set_output_delay -clock clk -max 2.000 [get_ports done]

# ------------------------------------------------------------
# 4. Performance counter outputs (small 32-bit busses)
#    Only constrained when NOT using SYN-FIX-1 synthesis variant.
# ------------------------------------------------------------
set_output_delay -clock clk -max 2.000 [get_ports {cycle_count[*]}]
set_output_delay -clock clk -max 2.000 [get_ports {score_cycles[*]}]
set_output_delay -clock clk -max 2.000 [get_ports {softmax_cycles[*]}]
set_output_delay -clock clk -max 2.000 [get_ports {wsum_cycles[*]}]
set_output_delay -clock clk -max 2.000 [get_ports {load_events[*]}]
set_output_delay -clock clk -max 2.000 [get_ports {stall_cycles[*]}]

# ------------------------------------------------------------
# 5. False paths for asynchronous reset
#    rst_n is async; timing through it is meaningless to analyze.
# ------------------------------------------------------------
set_false_path -from [get_ports rst_n]

# ------------------------------------------------------------
# 6. NOTE on Q_mem / K_mem / V_mem / O_mem ports
#    These exist on the simulation-compatible attention_top.sv.
#    If SYN-FIX-1 (internal memories) is applied, these ports
#    are removed and these lines can be deleted.
#    With them present, Vivado will report IOB overuse (28k+
#    vs 100 available) — expected without SYN-FIX-1.
# ------------------------------------------------------------
# set_input_delay  -clock clk -max 2.000 [get_ports {Q_mem[*]}]
# set_input_delay  -clock clk -max 2.000 [get_ports {K_mem[*]}]
# set_input_delay  -clock clk -max 2.000 [get_ports {V_mem[*]}]
# set_output_delay -clock clk -max 2.000 [get_ports {O_mem[*]}]
