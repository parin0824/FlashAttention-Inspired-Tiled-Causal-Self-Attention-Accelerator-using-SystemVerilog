# ============================================================
# Vivado GUI-friendly Tcl for Option B
# Zynq-7000 PS + AXI4-Lite wrapped attention accelerator
#
# Usage from Vivado GUI Tcl console:
#   cd <directory containing RTL + this script>
#   source run_vivado_option_b_gui.tcl
#
# This script CREATES the project and block design, but does not force
# synthesis/implementation/bitstream. You can run those from the GUI.
# Helper procs are provided at the end.
# ============================================================

set proj_name "attention_zynq_option_b_gui"
set proj_dir  "./vivado_option_b_gui"
set bd_name   "design_1"
set part_name "xc7z010clg400-1"

# Clean create project
create_project $proj_name $proj_dir -part $part_name -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# RTL sources
set rtl_files [list \
  ./attention_pkg.sv \
  ./addr_gen.sv \
  ./attention_controller.sv \
  ./exp_lut.sv \
  ./kv_buffer.sv \
  ./mask_unit.sv \
  ./normalizer.sv \
  ./output_buffer.sv \
  ./perf_counters.sv \
  ./q_buffer.sv \
  ./reciprocal_lut.sv \
  ./row_max_unit.sv \
  ./row_score_store.sv \
  ./row_sum_unit.sv \
  ./score_engine.sv \
  ./weighted_sum_engine.sv \
  ./attention_top.sv \
  ./attention_axi_wrapper.v \
]

foreach f $rtl_files {
  if {![file exists $f]} {
    error "Missing source file: $f"
  }
  add_files -norecurse $f
}
update_compile_order -fileset sources_1

# ------------------------------------------------------------------
# Block design creation
# ------------------------------------------------------------------
create_bd_design $bd_name

# Processing system
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} \
  [get_bd_cells processing_system7_0]

# Enable M_AXI_GP0 and fabric interrupt
set_property -dict [list \
  CONFIG.PCW_USE_M_AXI_GP0 {1} \
  CONFIG.PCW_IRQ_F2P_INTR {1} \
  CONFIG.PCW_USE_FABRIC_INTERRUPT {1}] [get_bd_cells processing_system7_0]

# Reset block
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0]     [get_bd_pins rst_ps7_0_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]

# AXI interconnect
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}] [get_bd_cells smartconnect_0]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0]       [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins rst_ps7_0_100M/interconnect_aresetn]  [get_bd_pins smartconnect_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins smartconnect_0/S00_AXI]

# Custom accelerator
create_bd_cell -type module -reference attention_axi_wrapper_bd attention_axi_wrapper_0
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0]        [get_bd_pins attention_axi_wrapper_0/S_AXI_ACLK]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn]     [get_bd_pins attention_axi_wrapper_0/S_AXI_ARESETN]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI]      [get_bd_intf_pins attention_axi_wrapper_0/S_AXI]
connect_bd_net [get_bd_pins attention_axi_wrapper_0/irq]           [get_bd_pins processing_system7_0/IRQ_F2P]

# Address assignment
assign_bd_address

# Save and validate BD
regenerate_bd_layout
save_bd_design
validate_bd_design

# Generate HDL wrapper and set top
make_wrapper -files [get_files ${proj_dir}/${proj_name}.srcs/sources_1/bd/${bd_name}/${bd_name}.bd] -top
add_files -norecurse ${proj_dir}/${proj_name}.srcs/sources_1/bd/${bd_name}/hdl/${bd_name}_wrapper.v
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Open the block design and sources view in GUI
open_bd_design [get_files ${proj_dir}/${proj_name}.srcs/sources_1/bd/${bd_name}/${bd_name}.bd]
start_gui

puts ""
puts "Project created successfully in GUI-friendly mode."
puts "Top module: ${bd_name}_wrapper"
puts "Block design: ${bd_name}"
puts ""
puts "Next steps in GUI:"
puts "  1. Inspect Address Editor and validate the BD"
puts "  2. Run Synthesis"
puts "  3. Run Implementation"
puts "  4. Generate Bitstream"
puts ""
puts "Helper commands available:"
puts "  build_synth"
puts "  build_impl"
puts "  build_bitstream"
puts ""

# ------------------------------------------------------------------
# Helper procedures for GUI use
# ------------------------------------------------------------------
proc build_synth {} {
  launch_runs synth_1 -jobs 4
}

proc build_impl {} {
  launch_runs impl_1 -jobs 4
}

proc build_bitstream {} {
  launch_runs impl_1 -to_step write_bitstream -jobs 4
}
