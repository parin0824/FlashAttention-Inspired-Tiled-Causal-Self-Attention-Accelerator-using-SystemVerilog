# ==============================================================================
# build_vivado.tcl
# IO-Aware Tiled Causal Self-Attention Accelerator — Full Vivado Build Script
#
# PURPOSE:
#   Automates the complete Vivado flow:
#     1. Create project targeting xc7z010clg400-1 (Zynq-7010)
#     2. Add all RTL sources and the AXI-Lite wrapper
#     3. Create a Block Design with Zynq PS7 + custom AXI IP
#     4. Run behavioural simulation
#     5. Run synthesis with optimisation
#     6. Run implementation (place + route)
#     7. Generate bitstream
#     8. Export hardware (.xsa) for Vitis/SDK
#
# USAGE:
#   Open Vivado Tcl console or run in batch mode:
#     vivado -mode batch -source build_vivado.tcl
#
# TARGET BOARD:
#   xc7z010clg400-1  (PYNQ-Z1 / Zybo Z7-10 class)
#
# NOTES:
#   - The AXI wrapper (attention_axi_wrapper_bd) exposes a single
#     AXI4-Lite slave port + IRQ, solving the 28,868-IOB problem
#     from the standalone synthesis approach.
#   - Zynq PS7 provides clk (FCLK_CLK0) and reset, so no external
#     clock/reset pins are needed on the PL side.
#   - The block design automation handles address mapping.
# ==============================================================================

# ------------------------------------------------------------------
# 0. Configuration — edit these if your paths or part differ
# ------------------------------------------------------------------
set project_name   "attention_accel"
set project_dir    "./vivado_proj"
set part           "xc7z020clg400-1"
set board_part     ""
# Set board_part to e.g. "tul.com.tw:pynq-z1:part0:1.0" if installed.
# Leave empty to use raw part only (more portable).

set rtl_dir        "."
set tb_dir         "."
set report_dir     "./Report"

# Clock target for Zynq FCLK_CLK0 (in MHz)
set clk_freq_mhz   100

# ------------------------------------------------------------------
# 1. Create Project
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 1: Creating Vivado Project"
puts "========================================================"

# Remove old project if it exists (clean rebuild)
if {[file exists $project_dir]} {
    file delete -force $project_dir
}

create_project $project_name $project_dir -part $part -force

if {$board_part ne ""} {
    set_property board_part $board_part [current_project]
}

# Set project properties for Zynq
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# ------------------------------------------------------------------
# 2. Add RTL Source Files
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 2: Adding RTL Sources"
puts "========================================================"

# All SystemVerilog RTL sources in compile order
set rtl_files [list \
    "$rtl_dir/attention_pkg.sv"        \
    "$rtl_dir/addr_gen.sv"             \
    "$rtl_dir/q_buffer.sv"             \
    "$rtl_dir/kv_buffer.sv"            \
    "$rtl_dir/output_buffer.sv"        \
    "$rtl_dir/perf_counters.sv"        \
    "$rtl_dir/attention_controller.sv" \
    "$rtl_dir/score_engine.sv"         \
    "$rtl_dir/mask_unit.sv"            \
    "$rtl_dir/row_score_store.sv"      \
    "$rtl_dir/row_max_unit.sv"         \
    "$rtl_dir/exp_lut.sv"              \
    "$rtl_dir/row_sum_unit.sv"         \
    "$rtl_dir/reciprocal_lut.sv"       \
    "$rtl_dir/normalizer.sv"           \
    "$rtl_dir/weighted_sum_engine.sv"  \
    "$rtl_dir/attention_top.sv"        \
    "$rtl_dir/attention_axi_wrapper.sv" \
    "$rtl_dir/attention_axi_wrapper.v" \
]

add_files -norecurse $rtl_files
set_property file_type SystemVerilog [get_files *.sv]
set_property file_type Verilog       [get_files *.v]

# Ensure attention_pkg.sv is read first (global include)
set_property is_global_include true [get_files "$rtl_dir/attention_pkg.sv"]

# ------------------------------------------------------------------
# 3. Add Simulation Sources (testbench)
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 3: Adding Simulation Sources"
puts "========================================================"

set tb_files [list \
    "$tb_dir/tb_attention_top.sv" \
]

add_files -fileset sim_1 -norecurse $tb_files
set_property file_type SystemVerilog [get_files -of_objects [get_filesets sim_1] *.sv]
set_property top tb_attention_top [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]

# ------------------------------------------------------------------
# 4. Create Block Design
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 4: Creating Block Design"
puts "========================================================"

set bd_name "attention_system"

create_bd_design $bd_name

# --- 4a. Add Zynq PS7 ---
puts "  Adding ZYNQ PS7 Processing System..."
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]

# Configure PS7: enable FCLK_CLK0, GP0 AXI master, reset, fabric interrupts
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $clk_freq_mhz \
    CONFIG.PCW_USE_FABRIC_INTERRUPT      {1}            \
    CONFIG.PCW_IRQ_F2P_INTR              {1}            \
    CONFIG.PCW_USE_M_AXI_GP0             {1}            \
] $ps7

# Apply board preset if available (sets DDR, MIO, etc. automatically)
if {$board_part ne ""} {
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable"} \
        $ps7
}

# --- 4b. Add the Attention AXI Wrapper as RTL module ---
puts "  Adding Attention AXI Wrapper RTL module to block design..."
set attn_ip [create_bd_cell -type module -reference attention_axi_wrapper_bd attention_core_0]

# --- 4c. Add AXI Interconnect ---
puts "  Adding AXI Interconnect..."
set axi_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0]
set_property -dict [list \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_SI {1} \
] $axi_ic

# --- 4d. Add Processor System Reset ---
puts "  Adding Processor System Reset..."
set ps_reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]

# --- 4e. Wiring: Clock and Reset ---
puts "  Wiring clocks and resets..."

# PS7 FCLK_CLK0 -> everything
# NOTE: The attention_axi_wrapper.sv declares ports as UPPERCASE (S_AXI_ACLK).
#       The .v wrapper (attention_axi_wrapper_bd) also uses UPPERCASE port names.
#       Vivado BD pin names are derived from the RTL port names, so they are UPPERCASE.
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins axi_interconnect_0/ACLK] \
    [get_bd_pins axi_interconnect_0/S00_ACLK] \
    [get_bd_pins axi_interconnect_0/M00_ACLK] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins attention_core_0/S_AXI_ACLK] \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK]

# PS7 FCLK_RESET0_N -> reset block ext_reset_in
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
    [get_bd_pins proc_sys_reset_0/ext_reset_in]

# Reset block interconnect_aresetn -> interconnect
connect_bd_net [get_bd_pins proc_sys_reset_0/interconnect_aresetn] \
    [get_bd_pins axi_interconnect_0/ARESETN] \
    [get_bd_pins axi_interconnect_0/S00_ARESETN] \
    [get_bd_pins axi_interconnect_0/M00_ARESETN]

# Reset block peripheral_aresetn -> attention core
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins attention_core_0/S_AXI_ARESETN]

# --- 4f. Wiring: AXI data path ---
puts "  Wiring AXI data path..."

# PS7 M_AXI_GP0 -> interconnect S00_AXI
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] \
    [get_bd_intf_pins axi_interconnect_0/S00_AXI]

# Interconnect M00_AXI -> attention core S_AXI
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] \
    [get_bd_intf_pins attention_core_0/S_AXI]

# --- 4g. Wiring: IRQ ---
puts "  Wiring IRQ..."
connect_bd_net [get_bd_pins attention_core_0/irq] \
    [get_bd_pins processing_system7_0/IRQ_F2P]

# --- 4h. Make external ports (DDR, FIXED_IO) ---
puts "  Making external DDR and FIXED_IO ports..."
make_bd_intf_pins_external [get_bd_intf_pins processing_system7_0/DDR]
make_bd_intf_pins_external [get_bd_intf_pins processing_system7_0/FIXED_IO]

# --- 4i. Assign address space ---
puts "  Assigning address space..."
assign_bd_address

# --- 4j. Validate and save ---
puts "  Validating block design..."
validate_bd_design

puts "  Saving block design..."
save_bd_design

# --- 4k. Create HDL wrapper ---
puts "  Creating HDL wrapper for block design..."
set bd_file [get_files "${bd_name}.bd"]
set wrapper_file [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper_file
set_property top [file rootname [file tail $wrapper_file]] [current_fileset]

# Update compile order
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

puts "  Block design creation complete."

# ------------------------------------------------------------------
# 5. Add Timing Constraints
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 5: Adding Timing Constraints"
puts "========================================================"

# The Zynq PS7 FCLK_CLK0 is auto-constrained, but we add explicit
# constraints for completeness and to ensure clean timing reports.
set xdc_file "$project_dir/timing_constraints.xdc"
set xdc_fh [open $xdc_file w]

puts $xdc_fh "# =============================================="
puts $xdc_fh "# Attention Accelerator - Timing Constraints"
puts $xdc_fh "# Target: ${clk_freq_mhz} MHz on Zynq FCLK_CLK0"
puts $xdc_fh "# =============================================="
puts $xdc_fh ""
puts $xdc_fh "# Primary clock is provided by PS7 FCLK_CLK0."
puts $xdc_fh "# Vivado auto-creates this constraint from the PS7 IP config."
puts $xdc_fh ""
puts $xdc_fh "# All PL logic uses the single FCLK_CLK0 domain."
puts $xdc_fh "# No false paths or multicycle paths required."
puts $xdc_fh ""

close $xdc_fh

add_files -fileset constrs_1 -norecurse $xdc_file
set_property used_in_synthesis       true [get_files $xdc_file]
set_property used_in_implementation  true [get_files $xdc_file]

puts "  Timing constraints added."

# ------------------------------------------------------------------
# 6. Run Behavioural Simulation
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 6: Running Behavioural Simulation"
puts "========================================================"

# Set simulation properties
set_property -name {xsim.simulate.runtime} -value {all} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.log_all_signals} -value {true} -objects [get_filesets sim_1]

# Launch simulation
launch_simulation -mode behavioral

# Wait for simulation to complete (tb has $finish)
puts "  Simulation complete."

# Close simulation to free resources before synthesis
close_sim -force -quiet

# ------------------------------------------------------------------
# 7. Run Synthesis
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 7: Running Synthesis"
puts "========================================================"

# Configure synthesis strategy for best results
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE PerformanceOptimized [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING on [get_runs synth_1]

# Reset and launch synthesis
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis status
set synth_status [get_property STATUS [get_runs synth_1]]
puts "  Synthesis status: $synth_status"

if {[string match "*ERROR*" $synth_status] || [string match "*FAILED*" $synth_status]} {
    puts "ERROR: Synthesis failed. Check logs in $project_dir."
}

# Open synthesised design for reports
open_run synth_1 -name synth_1

# ------------------------------------------------------------------
# 8. Generate Post-Synthesis Reports
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 8: Generating Post-Synthesis Reports"
puts "========================================================"

# Create report directory if it doesn't exist
file mkdir $report_dir

report_utilization     -file "$report_dir/synth_utilization.rpt"
report_timing_summary  -file "$report_dir/synth_timing_summary.rpt" \
                       -max_paths 10
report_power           -file "$report_dir/synth_power.rpt"
report_clock_networks  -file "$report_dir/synth_clock_networks.rpt"
report_methodology     -file "$report_dir/synth_methodology.rpt"
report_drc             -file "$report_dir/synth_drc.rpt"

puts "  Synthesis reports saved to $report_dir/"

# ------------------------------------------------------------------
# 9. Run Implementation (Place & Route)
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 9: Running Implementation"
puts "========================================================"

# Configure implementation strategy for best timing closure
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE ExploreWithRemap [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraNetDelay_high [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

launch_runs impl_1 -jobs 4
wait_on_run impl_1

set impl_status [get_property STATUS [get_runs impl_1]]
puts "  Implementation status: $impl_status"

if {[string match "*ERROR*" $impl_status] || [string match "*FAILED*" $impl_status]} {
    puts "ERROR: Implementation failed. Check logs in $project_dir."
}

# Open implemented design for reports
open_run impl_1 -name impl_1

# ------------------------------------------------------------------
# 10. Generate Post-Implementation Reports
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 10: Generating Post-Implementation Reports"
puts "========================================================"

report_utilization      -file "$report_dir/impl_utilization.rpt"
report_timing_summary   -file "$report_dir/impl_timing_summary.rpt" \
                        -max_paths 20
report_timing           -file "$report_dir/impl_timing_detail.rpt" \
                        -max_paths 50 \
                        -sort_by group -path_type full
report_power            -file "$report_dir/impl_power.rpt"
report_clock_utilization -file "$report_dir/impl_clock_utilization.rpt"
report_route_status     -file "$report_dir/impl_route_status.rpt"
report_io               -file "$report_dir/impl_io.rpt"
report_drc              -file "$report_dir/impl_drc.rpt"
report_methodology      -file "$report_dir/impl_methodology.rpt"

# Hierarchical utilization for detailed area breakdown
report_utilization -hierarchical -file "$report_dir/impl_utilization_hierarchical.rpt"

puts "  Implementation reports saved to $report_dir/"

# ------------------------------------------------------------------
# 11. Generate Bitstream
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 11: Generating Bitstream"
puts "========================================================"

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

set bit_status [get_property STATUS [get_runs impl_1]]
puts "  Bitstream generation status: $bit_status"

# Find and report the bitstream file location
set bit_file [glob -nocomplain "$project_dir/$project_name.runs/impl_1/*.bit"]
if {$bit_file ne ""} {
    puts "  Bitstream file: $bit_file"

    # Copy bitstream to project root for easy access
    file copy -force $bit_file "./${project_name}.bit"
    puts "  Bitstream copied to ./${project_name}.bit"
} else {
    puts "WARNING: Bitstream file not found. Check impl_1 run logs."
}

# ------------------------------------------------------------------
# 12. Export Hardware (.xsa) for Vitis / SDK
# ------------------------------------------------------------------
puts "========================================================"
puts " STEP 12: Exporting Hardware Platform (.xsa)"
puts "========================================================"

write_hw_platform -fixed -include_bit -force -file "./${project_name}.xsa"
puts "  Hardware platform exported to ./${project_name}.xsa"

# ------------------------------------------------------------------
# 13. Final Summary
# ------------------------------------------------------------------
puts ""
puts "========================================================"
puts " BUILD COMPLETE - SUMMARY"
puts "========================================================"
puts ""
puts "  Project     : $project_dir/$project_name.xpr"
puts "  Part        : $part"
puts "  Clock       : ${clk_freq_mhz} MHz (FCLK_CLK0)"
puts ""
puts "  Reports     : $report_dir/"
puts "    Synthesis   : synth_utilization.rpt, synth_timing_summary.rpt, synth_power.rpt"
puts "    Impl        : impl_utilization.rpt, impl_timing_summary.rpt, impl_power.rpt"
puts "    Timing      : impl_timing_detail.rpt"
puts "    Debug       : impl_drc.rpt, impl_methodology.rpt"
puts ""
puts "  Bitstream   : ./${project_name}.bit"
puts "  HW Platform : ./${project_name}.xsa"
puts ""

# Check final timing
set wns [get_property STATS.WNS [get_runs impl_1]]
set tns [get_property STATS.TNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]

puts "  Timing Results:"
puts "    WNS (setup) : ${wns} ns"
puts "    TNS (setup) : ${tns} ns"
puts "    WHS (hold)  : ${whs} ns"

if {$wns >= 0 && $whs >= 0} {
    puts ""
    puts "  *** TIMING MET - Design is clean! ***"
} else {
    puts ""
    puts "  *** WARNING: Timing violations detected. ***"
    puts "  Review impl_timing_detail.rpt for failing paths."
}

puts ""
puts "========================================================"
puts " Done."
puts "========================================================"
