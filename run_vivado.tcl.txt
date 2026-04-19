# ============================================================
# Vivado Tcl Script for attention_top
# Target: Zynq-7000 (xc7z010clg400-1)
# ============================================================

# Clean project name / directory
set proj_name "attention_top_vivado"
set proj_dir "./vivado_proj"
set top_module "attention_top"
set part_name "xc7z010clg400-1"

# Create project
create_project $proj_name $proj_dir -part $part_name -force

# Optional: set target language
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# Add RTL sources
add_files -norecurse ./attention_pkg.sv
add_files -norecurse ./addr_gen.sv
add_files -norecurse ./attention_controller.sv
add_files -norecurse ./exp_lut.sv
add_files -norecurse ./kv_buffer.sv
add_files -norecurse ./mask_unit.sv
add_files -norecurse ./normalizer.sv
add_files -norecurse ./output_buffer.sv
add_files -norecurse ./perf_counters.sv
add_files -norecurse ./q_buffer.sv
add_files -norecurse ./reciprocal_lut.sv
add_files -norecurse ./row_max_unit.sv
add_files -norecurse ./row_score_store.sv
add_files -norecurse ./row_sum_unit.sv
add_files -norecurse ./score_engine.sv
add_files -norecurse ./weighted_sum_engine.sv
add_files -norecurse ./attention_top.sv

# Add testbench only for simulation, not synthesis
add_files -fileset sim_1 -norecurse ./tb_attention_top.sv

# Set top module
set_property top $top_module [current_fileset]
set_property top tb_attention_top [get_filesets sim_1]

# Optional constraint file
# Uncomment if you have one
# add_files -fileset constrs_1 -norecurse ./attention_top.xdc

# Update compile order
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# Run synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Open synthesized design and write reports
open_run synth_1
report_utilization -file $proj_dir/synth_utilization.rpt
report_timing_summary -file $proj_dir/synth_timing_summary.rpt
report_power -file $proj_dir/synth_power.rpt

# Optional: run implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Open implemented design and write reports
open_run impl_1
report_utilization -file $proj_dir/impl_utilization.rpt
report_timing_summary -file $proj_dir/impl_timing_summary.rpt
report_power -file $proj_dir/impl_power.rpt

# Optional: generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

puts "Vivado flow completed successfully."