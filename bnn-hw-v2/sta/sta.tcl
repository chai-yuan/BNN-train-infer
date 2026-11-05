# OpenSTA script for one corner. The Makefile supplies the environment:
#   TOP NETLIST LIB CLK_PORT CLK_PERIOD CLK_UNC TOP_N
#   IO_DELAY OUT_LOAD JSON_OUT RPT_OUT
#
# Fmax is derived from the setup slack: Fmax = 1 / (period - slack).
# JSON_OUT is machine-readable for summarize.py; RPT_OUT is human-readable.

set top      $env(TOP)
set netlist  $env(NETLIST)
set lib      $env(LIB)
set clk_port $env(CLK_PORT)
set period   $env(CLK_PERIOD)
set unc      $env(CLK_UNC)
set top_n    $env(TOP_N)
set json_out $env(JSON_OUT)
set rpt_out  $env(RPT_OUT)

read_liberty $lib
read_verilog $netlist
link_design $top

create_clock -name clk -period $period [get_ports $clk_port]
if {$unc ne ""} {
    set_clock_uncertainty $unc [get_clocks clk]
}

# Optional interface constraints. They are off by default so the reported
# Fmax is an internal reg-to-reg number.
if {[info exists env(IO_DELAY)] && $env(IO_DELAY) ne ""} {
    set io_delay $env(IO_DELAY)
    set non_clock_inputs [remove_from_collection [all_inputs] [get_ports $clk_port]]
    set_input_delay $io_delay -clock clk $non_clock_inputs
    set_output_delay $io_delay -clock clk [all_outputs]
}
if {[info exists env(OUT_LOAD)] && $env(OUT_LOAD) ne ""} {
    set_load $env(OUT_LOAD) [all_outputs]
}

# Machine-readable path data for report generation.
report_checks -path_delay max -group_path_count $top_n -format json -digits 6 > $json_out
# Full path detail for the worst paths.
report_checks -path_delay max -group_path_count $top_n -format full_clock_expanded -digits 4 > $rpt_out

exit
