#ifndef VERILATOR_BNN_BRIDGE_H
#define VERILATOR_BNN_BRIDGE_H

#include "bnn-accelerator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bnn_rtl_sim BNN_RtlSim;

/* Runtime waveform switch, sampled at device/simulation creation.
 * Unset or empty: disabled. Otherwise an existing writable directory receives
 * one unique bnn-core-XXXXXX.vcd per instance, including reset and all jobs.
 * The directory is never created implicitly. An open failure returns BNN_EIO.
 * Close the device to finish the file; each completed/failed job also flushes it.
 */
#define BNN_RTL_TRACE_DIR_ENV "BNN_RTL_TRACE_DIR"

/* Each simulation instance owns its context and can be reused across jobs. */
BNN_Status bnn_rtl_create(BNN_RtlSim** out_sim);
void bnn_rtl_destroy(BNN_RtlSim* sim);

/* One vector, packed row-major W1 weights, contiguous [N][K/G] int32 output.
 * Input pointers must span their implied sizes. max_cycles=0 selects the
 * shape-derived budget. Failure may modify output, but resets the model so
 * a subsequent job can run. No C++ exceptions cross this interface.
 */
BNN_Status bnn_rtl_dot_groups(BNN_RtlSim* sim, const int8_t* input, const uint8_t* weights,
                              uint32_t in_features, uint32_t out_features, uint32_t group_size,
                              int32_t* output, uint64_t max_cycles);

#ifdef __cplusplus
}
#endif

#endif
