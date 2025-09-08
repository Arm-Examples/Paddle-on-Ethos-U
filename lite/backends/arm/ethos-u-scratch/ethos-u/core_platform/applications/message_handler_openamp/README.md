# Enabling trace resource in the message_handler_openamp application

The message_handler_openamp application provides support for logging to a trace buffer via the trace entry in the resource table. The following instructions briefly describe one way of enabling the trace resource in the firmware's resource table:

1. Enable building message_handler_openamp with trace buffer support by setting the CMake BOOL cache entry ENABLE_REMOTEPROC_TRACE_BUFFER to ON in the target's CMakeLists.txt

2. Update the target's linker scripts to include an `ethosu_core_trace_buffer` section that is placed in the DRAM as shown in the examples below. The linker-defined symbols for the trace buffer's address and size are later used by the message_handler_openamp to setup the trace buffer in the resource table.

linker-script.ld:
```
SECTIONS
{
  .dram ():
  {
    /* Trace buffer */
    __ethosu_core_trace_buffer_start__ = .;
    *(ethosu_core_trace_buffer)
    __ethosu_core_trace_buffer_size__ = ABSOLUTE(. - __ethosu_core_trace_buffer_start__);
  } > DRAM_REGION
```

scatter-file.scatter:
```
LOAD_REGION_DRAM DRAM_ADDRESS DRAM_SIZE
{
    /* Trace buffer */
    trace_buffer (DRAM_ADDRESS + 0x20000) FIXED
    {
        * (ethosu_core_trace_buffer)
    }
}
```

3. Allocate the trace buffer and place it in the section defined above using the `section` attribute. Finally, implement the `fputc` function in the retarget file so that writes to stderr and stdout are redirected to the trace buffer. Example:

retarget.c:
```
#define TRACE_BUFFER_SIZE 0xXXXX
__attribute__((section("ethosu_core_trace_buffer"))) char trace_buffer[TRACE_BUFFER_SIZE] = {0};
int fputc(int ch, FILE *f) {
    (void)f;
    uint8_t c = (uint8_t)ch;

    static uint32_t write = 0;
    trace_buffer[write % TRACE_BUFFER_SIZE] = c;
    write++;

    return c;
}
```

