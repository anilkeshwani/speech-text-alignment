# AGENTS.md

## Runtime Environment: Ubuntu NVIDIA A6000 GPUs (48GB)

All development in this repository (even when conducted on macOS) ultimately targets a runtime on an Linux Ubuntu machine.

```bash
$ uname -a
Linux artemis 5.15.0-124-generic #134-Ubuntu SMP Fri Sep 27 20:20:17 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux
```

Specifically, any GPU-native code is run on NVIDIA A6000 GPUs with 48GB of VRAM. Execution should assume runs on a single GPU as NVLink is not available so inter-GPU communication overhead is high.
