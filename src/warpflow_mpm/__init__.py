"""warpflow-mpm: GPU-accelerated Material Point Method solver.

NVIDIA Warp kernels for intra-node compute, MPI domain decomposition for
multi-node parallelism. A port of the Karamelo C++/MPI MPM code.
"""

import os

__version__ = "0.1.0"

# The default NVRTC+PCH compile path hangs on some setups (observed on
# WSL2 with an sm_120 Blackwell GPU); the LLVM backend is reliable and
# produces the same runtime code. One-time module compile cost is cached.
import warp as wp

if os.environ.get("WARPFLOW_LLVM_CUDA", "1") not in ("0", "false", "False"):
    wp.config.llvm_cuda = True
