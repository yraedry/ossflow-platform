"""CUDA environment setup for NVIDIA DLLs and PyTorch compatibility."""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("subtitler")


def setup_nvidia_dlls() -> None:
    """Patch DLL search paths for NVIDIA CUDA 12 libraries on Windows."""
    try:
        import nvidia.cublas  # type: ignore
        import nvidia.cudnn   # type: ignore

        def _find_bin(module: object) -> Optional[str]:
            if hasattr(module, "__file__") and module.__file__:
                return os.path.join(os.path.dirname(module.__file__), "bin")
            if hasattr(module, "__path__"):
                return os.path.join(list(module.__path__)[0], "bin")  # type: ignore
            return None

        for mod in (nvidia.cublas, nvidia.cudnn):
            bin_path = _find_bin(mod)
            if bin_path and os.path.exists(bin_path):
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_path)
                    except OSError:
                        pass
        log.debug("NVIDIA CUDA 12 DLL paths injected.")
    except ImportError:
        pass


def setup_pytorch_safety() -> None:
    """Register safe globals for OmegaConf and patch torch.load for WhisperX compatibility."""
    import torch

    try:
        import omegaconf
        torch.serialization.add_safe_globals([
            omegaconf.listconfig.ListConfig,
            omegaconf.dictconfig.DictConfig,
            omegaconf.base.ContainerMetadata,
        ])
    except (ImportError, AttributeError):
        pass

    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load  # type: ignore
