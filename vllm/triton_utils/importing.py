# SPDX-License-Identifier: Apache-2.0

import importlib.machinery
import sys
import types
from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)

HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)

if not HAS_TRITON:
    # PyTorch probes Triton by importing it. Load its CPU fallback before
    # installing vLLM's placeholder, which would otherwise look like Triton.
    import torch._inductor.compile_fx  # noqa: F401

    logger.info("Triton not installed or not compatible; certain GPU-related"
                " functions will not be available.")

    class TritonPlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton")
            # torch.cuda's lazy init calls importlib.util.find_spec("triton"),
            # which raises ValueError for a module whose __spec__ is None. A
            # bare spec lets it pass; torch's real probe (has_triton_package)
            # imports triton.compiler, which this placeholder lacks, so torch
            # still sees no Triton. Needed on aarch64 (no Triton 3.2 wheels).
            self.__spec__ = importlib.machinery.ModuleSpec("triton", None)
            self.jit = self._dummy_decorator("jit")
            self.autotune = self._dummy_decorator("autotune")
            self.heuristics = self._dummy_decorator("heuristics")
            self.language = TritonLanguagePlaceholder()
            logger.warning_once(
                "Triton is not installed. Using dummy decorators. "
                "Install it via `pip install triton` to enable kernel"
                "compilation.")

        def _dummy_decorator(self, name):

            def decorator(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func

            return decorator

    class TritonLanguagePlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton.language")
            self.__spec__ = importlib.machinery.ModuleSpec(
                "triton.language", None)
            self.constexpr = None
            self.dtype = None

    sys.modules['triton'] = TritonPlaceholder()
    sys.modules['triton.language'] = TritonLanguagePlaceholder()

if 'triton' in sys.modules:
    logger.info("Triton module has been replaced with a placeholder.")
