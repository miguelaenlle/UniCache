# SPDX-License-Identifier: Apache-2.0
# timer_utils.py
class VirtualTimer:

    def __init__(self, start_time=0):
        self.start_time = start_time
        self.current_time = start_time

    def advance(self, delta_time):
        self.current_time += delta_time

    def get_current_time(self):
        return self.current_time

    def set_time(self, time):
        self.current_time = time


GLOBAL_TIMER = VirtualTimer()


def engine_time():
    """The clock the vLLM core uses: virtual time in simulation, wall time on GPU.

    Policy code must read time through this, so a policy sees the same clock as
    the block timestamps (``last_accessed``) the core hands it.
    """
    from vllm.simulator.config import GLOBAL_SIMULATOR_CONFIG
    if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
        return GLOBAL_TIMER.get_current_time()
    import time
    return time.time()
