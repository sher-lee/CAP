"""
CAP Layer 1: Simulation Backends
=================================
Drop-in replacements for real hardware modules. Every class here
implements the same interface as its real counterpart in the parent
directory, but operates entirely in memory with no GPIO, serial,
or motor hardware required.

Activated when hardware_mode == "simulation" in cap_config.yaml.
"""
