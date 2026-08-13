# Copyright (c) 2026, HYGON-AI. All rights reserved.
#
# Makes tests a regular package: the Megatron-Energon submodule ships its own
# regular `tests` package (with __init__.py) that shadows this repo's namespace
# package on PYTHONPATH, so `from tests.unit_tests...` must resolve here first.
