# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import os
import pytest


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    """

    parser.addoption(
        '--rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    parser.addoption(
        '--world-size',
        type=int,
        default=8,
        help='number of nodes for distributed training')
    parser.addoption(
        '--dist-url',
        help='Which master node url for distributed training.')
    parser.addoption(
        '--local-rank',
        type=int,
        default=int(os.getenv('LOCAL_RANK', '0')),
        help='local rank passed from distributed launcher.')
    parser.addoption(
        '--distributed-backend',
        default='nccl',
        choices=['nccl', 'gloo'],
        help='Which backend to use for distributed training.')


@pytest.fixture(scope='session')
def distributed_args(request):
    return {
        'rank': request.config.getoption('--rank'),
        'dist_url': request.config.getoption('--dist-url'),
        'world_size': request.config.getoption('--world-size'),
        'local_rank': request.config.getoption('--local-rank'),
        'distributed_backend': request.config.getoption('--distributed-backend'),
    }
