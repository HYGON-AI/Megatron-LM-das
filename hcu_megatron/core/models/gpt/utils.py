# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

SkipEmbeddingAllocation = False

def get_skip_embedding_allocation():
    global SkipEmbeddingAllocation
    return SkipEmbeddingAllocation


def set_skip_embedding_allocation(skip_embedding_allocation):
    global SkipEmbeddingAllocation
    SkipEmbeddingAllocation = skip_embedding_allocation


class SkipEmbeddingAllocationContextManager:
    """A reusable context manager for switch SkipEmbeddingAllocation"""

    def __init__(self, skip_embedding_allocation):
        self.skip_embedding_allocation = skip_embedding_allocation

    def __enter__(self):
        self.origin_skip_embedding_allocation = get_skip_embedding_allocation()
        set_skip_embedding_allocation(self.skip_embedding_allocation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_skip_embedding_allocation(self.origin_skip_embedding_allocation)
