from .builder import HCUMegatronOpBuilder


class AlgorithmOpBuilder(HCUMegatronOpBuilder):
    OP_NAME = "algorithm"

    def __init__(self):
        super(AlgorithmOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/algorithm/algorithm.cpp']

    def compiled_files(self):
        return ['ops/csrc/algorithm/compiled/algorithm.so']
