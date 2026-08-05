# Some of this code was adopted from https://github.com/MayDomine/Seq1F1B/

from sympy import symbols, Eq, solve


def round_down(x, tp_size):
    return x // tp_size * tp_size


class solver:
    def __init__(self, total_seqlen, config, causal=True):
        self.total_seqlen = total_seqlen
        self.config = config
        self.total_tflops = config.get_seq_tflops(total_seqlen, causal)

    def solve_partition(self, num_splits, tp_size=1):
        res = []
        prefix = self.total_seqlen
        for i in range(1, num_splits):
            seqlen = symbols('seqlen')
            tflops = self.config.get_prefix_tflops(seqlen, prefix)
            eq = Eq(tflops, self.total_tflops / num_splits)
            sol = solve(eq, seqlen)
            sol = round_down(int(sol[0]), tp_size)
            res.insert(0, int(sol))
            prefix -= int(sol)
        res.insert(0, prefix)
        return res
