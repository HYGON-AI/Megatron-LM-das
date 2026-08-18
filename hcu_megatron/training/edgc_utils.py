# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import os
import csv
from datetime import datetime
from megatron.training.global_vars import get_args
from scipy.interpolate import interp1d
from megatron.core import mpu
import torch


def read_data_from_csv(file_path, iteration):
    with open(file_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['iteration']) == iteration:
                return eval(row['data'])
    return None


def read_error_from_csv(file_path):
    with open(file_path, mode='r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            return row[0]
    return None


def append_time_to_csv(file_path, iteration):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Iteration'])
        writer.writerow([current_time, iteration])


def append_data_to_csv(file_path, iteration, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['iteration', 'data'])
        writer.writerow([iteration, data])


def append_init_error_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['data'])
        writer.writerow([data])


class Utils:
    data = {}
    loss = []
    mapped_rank = []

    @staticmethod
    def is_find_rank_upper_limit(params_all_reduce_time):
        args = get_args()
        current_iteration = args.curr_iteration
        base_iteration = args.latest_iteration if args.is_loading_checkpoint else 0
        base_iteration += 9
        if current_iteration > base_iteration:
            rank_size = 2 ** int((current_iteration - base_iteration) / 3)
            mod_iter = (current_iteration - base_iteration) % 3
            if mod_iter == 1:
                if current_iteration == base_iteration + 1:
                    Utils.data[0] = params_all_reduce_time
                else:
                    Utils.data[rank_size] = params_all_reduce_time
            elif mod_iter == 2:
                if current_iteration == base_iteration + 2:
                    Utils.data[0] = int((Utils.data[0] + params_all_reduce_time) / 2)
                else:
                    Utils.data[rank_size] = int((Utils.data[rank_size] + params_all_reduce_time) / 2)
                    time_value = Utils.data.get(0)
                    if Utils.data[rank_size] > time_value:
                        max_rank = Utils.time_predict_rank(time_value)
                        return True, max_rank

        return False, None

    @staticmethod
    def map_loss_change_to_rank(min_rank=8, max_rank=32, window_size=1000):
        args = get_args()
        step = 4
        iter_sample_interval = int(1 / args.iteration_sample_ratio)
        required_points = int(2 * window_size / iter_sample_interval)
        if len(Utils.loss) < required_points:
            default_rank = max_rank
            Utils.mapped_rank.append(default_rank)
            return default_rank
        sample_window = int(window_size / iter_sample_interval)
        curr_window = Utils.loss[-sample_window:]
        prev_window = Utils.loss[-2 * sample_window:-sample_window]
        curr_mean = sum(curr_window) / sample_window
        prev_mean = sum(prev_window) / sample_window
        loss_change = abs(curr_mean - prev_mean)

        min_loss_change = 0.0
        if args.is_loading_checkpoint:
            first_save_interval = read_data_from_csv(args.loss_path, args.save_interval)
        else:
            first_save_interval = Utils.loss
        first_window = first_save_interval[:sample_window]
        second_window = first_save_interval[sample_window:2 * sample_window]
        first_mean = sum(first_window) / sample_window
        second_mean = sum(second_window) / sample_window
        max_loss_change = abs(first_mean - second_mean)

        norm_loss_change = loss_change / (max_loss_change - min_loss_change)
        mapped_rank = min_rank + (max_rank - min_rank) * norm_loss_change
        mapped_rank = round(mapped_rank)
        mapped_rank = max(min_rank, min(mapped_rank, max_rank))
        mapped_rank = (mapped_rank // 2) * 2
        if not Utils.mapped_rank:
            final_rank = mapped_rank if (max_rank - mapped_rank) <= step else (max_rank - step)
        else:
            prev_rank = Utils.mapped_rank[-1]
            delta = mapped_rank - prev_rank
            if abs(delta) > step:
                final_rank = prev_rank + step if delta > 0 else prev_rank - step
            else:
                final_rank = mapped_rank

        Utils.mapped_rank.append(final_rank)
        return final_rank

    @staticmethod
    def use_rank_predict_time(rank):
        keys, values = list(Utils.data.keys())[1:], list(Utils.data.values())[1:]
        interpolation = interp1d(keys, values, kind='linear', fill_value="extrapolate")
        return interpolation(rank)

    @staticmethod
    def use_time_predict_rank(first_stage_predict_time):
        args = get_args()
        keys, values = list(Utils.data.keys())[1:], list(Utils.data.values())[1:]
        baseline = list(Utils.data.values())[0]
        times = (
                    mpu.get_pipeline_model_parallel_rank() * args.per_microbatch_time + first_stage_predict_time).cpu().numpy()
        if times > baseline:
            return None
        interpolation = interp1d(values, keys, kind='linear', fill_value='extrapolate')
        predicted_rank = ((interpolation(times) // 4) + 1) * 4
        return predicted_rank if predicted_rank <= args.max_rank else None

    @staticmethod
    def start_use_time_predict_rank():
        keys, values = list(Utils.data.keys())[1:], list(Utils.data.values())[1:]
        baseline = list(Utils.data.values())[0]
        interpolation = interp1d(values, keys, kind='linear', fill_value='extrapolate')
        return int((interpolation(baseline) // 4) * 4)

    @staticmethod
    def syn_data_parallel_group():
        args = get_args()
        signal = torch.tensor(args.max_rank if args.find_rank_upper_limit else 0.0, device="cuda", dtype=torch.float64)
        signal_list = [torch.zeros(1, device="cuda", dtype=torch.float64) for _ in
                       range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather(signal_list, signal, group=mpu.get_data_parallel_group())

        if any(s.item() > 0 for s in signal_list):
            if not args.find_rank_upper_limit:
                args.max_rank = Utils.start_use_time_predict_rank()
                args.find_rank_upper_limit = True

    @staticmethod
    def second_syn_data_parallel_group(rank):
        tensor_rank = torch.tensor(rank if rank is not None else 0.0, device="cuda", dtype=torch.float64)
        rank_list = [torch.zeros(1, device="cuda", dtype=torch.float64) for _ in
                     range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather(rank_list, tensor_rank, group=mpu.get_data_parallel_group())
        values = [int(t.item()) for t in rank_list if t.item() > 0]
        return None if all(v == values[0] for v in values) else max(values)

    @staticmethod
    def syn_pipeline_parallel_group():
        args = get_args()
        signal = torch.tensor(mpu.get_pipeline_model_parallel_rank() if args.find_rank_upper_limit else -1,
                              device="cuda", dtype=torch.float64)
        signal_list = [torch.zeros(1, device="cuda", dtype=torch.float64) for _ in
                       range(mpu.get_pipeline_model_parallel_world_size())]
        torch.distributed.all_gather(signal_list, signal, group=mpu.get_pipeline_model_parallel_group())
        for s in signal_list:
            if s.item() >= 0:
                if not args.find_rank_upper_limit:
                    args.find_rank_upper_limit = True
                    args.max_rank = Utils.start_use_time_predict_rank()
                break

    @staticmethod
    def syn_tensor_parallel_group():
        args = get_args()
        signal = torch.tensor(args.max_rank if args.find_rank_upper_limit else 0.0, device="cuda", dtype=torch.float64)
        signal_list = [torch.zeros(1, device="cuda", dtype=torch.float64) for _ in
                       range(mpu.get_tensor_model_parallel_world_size())]
        torch.distributed.all_gather(signal_list, signal, group=mpu.get_tensor_model_parallel_group())

        for s in signal_list:
            if s.item() > 1:
                args.find_rank_upper_limit = True
                args.max_rank = Utils.start_use_time_predict_rank()
                break

    @staticmethod
    def adjust_rank(rank):
        tensor_rank = torch.tensor(rank if rank is not None else 0.0, device="cuda", dtype=torch.float64)
        rank_list = [torch.zeros(1, device="cuda", dtype=torch.float64) for _ in
                     range(mpu.get_tensor_model_parallel_world_size())]
        torch.distributed.all_gather(rank_list, tensor_rank, group=mpu.get_tensor_model_parallel_group())

        non_zero_ranks = torch.stack(rank_list)[torch.any(torch.stack(rank_list) != 0, dim=1)]
        if non_zero_ranks.size(0) == 0:
            return None, None

        median_rank = int(torch.median(non_zero_ranks, dim=0)[0].item())
        return median_rank, Utils.use_rank_predict_time(median_rank)

    @staticmethod
    def syn_rank(rank):
        tensor_rank = torch.tensor(rank if rank is not None else 0, device="cuda", dtype=torch.float64)

        torch.distributed.broadcast(tensor_rank, src=0, group=torch.distributed.group.WORLD)

        final_rank = int(tensor_rank.item())
        return final_rank, Utils.use_rank_predict_time(final_rank)

    @staticmethod
    def time_predict_rank(time):
        keys, values = list(Utils.data.keys())[1:], list(Utils.data.values())[1:]
        interpolation = interp1d(values, keys, kind='linear', fill_value='extrapolate')
        return int((interpolation(time) // 4) * 4)
