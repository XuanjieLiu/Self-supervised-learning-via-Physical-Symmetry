import numpy as np
import os


class LossCounter:
    def __init__(self, name_list):
        self.name_list = name_list
        self.values_list = []  # iter X values

    def add_values(self, values):
        self.values_list.append(values)

    def calc_values_mean(self):
        return np.mean(self.values_list, axis=0)

    def clear_values_list(self):
        self.values_list = []

    def make_record(self, num, round_idx=3):
        means = self.calc_values_mean()
        str_list = [f'{self.name_list[i]}:{round(means[i], round_idx)}' for i in range(len(self.name_list))]
        split_str = ','
        start_str = str(num) + '-'
        final_str = start_str + split_str.join(str_list) + '\n'
        return final_str

    def record_and_clear(self, record_path, num, round_idx=3):
        final_str = self.make_record(num, round_idx)
        fo = open(record_path, "a")
        fo.writelines(final_str)
        fo.close()
        self.clear_values_list()

    def load_iter_num(self, record_path):
        if os.path.exists(record_path):
            f = open(record_path, "r")
            lines = f.readlines()
            t_list = [int(a.split('-')[0]) for a in lines]
            f.close()
            return t_list[-1] + 1
        else:
            return 0
