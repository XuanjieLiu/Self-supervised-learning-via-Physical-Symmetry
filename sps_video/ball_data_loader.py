import os
from os import path
import sys
import random

import torch
from PIL import Image
from torchvision import transforms
from shared import *

temp_dir = path.abspath(path.join(
    path.dirname(__file__), '..', 
))
sys.path.append(temp_dir)
assert sys.path.pop(-1) == temp_dir



def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def load_a_img_seq_with_position_from_disk(seq_path):
    img_path_list = os.listdir(seq_path)
    img_path_list.sort(key=name_in_int)
    img_list = []
    position_list = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in img_path_list:
        img = Image.open(seq_path + '/' + i).convert('RGB')
        img_tensor = transform(img)
        img_list.append(img_tensor)
        position = [torch.tensor(float(num), dtype=torch.float) for num in i.split('[')[1].split(']')[0].split(',')]
        position_list.append(torch.stack(position, dim=0))
    return torch.stack(img_list, dim=0), torch.stack(position_list, dim=0)


def name_in_int(elem):
    return int(elem.split(".")[0])


def load_a_img_seq_from_disk(seq_path):
    img_path_list = os.listdir(seq_path)
    img_path_list.sort(key=name_in_int)
    img_list = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in img_path_list:
        img = Image.open(seq_path + '/' + i).convert('RGB')
        img_tensor = transform(img)
        img_list.append(img_tensor)
    return torch.stack(img_list, dim=0)


class BallDataLoader:
    def __init__(self, data_path='dataset/120_80_0.2_20/', is_load_all_data_dict=False):
        self.data_path = data_path
        if not os.path.isdir(data_path):
            print('cannot find dataset:', data_path)
            print('we are now at', os.getcwd(), flush=True)
            raise FileNotFoundError
        self.f_list = os.listdir(data_path)  # 返回文件名
        random.shuffle(self.f_list)
        self.curr_load_idx = 0
        self.epoch_num = 0
        self.is_load_all_data_dict = is_load_all_data_dict
        if is_load_all_data_dict:
            self.all_data_dict = {}
            self.load_all_data_dict()

    def load_all_data_dict(self):
        data_num = len(self.f_list)
        print(f"============loading {data_num} data=============")
        for i in range(0, data_num):
            seq_path = self.data_path + self.f_list[i]
            data_tuple = load_a_img_seq_with_position_from_disk(seq_path)
            self.all_data_dict[seq_path] = data_tuple
            if i % 20 == 0:
                print(f'Process: {i} / {data_num}, {int(i/data_num*100)}%')
        print("============loading finished=============")

    def load_a_img_seq(self, seq_path):
        if self.is_load_all_data_dict:
            return self.all_data_dict[seq_path][0]
        else:
            return load_a_img_seq_from_disk(seq_path)

    def load_a_img_seq_with_position(self, seq_path):
        if self.is_load_all_data_dict:
            return self.all_data_dict[seq_path]
        else:
            return load_a_img_seq_with_position_from_disk(seq_path)

    def get_iter_num_of_an_epoch(self, batch_size):
        return int(len(self.f_list)/batch_size)

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def get_epoch_num_by_iter_and_batch_size(self, iter_num, batch_size):
        return int(iter_num * batch_size / len(self.f_list))

    def load_a_batch_from_an_epoch(self, batch_size):
        batch = []
        begin = self.curr_load_idx
        end = self.curr_load_idx + batch_size
        if end > len(self.f_list):
            random.shuffle(self.f_list)
            self.curr_load_idx = 0
            self.epoch_num += 1
            return self.load_a_batch_from_an_epoch(batch_size)
        for i in range(begin, end):
            seq_path = self.data_path + self.f_list[i]
            batch.append(self.load_a_img_seq(seq_path))
        self.curr_load_idx = end
        return torch.stack(batch, dim=0), self.epoch_num, round(self.curr_load_idx/len(self.f_list), 3)

    def load_a_batch_of_random_img_seq(self, batch_size):
        seq_list = random.sample(self.f_list, batch_size)
        batch = []
        for i in seq_list:
            seq_path = self.data_path + i
            batch.append(self.load_a_img_seq(seq_path))
        return torch.stack(batch, dim=0)

    def load_a_batch_of_fixed_img_seq(self, batch_size):
        batch = []
        self.f_list.sort(key=lambda x: float(x))
        for i in range(batch_size):
            seq_path = self.data_path + self.f_list[i]
            batch.append(self.load_a_img_seq(seq_path))
        return torch.stack(batch, dim=0)

    def load_a_batch_of_img_seq_with_position_from_a_epoch(self, batch_size):
        begin = self.curr_load_idx
        end = self.curr_load_idx + batch_size
        if end > len(self.f_list):
            random.shuffle(self.f_list)
            self.curr_load_idx = 0
            self.epoch_num += 1
            return self.load_a_batch_of_img_seq_with_position_from_a_epoch(batch_size)

        img_batch = []
        position_batch = []
        for i in range(begin, end):
            seq_path = self.data_path + self.f_list[i]
            img, position = self.load_a_img_seq_with_position(seq_path)
            img_batch.append(img)
            position_batch.append(position)
        return torch.stack(img_batch, dim=0).to(DEVICE), torch.stack(position_batch, dim=0).to(DEVICE)

    def load_a_batch_of_random_img_seq_with_position(self, batch_size):
        seq_list = random.sample(self.f_list, batch_size)
        img_batch = []
        position_batch = []
        for i in seq_list:
            seq_path = self.data_path + i
            img, position = self.load_a_img_seq_with_position(seq_path)
            img_batch.append(img)
            position_batch.append(position)
        return torch.stack(img_batch, dim=0).to(DEVICE), torch.stack(position_batch, dim=0).to(DEVICE)

    def IterWithPosition(self, batch_size):
        assert len(self.f_list) % batch_size == 0
        for i in range(0, len(self.f_list), batch_size):
            f_list = self.f_list[i : i + batch_size]
            img_batch = []
            position_batch = []
            for filename in f_list:
                seq_path = os.path.join(self.data_path, filename)
                img, position = self.load_a_img_seq_with_position(seq_path)
                img_batch.append(img)
                position_batch.append(position)
            yield (
                torch.stack(img_batch, dim=0), 
                torch.stack(position_batch, dim=0), 
            )

    def data_checker(self):
        for i in self.f_list:
            sub_folder = os.path.join(self.data_path, str(i))
            img_seq = os.listdir(sub_folder)
            if len(img_seq) < 20:
                print(i)

    def delete_traj_with_no_ball_imgs(self, is_delete=False):
        transform = transforms.Compose([transforms.ToTensor()])
        bad_traj_num = 0
        for i in self.f_list:
            sub_folder = os.path.join(self.data_path, str(i))
            img_seq = os.listdir(sub_folder)
            for j in img_seq:
                img = Image.open(sub_folder + '/' + j).convert('RGB')
                img_tensor = transform(img)
                if img_tensor[0].max().item() < 0.5 \
                        and img_tensor[1].max().item() < 0.5 \
                        and img_tensor[2].max().item() < 0.5:
                    bad_traj_num += 1
                    print(sub_folder)
                    if is_delete:
                        del_file(sub_folder)
                        os.removedirs(sub_folder)
                    break
        print(f'{bad_traj_num} trajectories has been deleted!')



if __name__ == "__main__":
    # eval_data_path = 'dataset/32_32_0.2_20_3_init_points/'
    # dc = BallDataLoader(eval_data_path)
    # img, position = dc.load_a_batch_of_random_img_seq_with_position(8)
    # dc.data_checker()

    # img_data_path = 'dataset/ball_test/1635404761.040899/17.[-2.635, 0.011, 11.721].png'
    # img = Image.open(img_data_path).convert('RGB')
    # transform = transforms.Compose([transforms.ToTensor()])
    # img_tensor = transform(img)
    # img_np = img_tensor.numpy()
    # print(img_tensor[1].max().item())

    check_path = 'dataset/same_position_diff_color_v2.0'
    dc = BallDataLoader(check_path)
    dc.data_checker()
    dc.delete_traj_with_no_ball_imgs(is_delete=True)




