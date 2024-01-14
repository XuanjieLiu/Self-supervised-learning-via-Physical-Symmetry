import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')


def load_config(exp_name):
    exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
    os.chdir(exp_path)
    sys.path.append(exp_path)
    t_config = __import__('train_config')
    sys.path.pop()
    return t_config.CONFIG
