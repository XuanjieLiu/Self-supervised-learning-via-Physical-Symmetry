import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from trainer_symmetry import BallTrainer, is_need_train
from common_utils import load_config


exp_name = sys.argv[1]
CONFIG = load_config(exp_name)
sys.path.pop()
trainer = BallTrainer(CONFIG)
if is_need_train(CONFIG):
    trainer.train()
