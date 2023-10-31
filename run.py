import os
from shutil import copyfile
import argparse
import yaml
from yacs.config import CfgNode
from train import train_iter0
from test import test_iter0
from utils.logger import Logger

def str2bool(v):
    if v == "True":
        return True
    else:
        return False

def run(args):
    cfg = yaml.load(open(args.config,'r'), Loader=yaml.FullLoader)
    cfg = CfgNode(cfg)

    os.makedirs(cfg.workspace,exist_ok=True)
    os.makedirs(os.path.join(cfg.workspace, cfg.results_val), exist_ok=True)
    os.makedirs(os.path.join(cfg.workspace, cfg.results_test), exist_ok=True)
    if args.train_pass == True:
        copyfile(args.config, os.path.join(cfg.workspace, os.path.basename(args.config)))
    logger =Logger(cfg)
    logger.info(cfg)
    if args.train_pass == True:
        logger.info("Starting training iter0 pass....")
        train_iter0(cfg, logger)

    if args.test_pass == True:
        logger.info("Starting testing iter0 pass....")
        test_iter0(cfg, logger)
        
    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser("CellSeg training argument parser.")
    parser.add_argument('--config', default='/home/zby/WSCellseg/configs/debug.yaml', type=str)
    parser.add_argument("--train_pass", default=False, type=str2bool)
    parser.add_argument("--test_pass", default=False, type=str2bool)
    args = parser.parse_args()
    run(args)