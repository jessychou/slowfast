#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()

    print(f'args: {args}')
    cfg = load_config(args)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        # launch_job(cfg=cfg, init_method=args.init_method, func=train)
        pass

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        # launch_job(cfg=cfg, init_method=args.init_method, func=test)
        pass

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)
        pass

    # Run demo.
    if cfg.DEMO.ENABLE:
        cfg.DEMO.INPUT_VIDEO = "/Users/jessy/slowfast/slowfast/data/my_video2.mp4"
        cfg.DEMO.OUTPUT_FILE = "/Users/jessy/slowfast/slowfast/output/my_video2_result.mp4"
        demo(cfg)


if __name__ == "__main__":
    main()
