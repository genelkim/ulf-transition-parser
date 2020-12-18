import torch
from stog.utils import logging
import stog.utils.environment as stog_env


logger = logging.init_logger()


def set_seed(params):
    """
    Adopted from AllenNLP:
        https://github.com/allenai/allennlp/blob/606a61abf04e3108949022ae1bcea975b2adb560/allennlp/common/util.py

    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.
    """
    stog_env.set_seed(params)

    if params['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info('CuDNN set to be deterministic. THIS WILL RUN SLOWER, but allow replicability.')
    else:
        logger.info('CuDNN not set to be deterministic. This may result in non-replicability.')

