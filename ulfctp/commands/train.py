import os
import re
import argparse
import yaml

import torch

from stog.utils import logging
from stog.utils.params import Params, remove_pretrained_embedding_params
from stog.data.vocabulary import Vocabulary
from stog.utils import environment
from stog.utils.checks import ConfigurationError
from stog.utils.archival import CONFIG_NAME, _DEFAULT_WEIGHTS, archive_model
from stog.commands.evaluate import evaluate
from stog.metrics import dump_metrics
from ulfctp.training.trainer import Trainer
from ulfctp import models as Models
from ulfctp.data.dataset_builder import dataset_from_params, iterator_from_params
from ulfctp.utils.environment import set_seed

logger = logging.init_logger()


def create_serialization_dir(params: Params, force_params: bool=False) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.
    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    force_params: ``bool''
        Whether to use the current params even if they don't match those of the
        restored model. This places the onus on the caller to ensure the new parameters
        are compatible with the model being restored. For example, decoding-specific
        parameters may differ for the same model.
    """
    serialization_dir = params['environment']['serialization_dir']
    recover = params['environment']['recover']
    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        #if not recover:
        #    raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
        #                             f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            if not force_params:
                loaded_params = Params.from_file(recovered_config_file)
                if params != loaded_params:
                    raise ConfigurationError("Training configuration does not match the configuration we're "
                                             "recovering from.")
            # In the recover mode, we don't need to reload the pre-trained embeddings.
            remove_pretrained_embedding_params(params)
    else:
        if recover:
            raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
                                     "does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)
        params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

def multistage_train_model(params: Params, decode_only=False, restore_best_model=False,
        force_params=False, tuning_off=False, fine_tune_k=-1, eval_test_k=-1):
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results.
    Best n models will be saved first, then with larger beam size and type checking, the best 
    model among these n models are saved to best.th
    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    decode_only: ``bool''
        Whether to only perform the decoding stage.
    restore_best_model: ``bool''
        Whether to restore the model state to the best previously performing model,
        rather than the default behavior of restoring to the most recent epoch.
    force_params: ``bool''
        Whether to use the current params even if they don't match those of the
        restored model. This places the onus on the caller to ensure the new parameters
        are compatible with the model being restored. For example, decoding-specific
        parameters may differ for the same model.
    fine_tune_k: ``int''
        Specify which of the best n models is used to do the fine tuning. By default
        -1 indicates that all models are used for fine tuning. Otherwise, the best kth
        model will be used for fine tuning. Hence, k <= best_n
    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights after training.
    """

    # Set up the environment.
    environment_params = params['environment']
    create_serialization_dir(params, force_params)
    environment.prepare_global_logging(environment_params)
    set_seed(environment_params)
    environment.check_for_gpu(environment_params)
    if environment_params['gpu']:
        device = torch.device('cuda:{}'.format(environment_params['cuda_device']))
        environment.occupy_gpu(device)
    else:
        device = torch.device('cpu')
    params['trainer']['device'] = device

    # Load data.
    data_params = params['data']
    dataset = dataset_from_params(data_params)
    train_data = dataset['train']
    dev_data = dataset.get('dev')
    test_data = dataset.get('test')

    # Vocabulary and iterator are created here.
    vocab_params = params.get('vocab', {})
    vocab = Vocabulary.from_instances(instances=train_data, **vocab_params)
    # Initializing the model can have side effect of expanding the vocabulary
    vocab.save_to_files(os.path.join(environment_params['serialization_dir'], "vocabulary"))

    train_iterator, dev_iterater, test_iterater = iterator_from_params(vocab, data_params['iterator'])

    # Build the model with type checking option off
    model_params = params['model']
    # If include fine tuning or one want to skip fine tuning section, 
    # turn off type_checking regardless of the params file
    tuning_off = (not 'fine-tuning' in params) or (tuning_off)
    if not tuning_off:
        model_params['transition_system']['type_checking_method'] = 'none'
    model = getattr(Models, model_params['model_type']).from_params(vocab, model_params)
    logger.info(model)

    # Train
    trainer_params = params['trainer']
    no_grad_regexes = trainer_params['no_grad']
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
        environment.get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    trainer = Trainer.from_params(model, train_data, dev_data, train_iterator, dev_iterater, trainer_params)

    serialization_dir = trainer_params['serialization_dir']
    try:
        if tuning_off:
            metrics = trainer.train(decode_only, restore_best_model)
        else:
            # Obtaining fine-tuning parameters
            tuning_params = params['fine-tuning']
            best_n = int(tuning_params['best_n'])
            # Set up best_n argument
            metrics = trainer.train(decode_only, restore_best_model, best_n)
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logger.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir)
        raise
    
    # Can be successfully retrived even if best_n is not supplied
    best_n_models = trainer.best_n_models()
    
    # If flag on fine tuning
    if not tuning_off:
        # Obtain paths for best n models and validates against type checking
        logger.info('Selecting best one among the best %d models...'%best_n)
        # If fine_tune_epoch argument is not -1, grab only the kth model (i.e. the best kth model)
        if fine_tune_k != -1:
            logger.info('Using fine_tune_k (k = {})'.format(fine_tune_k))
            if fine_tune_k >= int(tuning_params['best_n']):
                logger.error('Invalid fine_tune_k parameters, set to 0 by default')
                fine_tune_k = 0
            # Keep only the kth model path
            best_n_models = [best_n_models[fine_tune_k]]

        # Build new model with type checking on and larger beam size
        model_params = params['model']
        model_params['transition_system']['type_checking_method'] = tuning_params['type_checking_method']
        model_params['beam_size'] = tuning_params['beam_size']
        model = getattr(Models, model_params['model_type']).from_params(vocab, model_params)
        logger.info(model)

        frozen_parameter_names, tunable_parameter_names = \
            environment.get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        trainer = Trainer.from_params(model, train_data, dev_data, train_iterator, dev_iterater, trainer_params)

        serialization_dir = trainer_params['serialization_dir']
        best_model_path, best_metric = trainer.best_model(best_n_models)
        logger.info('Best model path: %s'%best_model_path)
    else:
        # If specify the model to use
        if eval_test_k != -1:
            # Retrieve model path based on epoch number
            best_model_path = os.path.join(serialization_dir, "model_state_epoch_{}.th".format(eval_test_k))
        else:
            # If skip the fine tuning, still retrive the best model path from the training
            # If best_n is not set (in case of fine tuning off) the default best n is 1, one 
            # can still retrive best n models except now it return one path only
            # If instead one just turn off the fine tuning but best_n is actually set
            # it retrives the best model obtained during training
            best_model_path = best_n_models[0]
        
    # Now tar up results
    archive_model(serialization_dir)

    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    if not isinstance(best_model, torch.nn.DataParallel):
        best_model_state = {re.sub(r'^module\.', '', k):v for k, v in best_model_state.items()}
    best_model.load_state_dict(best_model_state)

    # Validate on test set if required
    if params['test']['evaluate_on_test']:
        # Build model with test data
        model_params['mimick_test']['data'] = params['test']['data']
        model_params['mimick_test']['prediction_basefile'] = params['test']['prediction_basefile']
        logger.info('Testing with type checking method: {} and beamsize: {}'.format(
            model_params['transition_system']['type_checking_method'],
            model_params['beam_size']))
        model = getattr(Models, model_params['model_type']).from_params(vocab, model_params)
        # Update trainer
        trainer = Trainer.from_params(model, train_data, dev_data, train_iterator, dev_iterater, trainer_params)

        #Compute test metric for best model
        test_metric = trainer.eval_test(best_model_path)

    return best_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('params', help='Parameters YAML file.')
    parser.add_argument('--decode_only', default=False, required=False, action='store_true', help='Flag to only perform the decoding stage.')
    parser.add_argument('--best_model', default=False, required=False, action='store_true', help='Flag to restore to the best performing model, rather than the model for the most recent epoch.')
    parser.add_argument('--force_params', default=False, required=False, action='store_true', help='Flag to force the use of the given parameters even if they don\'t match those of the restored model.')
    parser.add_argument('--tuning_off', default=False, required=False, action='store_true', help='Flag to skip the fine-tuning section')
    parser.add_argument('--fine_tune_k', default=-1, type=int, required=False, help='Flag to use only the kth best model to perform the fine tuning')
    parser.add_argument('--eval_test_k', default=-1, type=int, required=False, help='Flag to use the kth epoch model to do test evaluation')
    args = parser.parse_args()

    params = Params.from_file(args.params)
    logger.info(params)

    multistage_train_model(params, args.decode_only, args.best_model, args.force_params, args.tuning_off, args.fine_tune_k, args.eval_test_k)
