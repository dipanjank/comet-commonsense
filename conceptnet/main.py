import logging
import os.path
import random

import torch

import conceptnet.data.config as cfg
import conceptnet.data.data_utils as data
import conceptnet.train.conceptnet_train as train
from conceptnet import gpt as models, utils as utils
from conceptnet.data.data_loader import GenerationDataLoader, conceptnet_relations
from conceptnet.data.data_utils import TextEncoder
from conceptnet.train.opt import OpenAIAdam


def main():
    # Loads the correct configuration file
    config_file = os.path.join(os.path.basename(__file__), 'config.json')
    logging.info(f'Reading config_file {config_file}')

    # Read config file to option
    config = cfg.read_config(cfg.load_config(config_file))
    opt, meta = cfg.get_parameters(config)

    # Set the random seeds
    torch.manual_seed(opt.train.static.seed)
    random.seed(opt.train.static.seed)
    if config.gpu_mode:
        torch.cuda.manual_seed_all(opt.train.static.seed)

    opt.train.dynamic.epoch = 0

    logging.info("Loading Data")

    # Initialize path to pre-set data loader
    path = "data/conceptnet/processed/{}/{}.pickle".format(
        opt.exp, utils.make_name_string(opt.data))

    # Make data loader
    data_loader = GenerationDataLoader(opt)
    loaded = data_loader.load_data(path)
    logging.info(data_loader.sequences["train"]["total"].size(0))
    data_loader.opt = opt
    data_loader.batch_size = opt.train.dynamic.bs

    logging.info("Done.")

    text_encoder = TextEncoder(config.encoder_path, config.bpe_path)

    special = [data.start_token, data.end_token]
    special += ["<{}>".format(cat) for cat in conceptnet_relations]

    if loaded:
        text_encoder.encoder = data_loader.vocab_encoder
        text_encoder.decoder = data_loader.vocab_decoder
    else:
        raise NotImplementedError
        # for special_token in special:
        #     text_encoder.decoder[len(encoder)] = special_token
        #     text_encoder.encoder[special_token] = len(encoder)
        # data_loader.make_tensors(text_encoder, special)

    # Set max size of different parts of relation
    context_size_e1 = data_loader.max_e1
    context_size_e2 = data_loader.max_e2
    context_size_r = data_loader.max_r

    opt.data.maxr = context_size_r

    n_special = len(special)
    n_ctx = context_size_e1 + context_size_r + context_size_e2
    n_vocab = len(text_encoder.encoder) + n_ctx

    logging.info(data_loader.__dict__.keys())
    opt.net.vSize = n_vocab

    # Build Model
    logging.info("Building Model")

    model = models.make_model(
        opt, n_vocab, n_ctx, n_special,
        load=(opt.net.init=="pt"))

    logging.info("Done.")

    logging.info("Files will be logged at: {}".format(
        utils.make_name(opt, prefix="results/losses/",
                        is_dir=True, eval_=True)))

    data_loader.reset_offsets("train", keys=["total"])

    data.set_max_sizes(data_loader)

    # Push to GPU
    if config.gpu_mode:
        logging.info("Pushing to GPU: {}".format(config.gpu_index))
        cfg.device = config.gpu_index
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        if config.multigpu:
            model = models.multi_gpu(
                model, config.gpu_indices).cuda()
        else:
            model.cuda(cfg.device)
        logging.info("Done.")

    logging.info("Training")

    optimizer = OpenAIAdam(model.parameters(),
                           lr=opt.train.dynamic.lr,
                           schedule=opt.train.static.lrsched,
                           warmup=opt.train.static.lrwarm,
                           t_total=meta.iterations,
                           b1=opt.train.static.b1,
                           b2=opt.train.static.b2,
                           e=opt.train.static.e,
                           l2=opt.train.static.l2,
                           vector_l2=opt.train.static.vl2,
                           max_grad_norm=opt.train.static.clip)

    trainer = train.ConceptNetGenerationIteratorTrainer(
        opt, meta, data_loader, model, optimizer)

    logging.info(data_loader.sequences["dev"]["total"].max())
    trainer.set_generator(opt, model, data_loader)
    trainer.set_evaluator(opt, model, data_loader)

    trainer.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] - %(message)s')
    main()

