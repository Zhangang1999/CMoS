
import argparse
import logging
import sys

from models import MOS
from datasets import DATASETS
from datasets import LOADERS
from trainers import TRAINERS
from trainers import HOOKS
from trainers import OPTIMIZERS

from managers import FileManager

from utils.instantiate import instantiate_from_args

def print_and_save_configs(cfg):
    msg = '-'*25 + 'Options' + '-'*25 + '\n'
    for k, v in sorted(vars(cfg).items()):
        msg += f'{k:>25}: {v:<25}\n'
    msg += '-'*25 + ' End ' + '-'*25 + '\n'
    print(msg)

    #TODO: save configs here.

def main(cfg, device, work_dir):
    print_and_save_configs(cfg)
    
    file_manager = FileManager(work_dir)
    file_manager.makedirs(cfg.model.name)

    dataset = instantiate_from_args(cfg.data.train, DATASETS, dict(transform=cfg.transform))

    data_loaders = [
        instantiate_from_args(cfg.data.loader, LOADERS, dict(dataset=dataset, shuffle=shuffle))
            for dataset, shuffle in [(dataset.train, True), (dataset.valid, False)]
    ]
    
    model = instantiate_from_args(cfg.model, MOS, dict(device=device))
    optimizer = instantiate_from_args(cfg.optimizer, OPTIMIZERS, dict(model=model))

    trainer = instantiate_from_args(cfg.trainer, TRAINERS, 
        dict(model=model, optimizer=optimizer, file_manager=file_manager))

    for hook, priority in cfg.hook:
        trainer.register_hook(instantiate_from_args(hook, HOOKS), priority)
    trainer.run(data_loaders, cfg.workflow)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | (%(filename)s:%(lineno)d | %(level_name)s | %(messages)s)")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='')
    parser.add_argument("--device", type=str, default='cuda', help='')
    parser.add_argument("--work_dir", type=str, default='./', help='')

    args = parser.parse_args()
    logging.info(args)
    main(args.config, args.device, args.workdir)