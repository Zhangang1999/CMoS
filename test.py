
import os
import argparse
import logging
import sys

from models.mos.mos_builder import MOS
from datasets.dataset_builder import DATASETS
from datasets.loaders.loader_builder import LOADERS
from metrics.metric_builder import METRICS

from managers import FileManager

from utils.metric_utils import calc_metric, format_metric_msg
from utils.instantiate import instantiate_from_args

def load_and_print_configs():
    #TODO: load configs here.
    cfg = None

    msg = '-'*25 + 'Options' + '-'*25 + '\n'
    for k, v in sorted(vars(cfg).items()):
        msg += f'{k:>25}: {v:<25}\n'
    msg += '-'*25 + ' End ' + '-'*25 + '\n'
    print(msg)

    return cfg

def calc_print_and_save_results(metric_dict, metric_classes, metric_labels, file):
    metrics = calc_metric(metric_dict, metric_classes)
    metric_msg = format_metric_msg(metrics, metric_labels, metric_classes)
    file.log_log(
        session='test',
        data={'metric_msg': metric_msg}
    )

def main(device, work_dir, display=False):
    cfg = load_and_print_configs()
    
    file_manager = FileManager(work_dir)
    for field in file_manager.IN_MANAGEMENT:
        os.makedirs(getattr(file_manager.path, field), exist_ok=True)

    test_dataset = instantiate_from_args(cfg.data.test, DATASETS, dict(pipeline=cfg.test_pipeline))
    data_loader = instantiate_from_args(cfg.data.loader, LOADERS, dict(dataset=test_dataset, shuffle=False))
            
    model = instantiate_from_args(cfg.model, MOS, dict(device=device))
    metrics = [instantiate_from_args(metric, METRICS) for metric in cfg.metrics]

    model.eval()
    
    results = {}
    for idx, datum in enumerate(data_loader):
        outputs = model.test_step(*datum)

        for _, metric_obj in metrics:
            metric_obj.__call__(outputs['results'], outputs['gts'])

        if display:
            #TODO: display the results.
            pass
    
    calc_print_and_save_results(metrics)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | (%(filename)s:%(lineno)d | %(level_name)s | %(messages)s)")

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda', help='')
    parser.add_argument("--work_dir", type=str, default='./', help='')
    parser.add_argument("--display", action='store_true', default=False, help='')

    args = parser.parse_args()
    logging.info(args)
    main(args.device, args.workdir, args.display)
