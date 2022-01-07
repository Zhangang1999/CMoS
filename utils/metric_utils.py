import numpy as np

from collections import OrderedDict
from typing import Dict, List

from metrics import metric_builder


def calc_metric(metric_dict, metric_classes:List[str]) -> Dict:
    """Collecting the metrics to dict."""

    metrics = {}
    for metric, metric_obj in metric_dict.items():
        metrics.update({metric: OrderedDict()})
        metrics[metric]['all'] = 0
        resDict = metric_obj.gather([f'{metric}_{_cls}' for _cls in range(len(metric_classes))])

        for _cls in range(len(metric_classes)):
            results = resDict[f'{metric}_{_cls}']
            mean = np.mean(np.array(results).reshape(-1))
            metrics[metric][metric_classes[_cls]] = mean
        metrics[metric]['all'] = \
            sum(metrics[metric].values()) / (len(metrics[metric].values())-1)
    return metrics

def format_metric_msg(metrics, metric_labels, metric_classes):
    """format the metric messages."""

    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)

    title_msg = '\n'
    title_msg += make_row([''] + ['all'] + [x for x in metric_classes]) + '\n'
    title_msg += make_sep(len(metric_classes)+2)

    metric_msg = '\n'
    for metric in metric_labels:
        metric_msg += make_row([metric] + ['%.2f' % x if x<100 else '%.1f' \
                        % x for x in metrics[metric].values()])
        metric_msg += '\n'

    ending_msg = make_sep(len(metric_classes)+2) + '\n'
    ending_msg += '\n'

    return title_msg + metric_msg + ending_msg