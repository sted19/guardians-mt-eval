"""
Defines and operates on tasks in the style of mt-metrics-eval (https://github.com/google-research/mt-metrics-eval).
"""
import argparse
from collections import defaultdict
import dataclasses
from typing import List, Literal, Dict, Optional, Iterable, Tuple, Iterator
import yaml
import json
from pathlib import Path
from scipy import stats

import sentinel_metric

import mt_metrics_eval
import mt_metrics_eval.data

wmt24_esa_lps = ['cs-uk', 'en-cs', 'en-hi', 'en-is', 'en-ja', 'en-ru', 'en-uk', 'en-zh']
wmt24_mqm_lps = ['en-de', 'en-es', 'ja-zh']
wmt25_esa_lps = ['cs-de_DE', 'cs-uk_UA', 'en-ar_EG', 'en-bho_IN', 'en-zh_CN', 'en-cs_CZ', 'en-et_EE', 'en-is_IS', 'en-it_IT', 'en-ja_JP', 'en-mas_KE', 'en-ru_RU', 'en-sr_Cyrl_RS', 'en-uk_UA']
wmt25_mqm_lps = ['en-ko_KR', 'ja-zh_CN']


year_to_lps = {
    "wmt24": {
        "esa": wmt24_esa_lps,
        "mqm": wmt24_mqm_lps,
    },
    "wmt25": {
        "esa": wmt25_esa_lps,
        "mqm": wmt25_mqm_lps,
    }
}


year_and_protocol_to_scorers = {
    "wmt25": { 
        "esa": ['esa-human-1', 'esa-human-2'],
        "mqm": ['mqm'],
    },
    "wmt24": {
        "esa": ['esa'],
        "mqm": ['mqm'],
    }
}

year_and_protocol_to_ref = {
    "wmt25": {
        "esa": "refA",
        "mqm": "refA",
    },
    "wmt24": {
        "esa": "refA",
        "mqm": "refA",
    }
}


wmt24_domains = {"news", "social", "literary", "speech"}


def read_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-to-evaluate",
        type=str,
        default="configs/metrics_to_evaluate.yaml",
        help="Path to the input data file.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="esa",
        help="Protocol to use for the evaluation. Allowed values: 'esa', 'mqm'.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="wmt24",
        help="Year of the WMT test set to use for the evaluation. Examples: 'wmt22', 'wmt23', 'wmt24'.",
    )
    return parser.parse_args()



def load_metric_scores(path: str) -> Dict[str, Dict[str, List[Optional[float]]]] | None:
    """
    Load metric scores from a JSONL file.
    """

    _path: Path = Path(path)
    if not _path.exists():
        return None

    with open(_path, "r") as f:
        scores = json.load(f)
    
    return scores



def load_sentinel_metric(name: str, path: str) -> sentinel_metric.models.sentinel_regression.sentinel_regression_metric.SentinelRegressionMetric:

    print(f"Loading sentinel metric {name} from path {path}...")

    try:
        sentinel = sentinel_metric.load_from_checkpoint(
            path,
            strict=True,
            class_identifier="sentinel_regression_metric",
        )
    except Exception as e:
        metric_model_path = sentinel_metric.download_model(
            name
        )
        sentinel = sentinel_metric.load_from_checkpoint(
            metric_model_path,
            strict=True,
            class_identifier="sentinel_regression_metric",
        )

    return sentinel

def compute_metric_scores(
        year: str,
        protocol: str,
        sentinel_name: str, 
        sentinel_path: str, 
        output_path: str,
        batch_size: int = 16,
        gpus: int = 1
    ) -> Dict[str, List[float]]:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentinel = load_sentinel_metric(sentinel_name, sentinel_path)

    lps = year_to_lps[year][protocol]
    source_langs = [lp.split('-')[0] for lp in lps]
    source_langs = list(set(source_langs))

    print("Computing scores for the following language pairs:", lps)

    src_lang2metric_scores = dict()
    for src_lang in source_langs:
        lp = next(lp for lp in lps if lp.split('-')[0] == src_lang)
        evs = mt_metrics_eval.data.EvalSet(args.year, lp)

        srcs = evs.src

        scores: sentinel_metric.models.utils.Prediction = sentinel.predict(
            [{"src": src} for src in srcs],
            batch_size=batch_size,
            gpus=gpus,
        ).scores

        src_lang2metric_scores[src_lang] = scores

    with open(output_path, "w") as f:
        json.dump(src_lang2metric_scores, f, indent=2)

    return src_lang2metric_scores

    
def compute_metrics_correlation(
        year: str, 
        protocol: str,
        metric2src_lang2scores: Dict[str, Dict[str, List[Optional[float]]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
    

    lps = year_to_lps[year][protocol]

    lp2metric2sys2corrs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for lp in lps:
        evs = mt_metrics_eval.data.EvalSet(year, lp)
        gold_scorers = year_and_protocol_to_scorers[year][protocol]

        assert(len(gold_scorers) == 1), f"Averaging across gold scorers not implemented yet, but found {len(gold_scorers)} gold scorers for year {year} and protocol {protocol}."
        gold_scorer = gold_scorers[0]

        for metric_name, src_lang2scores in metric2src_lang2scores.items():
            print(f"Computing correlation between {metric_name} and {gold_scorer}...")

            metric_scores = src_lang2scores[lp.split('-')[0]]

            gold_scores = evs._scores['seg'][gold_scorer]

            for sys_name, sys_gold_scores in gold_scores.items():

                assert(len(sys_gold_scores) == len(metric_scores)), f"Length mismatch for system {sys_name}: {len(sys_gold_scores)} gold scores vs {len(metric_scores)} metric scores."

                # Filter out None values
                paired_scores = [(gs, ms) for gs, ms in zip(sys_gold_scores, metric_scores) if gs is not None and ms is not None]

                if len(paired_scores) == 0:
                    print(f"No valid scores for system {sys_name}, skipping correlation.")
                    continue

                gold_scores_filtered, metric_scores_filtered = zip(*paired_scores)

                # compute kendall tau b correlation
                results = stats.kendalltau(gold_scores_filtered, metric_scores_filtered, variant='b')
                kendall_tau_b_corr, kendall_tau_b_pval = results.correlation, results.pvalue
                lp2metric2sys2corrs[lp][metric_name][sys_name] = kendall_tau_b_corr
    
    return lp2metric2sys2corrs



def compute_dec(lp2metric2sys2corrs: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    for lp, metric2sys2corr in lp2metric2sys2corrs.items():

        for metric_name, sys2corr in metric2sys2corr.items():
            metric_sys_avg = sum(sys2corr.values()) / len(sys2corr)

            lp2metric2sys2corrs[lp][metric_name]['avg'] = metric_sys_avg
        
    metric_names = set(lp2metric2sys2corrs[next(iter(lp2metric2sys2corrs))].keys())
    metric2dec = dict()
    for metric_name in metric_names:
        metric_lp_avg = sum(lp2metric2sys2corrs[lp][metric_name]['avg'] for lp in lp2metric2sys2corrs) / len(lp2metric2sys2corrs)

        metric2dec[metric_name] = metric_lp_avg
    
    return metric2dec
    



def main(args: argparse.Namespace) -> None:
    with open(args.metrics_to_evaluate, "r") as f:
        config = yaml.safe_load(f)

    metric2src_lang2scores = dict()
    for metric in config["metrics"]:
        print(metric["name"], metric["path"], metric["esa-scores"], metric["mqm-scores"])

        scores_key = f"{args.protocol}-scores"

        scores = load_metric_scores(metric[scores_key])

        if scores is None:
            print(f"Scores for metric {metric['name']} not found at path {metric[scores_key]}, computing scores...")
            scores = compute_metric_scores(args.year, args.protocol, metric["name"], metric["path"], metric[scores_key])
        
        metric2src_lang2scores[metric["name"]] = scores


    lp2metric2sys2corr = compute_metrics_correlation(args.year, args.protocol, metric2src_lang2scores)

    metric2dec = compute_dec(lp2metric2sys2corr)
    
    # visualize dec scores nicely
    print("DEC scores:")
    for metric_name, dec_score in metric2dec.items():
        print(f"{metric_name}: {dec_score:.4f}")


if __name__ == "__main__":
    args = read_args()
    main(args)