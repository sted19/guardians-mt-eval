from argparse import ArgumentParser
import pandas as pd
import json
from collections import defaultdict

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
        "esa": ['esa-human1', 'esa-human2'],
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

def load_mqm_2025(raw_data_file: str, year: str, protocol: str) -> pd.DataFrame:

    """Preprocesses the raw data file containing the MQM 2025 annotations. 

    This function takes sources from the esa data file, and scores from the mqm data on mt-metrics-eval.

    Args:
        raw_data_file (str): Here, we use the same ESA 2025 data file.
        year (str): The year of the MQM annotations to load. Allowed values are in the format "wmt24", "wmt25".

    Returns:
        A data frame containing the loaded esa 2025 annotations. The dataframe has the following columns:
            - `lp`: The language pair in the format "src-tgt".
            - `src`: The source sentence.
            - `mt`: The hypothesis translation.
            - `ref`: The reference translation.
            - `score`: The human score assigned to the hypothesis sentence by the annotators.
            - `system`: The name of the system that produced the hypothesis translation.
            - `annotators`: The id of the annotator that assigned the score to the hypothesis sentence.
            - `domain`: The domain of the source sentence (e.g., "news", "social media", etc.).
    """
    lps = []
    srcs = []
    mts = []
    refs = []
    scores = []
    systems = []
    annotators = []
    domains = []

    with open(raw_data_file, "r") as f:
        data = [json.loads(line) for line in f]

    # convert data into a format similar to mt-metrics-eval
    new_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, item in enumerate(data):
        src = item["src_text"]
        lp, domain, doc_id, seg_id = item['doc_id'].split("_#_")

        tgt_texts = item["tgt_text"]

        for sys_name, mt in tgt_texts.items():

            item_scores = item['scores'].get(sys_name, None)
            if item_scores is None:
                continue
            
            # some items are scalar scores instead of dicts and I don't know what they are
            item_scores = [item for item in item_scores if isinstance(item, dict)]
            assert(len(item_scores)<= 2), f"Expected at most 2 scores per system, but found {len(item_scores)} for system {sys_name} in item {i}."
            for annotator_idx in range(2):

                score_dict = item_scores[annotator_idx] if annotator_idx < len(item_scores) else None

                if score_dict is None:
                    new_data[lp][sys_name][str(annotator_idx+1)].append(None)
                else:

                    score = score_dict['score']
                    annotator = score_dict['annotator']

                    new_data[lp][sys_name][str(annotator_idx+1)].append({
                        "src": src,
                        "mt": mt,
                        "score": score,
                        "domain": domain,
                        "annotator": annotator,
                    })

    year_lps = year_to_lps[year][protocol]
    for lp in year_lps:
        evs = mt_metrics_eval.data.EvalSet(year, lp)

        _scores = evs._scores
        gold_scorers = year_and_protocol_to_scorers[year][protocol]
        for scorer_idx, gold_scorer in enumerate(gold_scorers):
            _gold_scores = _scores['seg'][gold_scorer]

            for sys_name, sys_scores in _gold_scores.items():
                sys_outputs = evs._sys_outputs[sys_name]
                ref_name = year_and_protocol_to_ref[year][protocol]
                ref_outputs = evs._sys_outputs.get(ref_name, None)

                if not ref_outputs:
                    print(f"WARNING: Reference outputs not found for year {year} and protocol mqm. Proceeding with ref=None.")
                    ref_outputs = [None] * len(sys_outputs)

                raw_scorers = list(new_data[lp][sys_name].keys())
                
                raw_items = new_data[lp][sys_name][str(scorer_idx+1)]

                assert(len(sys_outputs) == len(raw_items)), f"Length mismatch between sys outputs ({len(sys_outputs)}) and raw items ({len(raw_items)}) for system {sys_name} in lp {lp} and scorer {gold_scorer}."

                for i, score in enumerate(sys_scores):
                    
                    src = raw_items[i]['src']
                    mt = raw_items[i]['mt']
                    domain = raw_items[i]['domain']

                    if domain=="canary":
                        continue

                    lps.append(lp)
                    srcs.append(src)
                    mts.append(mt)
                    refs.append(None)
                    scores.append(score)
                    systems.append(sys_name)
                    annotators.append(gold_scorer)
                    domains.append(domain)

                    breakpoint()
    
    # Ensure there are no missing values
    assert all(len(lst) == len(lps) for lst in [srcs, mts, refs, scores, systems, annotators, domains]), "All columns must have the same length."

    # Ensure there are no None values, and that types are consistent within columns
    for col, lst in zip(["src", "mt", "ref", "score", "system", "annotators", "domain"], [srcs, mts, refs, scores, systems, annotators, domains]):
        if col != "ref":  # ref can be None if not found in the data
            assert all(x is not None for x in lst), f"Column '{col}' contains None values."
        
        if col == "score":
            assert all(isinstance(x, (int, float)) for x in lst), f"Column '{col}' must contain numeric values."
        elif col != "ref":
            assert all(isinstance(x, str) for x in lst), f"Column '{col}' must contain string values."

    df = pd.DataFrame({
        "lp": lps,
        "src": srcs,
        "mt": mts,
        "ref": refs,
        "score": scores,
        "system": systems,
        "annotators": annotators,
        "domain": domains,
        "year": [year for _ in range(len(lps))],
    })

    # Strip stray \r that breaks CSV round-tripping (\n is handled by quoting)
    for col in ["src", "mt", "ref"]:
        df[col] = df[col].str.replace("\r", "", regex=False)

    return df



def load_esa_2025(raw_data_file: str, year: str, protocol: str) -> pd.DataFrame:
    """Preprocesses the raw data file containing the ESA 2025 annotations.

    Args:
        raw_data_file (str): Path to the input file containing the raw data to preprocess.

    Returns:
        A data frame containing the loaded esa 2025 annotations. The dataframe has the following columns:
            - `lp`: The language pair in the format "src-tgt".
            - `src`: The source sentence.
            - `mt`: The hypothesis translation.
            - `ref`: The reference translation.
            - `score`: The human score assigned to the hypothesis sentence by the annotators.
            - `system`: The name of the system that produced the hypothesis translation.
            - `annotators`: The id of the annotator that assigned the score to the hypothesis sentence.
            - `domain`: The domain of the source sentence (e.g., "news", "social media", etc.).
    """
    
    with open(raw_data_file, "r") as f:
        data = [json.loads(line) for line in f]

    lps = []
    srcs = []
    mts = []
    refs = []
    scores = []
    systems = []
    annotators = []
    domains = []

    for i, item in enumerate(data):
        src = item["src_text"]
        ref = None
        lp, domain, doc_id, seg_id = item['doc_id'].split("_#_")

        tgt_texts = item["tgt_text"]

        for sys_name, mt in tgt_texts.items():
            if sys_name.startswith("ref"):
                assert ref is None, f"Multiple reference translations found for item {i}."
                ref = mt

        for sys_name, mt in tgt_texts.items():

            item_scores = item['scores'].get(sys_name, None)
            if item_scores is None:
                continue

            for score_dict in item_scores:

                if not isinstance(score_dict, dict):
                    continue

                score = score_dict['score']
                annotator = score_dict['annotator']
                errors = score_dict.get('errors', []) # unused 

                # For each score, we build an entry in the dataframe
                lps.append(lp)
                srcs.append(src)
                mts.append(mt)
                refs.append(None)
                scores.append(score)
                systems.append(sys_name)
                annotators.append(annotator)
                domains.append(domain)

    # Ensure there are no missing values
    assert all(len(lst) == len(lps) for lst in [srcs, mts, refs, scores, systems, annotators, domains]), "All columns must have the same length."

    # Ensure there are no None values, and that types are consistent within columns
    for col, lst in zip(["src", "mt", "ref", "score", "system", "annotators", "domain"], [srcs, mts, refs, scores, systems, annotators, domains]):
        if col != "ref":  # ref can be None if not found in the data
            assert all(x is not None for x in lst), f"Column '{col}' contains None values."
        
        if col == "score":
            assert all(isinstance(x, (int, float)) for x in lst), f"Column '{col}' must contain numeric values."
        elif col != "ref":
            assert all(isinstance(x, str) for x in lst), f"Column '{col}' must contain string values."

    df = pd.DataFrame({
        "lp": lps,
        "src": srcs,
        "mt": mts,
        "ref": refs,
        "score": scores,
        "system": systems,
        "annotators": annotators,
        "domain": domains,
        "year": ["wmt25" for _ in range(len(lps))],
    })

    # Strip stray \r that breaks CSV round-tripping (\n is handled by quoting)
    for col in ["src", "mt", "ref"]:
        df[col] = df[col].str.replace("\r", "", regex=False)

    return df
        

select_loader = {
    "wmt25": {
        "esa": load_esa_2025,
        "mqm": load_mqm_2025,
    },
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--raw-data-file",
        type=str,
        required=True,
        help="Path to the input file containing the data to preprocess.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output file where the preprocessed data will be saved.",
    )
    parser.add_argument(
        "--annotation-protocol",
        type=str,
        default="esa",
        help="Name of the annotation protocol used to annotate the raw data file.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="wmt24",
        help="Year in which the raw annotations have been collected.",
    )
    parser.add_argument(
        "--znorm-per-sys",
        action="store_true",
        help="Whether to z-normalize scores per system.",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Preprocessing raw data file: {args.raw_data_file}")
    print(f"Output file: {args.output_file}")
    print(f"Annotation protocol: {args.annotation_protocol}")
    print(f"Year: {args.year}")

    loading_function = select_loader.get(args.year, {}).get(args.annotation_protocol)
    if loading_function:
        df = loading_function(args.raw_data_file, args.year, args.annotation_protocol)
    else:
        raise ValueError(f"Loading function not found for year {args.year} and annotation protocol {args.annotation_protocol}")

    if args.znorm_per_sys:
        # copy score column to avoid modifying the original scores in the dataframe
        df["raw"] = df["score"].copy()
        df["z_norm_without_domain"] = df.groupby(["year", "lp", "system"])["raw"].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
        df["score"] = df["z_norm_without_domain"]

    df.to_csv(args.output_file, index=False)
    print(f"Preprocessed data saved to: {args.output_file}")

if __name__ == "__main__":
    main()