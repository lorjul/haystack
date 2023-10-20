import numpy as np
import json
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import pandas as pd
from argparse import ArgumentParser


def health_check(arr: np.ndarray):
    assert isinstance(arr, np.ndarray)
    assert arr.shape[1] > 3
    assert not np.isnan(arr).any()
    assert (arr[:, :3].astype(int) == arr[:, :3]).all()
    np.testing.assert_almost_equal(
        arr[:, 3:].sum(-1), np.ones((len(arr),)), decimal=5, err_msg="Did you apply softmax?"
    )


def discrimination_disadvantage(labels: np.ndarray, ranks: np.ndarray, num_predicates: int):
    rs = []
    for k in range(1, num_predicates):
        if labels.sum() == 0:
            rs.append(1.0)
            raise RuntimeError()
        else:
            rs.append(recall_score(labels, ranks < k))
    return 1 - float(np.mean(rs))


def dominance_overestimation(labels: np.ndarray, ranks: np.ndarray, num_predicates: int):
    rs = []
    for k in range(1, num_predicates):
        pred = ranks < k
        if pred.sum() == 0:
            rs.append(1.0)
        else:
            rs.append(precision_score(labels, pred))
    return 1 - float(np.mean(rs))


def calc_scores(results: np.ndarray, anno: dict):
    num_predicates = len(anno["predicate_classes"])
    labels = [[] for _ in range(num_predicates)]
    scores = [[] for _ in range(num_predicates)]
    ranks = [[] for _ in range(num_predicates)]

    res_ids = results[:, :3].astype(int)

    for image in anno["data"]:
        img_results = results[res_ids[:, 0] == image["image_id"]]
        img_res_ids = res_ids[res_ids[:, 0] == image["image_id"]]
        assert len(img_results) > 0
        for rel_key, lbl_value in (("relations", 1), ("neg_relations", 0)):
            for sbj, obj, tgt_rel in image[rel_key]:
                if tgt_rel == -1:
                    continue
                assert 0 <= tgt_rel < num_predicates, tgt_rel

                labels[tgt_rel].append(lbl_value)

                rel_results = img_results[(img_res_ids[:, 1] == sbj) & (img_res_ids[:, 2] == obj)]
                # if rel_results is empty, something went wrong during inference,
                # causing some pairs to be skipped
                if len(rel_results) == 0:
                    scores[tgt_rel].append(0.0)
                    ranks[tgt_rel].append(num_predicates - 1)
                    continue

                assert len(rel_results) == 1, rel_results.shape
                rel_scores = rel_results[0, 3:]

                scores[tgt_rel].append(rel_scores[tgt_rel])
                ranks[tgt_rel].append((rel_scores[:-1].argsort()[::-1] == tgt_rel).nonzero()[0][0])

    labels = [np.array(x) for x in labels]
    scores = [np.array(x) for x in scores]
    ranks = [np.array(x) for x in ranks]

    rows = {}
    for p, l, s, r in zip(anno["predicate_classes"], labels, scores, ranks):
        if l.sum() > 3:
            rows[p] = dict(
                auc=roc_auc_score(l, s),
                dd=discrimination_disadvantage(l, r, num_predicates=num_predicates),
                do=dominance_overestimation(l, r, num_predicates=num_predicates),
            )
    return rows


def _cli():
    parser = ArgumentParser()
    parser.add_argument("annotation", help="Haystack annotation file (JSON)")
    parser.add_argument(
        "results",
        help="Model outputs file (NPY). If your model outputs no-relation scores, remove them from this file",
    )
    parser.add_argument("--csv", help="Optional output path to save metrics as CSV")
    args = parser.parse_args()

    with open(args.annotation) as f:
        anno = json.load(f)
    outputs = np.load(args.results)

    metrics = calc_scores(outputs, anno)

    df = pd.DataFrame(metrics)
    df["mean"] = df.values.mean(axis=-1)
    df.index.name = "predicate"
    if args.csv is None:
        print(df.T)
    else:
        df.T.to_csv(args.csv, index=True, index_label="predicate")


if __name__ == "__main__":
    _cli()
