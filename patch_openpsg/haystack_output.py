import numpy as np
import torch

def dump_haystack_output(outputs, dataset, output_path):
    img_ids = [i['id'] for i in dataset.data_infos]

    rows = []
    for result, img_id in zip(outputs, img_ids):
        img_id = int(img_id)
        no_bg_scores = torch.from_numpy(result.rel_scores).softmax(dim=-1)[:, 1:]
        for (sbj, obj), scores in zip(result.rel_pair_idxes, no_bg_scores):
            rows.append((img_id, sbj, obj, *scores))

    np.save(output_path, np.array(rows))

