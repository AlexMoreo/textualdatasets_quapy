import numpy as np
import os

target_dir = 'data-remap'
os.makedirs(target_dir, exist_ok=True)


# datanames = ["cifar10_resnet18", "cifar100_resnet18", "cifar100coarse_resnet18", "fashionmnist_basiccnn", "svhn_resnet18", "mnist_basiccnn"]

datanames = [
    "20newsgroups_roberta-base", "ag_news_roberta-base", "dbpedia_roberta-base",
    "imdb_roberta-base", "rt_roberta-base", "sst5_roberta-base", "trec_roberta-base",
    "yahoo_answers_roberta-base", "yelp_roberta-base",
]

for dataname in datanames:
    print(dataname)
    remap = {}
    for split in ['train', 'val', 'test']:
        filename = f"{dataname}/{dataname}_{split}_out.npz"
        npz_split_file = np.load(filename)

        for embedding_type in npz_split_file.files:
            print('\t', split, embedding_type)
            if embedding_type not in remap:
                remap[embedding_type] = {}
            remap[embedding_type][split] = npz_split_file[embedding_type]

    embedding_types = list(remap.keys())
    print('\t', embedding_types)
    for embedding_type in embedding_types:
        output_file = f"{target_dir}/{dataname}_{embedding_type}.npz"
        print(f'\tsaving {output_file}')
        np.savez(output_file, **remap[embedding_type])

print("[done]")


