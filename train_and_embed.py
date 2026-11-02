import os
from pathlib import Path

import numpy as np
import torch
from torch.xpu import device
from tqdm import tqdm
from itertools import batched

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_tr_outdir(args):
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    outdir = os.path.join("models", dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_data_outdir(args):
    # mirrors the layout the image pipeline reads via fetchVisual(): data_home/{dataset}_{net}/
    net_name = args.model_name.split("/")[-1]
    file_prefix = f'{args.dataset_name}_{net_name}'
    outdir = Path(args.data_home) / file_prefix
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir, file_prefix


def compute_clf_metrics(preds):
    _preds = preds.predictions.argmax(axis=1)
    _labels = preds.label_ids
    acc = accuracy_score(y_true=_labels, y_pred=_preds)
    recall = recall_score(y_true=_labels, y_pred=_preds, average="micro")
    precision = precision_score(y_true=_labels, y_pred=_preds, average="micro")
    f1 = f1_score(y_true=_labels, y_pred=_preds, average="micro")
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}


def get_classifier(model_name, n_classes=2, use_bfloat16=False, device="cuda"):
    torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes,
                                                               dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_training_args(args):
    training_outdir = get_tr_outdir(args)
    trainer_args = TrainingArguments(
        output_dir=training_outdir,
        do_train=True,
        learning_rate=args.lr,
        num_train_epochs=args.nepochs,
        per_device_train_batch_size=args.train_batchsize,
        per_device_eval_batch_size=args.train_batchsize * 4,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,  # < 1 is treated as a ratio of total training steps
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=50,
        bf16=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        eval_on_start=True,
        load_best_model_at_end=True,
    )

    return trainer_args


def get_cls_bertlike(x):
    # get representation associated with the "CLS" token. In BERT-like models this is usually assigned with the first token of the input sequence (idx=0)
    cls_emebds = x[:, 0, :]
    return cls_emebds


def embed(model, tokenizer, data, selection_strategy, args):
    # preallocate the full-size output arrays and fill them in place, instead of growing
    # Python lists of per-batch tensors and concatenating at the end: for a split with ~1M+
    # examples (e.g. yahoo_answers' train split) the list-accumulation pattern peaks at roughly
    # 2x the final size (the lists plus the freshly-allocated concatenated tensor coexisting in
    # memory), which was enough to get the host OOM-killed on the remote server
    n = len(data)
    hidden_size = model.config.hidden_size
    num_labels = model.config.num_labels

    split_logits = np.empty((n, num_labels), dtype=np.float32)
    split_hidden_states = np.empty((n, hidden_size), dtype=np.float32)
    split_labels = np.empty((n,), dtype=np.int64)

    idx = 0
    for batch in batched(tqdm(data), n=args.embed_batchsize):
        texts, labels = zip(*((d["text"], d["label"]) for d in batch))
        bsz = len(texts)
        with torch.no_grad():
            model_inputs = tokenizer(texts, truncation=True, max_length=args.max_length, padding=True,
                                     return_tensors="pt")  # pad to this batch's own longest sequence
            output = model(**model_inputs.to(args.device), output_hidden_states=True)
            last_hidden_states = output.hidden_states[-1]
            cls_embeds = selection_strategy(last_hidden_states.cpu())

        split_logits[idx:idx + bsz] = output.logits.cpu().numpy()
        split_hidden_states[idx:idx + bsz] = cls_embeds.numpy()
        split_labels[idx:idx + bsz] = np.asarray(labels, dtype=np.int64)
        idx += bsz

    return split_logits, split_hidden_states, split_labels


# every dataset is kept with its original codeframe (no binarization): num_classes is the
# ground truth used everywhere downstream (model head size, metric averaging, etc.)
DATASET_REGISTRY = {
    'imdb': dict(hf_path="stanfordnlp/imdb", num_classes=2),
    'rt': dict(hf_path="cornell-movie-review-data/rotten_tomatoes", num_classes=2),
    'yelp': dict(hf_path="Yelp/yelp_review_full", num_classes=5),
    'ag_news': dict(hf_path="fancyzhx/ag_news", num_classes=4),
    'sst5': dict(hf_path="SetFit/sst5", num_classes=5),
    '20newsgroups': dict(hf_path="SetFit/20_newsgroups", num_classes=20),
    'dbpedia': dict(hf_path="fancyzhx/dbpedia_14", num_classes=14, text_fields=["title", "content"]),
    'yahoo_answers': dict(
        hf_path="community-datasets/yahoo_answers_topics", num_classes=10,
        text_fields=["question_title", "question_content", "best_answer"], label_field="topic",
    ),
    # CogComp/trec is a legacy script-based dataset; datasets>=3 refuses to execute its loading
    # script ("Dataset scripts are no longer supported"). SetFit/TREC-QC is the same data
    # (verified: identical examples/sizes) re-hosted in modern Parquet format.
    'trec': dict(hf_path="SetFit/TREC-QC", num_classes=6, label_field="label_coarse"),
}

def get_num_classes(dataset_name):
    return DATASET_REGISTRY[dataset_name]['num_classes']


def prepare_dataset(dataset_name):
    config = DATASET_REGISTRY.get(dataset_name)
    if config is None:
        raise ValueError(f'unrecognized dataset name {dataset_name}; valid ones are {list(DATASET_REGISTRY.keys())}')

    dataset = load_dataset(config['hf_path'])

    text_fields = config.get('text_fields')
    if text_fields is not None:
        def build_text(example):
            example['text'] = ' '.join(str(example[f]) for f in text_fields if example[f])
            return example

        dataset = dataset.map(build_text)

    label_field = config.get('label_field', 'label')
    if label_field != 'label':
        if 'label' in dataset['train'].column_names:
            # drop the dataset's own unrelated 'label' column first (e.g. TREC-QC's fine-grained
            # 50-class label) so it doesn't collide with the field we're renaming into that slot
            dataset = dataset.remove_columns('label')
        dataset = dataset.rename_column(label_field, 'label')

    return dataset


# fetchVisual()'s file naming convention uses "val", not "validation"
SPLIT_FILENAME = {'train': 'train', 'validation': 'val', 'test': 'test'}


def extract_tensors(model, tokenizer, dataset, split, outdir, file_prefix, args):
    split_data = dataset[split]
    split_logits, split_features, split_labels = (
        embed(model, tokenizer, data=split_data, selection_strategy=get_cls_bertlike, args=args)
    )
    # numerically-stable softmax, in numpy since embed() now returns plain arrays
    shifted = split_logits - split_logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    split_predictions = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    out_path = Path(outdir) / f"{file_prefix}_{SPLIT_FILENAME[split]}_out.npz"
    np.savez(
        out_path,
        features=split_features,
        predictions=split_predictions,
        logits=split_logits,
        targets=split_labels,
    )
    print(f"  saved {split} ({len(split_labels)} examples) -> {out_path}")


def set_learnable_parameters(model, train_backbone):
    # freeze base model -> train only fresh init classification head
    if not train_backbone:
        print("- freezing base model weights")
        for _, layer_weights in model.base_model.named_parameters():
            layer_weights.requires_grad = False

        # check trainable layers
        trainable_layers = []
        for layer_name, layer_weights in model.named_parameters():
            if layer_weights.requires_grad == True:
                trainable_layers.append(layer_name)
        print(f"- trainable layers: {trainable_layers}")
    else:
        print("- fine-tuning the full backbone (all parameters trainable)")
        # gradients + AdamW optimizer state for the whole backbone (not just a small head) is
        # much heavier than the frozen-probe case; trade some compute for memory headroom
        model.gradient_checkpointing_enable()


VAL_FRACTION = 0.20


def compute_val_size(train_size, val_size_cap):
    # take 20% of train for validation, but never more than val_size_cap: with plenty of
    # data we want a large, robust validation pool (it becomes the quantifier's train/val
    # source downstream), but on small datasets 20% keeps enough left over to actually
    # fine-tune the classifier
    return min(val_size_cap, round(VAL_FRACTION * train_size))


def main(args):
    print(f"- model: {args.model_name}")
    print(f"- dataset: {args.dataset_name}")

    dataset = prepare_dataset(args.dataset_name)

    # create validation split if does not exist (datasets that already define one, e.g.
    # rt/sst5, are used as-is)
    if "validation" not in dataset:
        val_size = compute_val_size(len(dataset["train"]), args.val_size_cap)
        print(f"splitting training set into train/validation (val_size={val_size})...")
        _tmp_dataset = dataset["train"].train_test_split(test_size=val_size)
        dataset["train"] = _tmp_dataset["train"]
        dataset["validation"] = _tmp_dataset["test"]

    num_classes = get_num_classes(args.dataset_name)

    model, tokenizer = get_classifier(model_name=args.model_name, n_classes=num_classes, device=args.device)

    set_learnable_parameters(model, train_backbone=args.train_backbone)

    def tokenize_dataset(sample):
        # no padding here: left to the data collator, which pads each training batch to its
        # own longest sequence instead of always padding to max_length
        return tokenizer(sample["text"], truncation=True, max_length=args.max_length)

    # tokenize dataset
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=16)

    # train model
    trainer_args = get_training_args(args)
    print(f"- storing model in {trainer_args.output_dir}")
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=trainer_args,
        compute_metrics=compute_clf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],  # stop if it does not improve after 10 validation steps
    )
    print("\nTraining...")
    trainer.train()

    # Get embeddings, logits and predictions, in the same data_home/{dataset}_{net}/..._out.npz
    # layout the image pipeline reads via fetchVisual()
    data_outdir, file_prefix = get_data_outdir(args)
    print("\nEmbedding...")
    print(f"- storing embeddings in {data_outdir}")
    splits = ["train", "validation", "test"]
    for split in splits:
        extract_tensors(model, tokenizer, dataset, split=split, outdir=data_outdir, file_prefix=file_prefix, args=args)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="yelp", choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--nepochs", type=int, default=5)
    parser.add_argument("--train_batchsize", type=int, default=None,
                         help="default: 64 for a frozen backbone (small head only), 16 when "
                              "--train_backbone is set (gradients+optimizer state for the full "
                              "model need much more memory)")
    parser.add_argument("--embed_batchsize", type=int, default=128)
    parser.add_argument("--val_size_cap", type=int, default=10_000,
                         help="validation split size = min(this, 20%% of train); only used for datasets "
                              "without a native validation split")
    parser.add_argument("--data_home", type=str, default="../data",
                         help="output root for the {dataset}_{net}/{dataset}_{net}_{split}_out.npz files, "
                              "matching the layout fetchVisual() reads for the image datasets")
    parser.add_argument("--lr", type=float, default=None,
                         help="default: 5e-4 for a frozen backbone (training a head from scratch "
                              "tolerates a high LR), 2e-5 when --train_backbone is set (the usual "
                              "range for fine-tuning a pretrained transformer -- 5e-4 would likely "
                              "diverge/destroy the pretrained weights)")
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.lr is None:
        args.lr = 2e-5 if args.train_backbone else 5e-4
    if args.train_batchsize is None:
        args.train_batchsize = 16 if args.train_backbone else 64
    main(args)

    # model_names:
    #   - "google-bert/bert-base-uncased"
    #   - "FacebookAI/roberta-base"
    #   - "distilbert/distilbert-base-uncased"

    # datasets (all kept with their original codeframe, see DATASET_REGISTRY):
    #   - "imdb" ("stanfordnlp/imdb", 2 classes: neg/pos)
    #   - "rt" ("cornell-movie-review-data/rotten_tomatoes", 2 classes: neg/pos)
    #   - "yelp" ("Yelp/yelp_review_full", 5 classes: 1-5 stars)
    #   - "ag_news" ("fancyzhx/ag_news", 4 classes: world/sports/business/sci-tec) <-- only 1 epoch!
    #   - "sst5" ("SetFit/sst5", 5 classes: very neg/neg/neutral/pos/very pos)
    #   - "20newsgroups" ("SetFit/20_newsgroups", 20 classes)
    #   - "dbpedia" ("fancyzhx/dbpedia_14", 14 classes; text = title + content)
    #   - "yahoo_answers" ("community-datasets/yahoo_answers_topics", 10 classes;
    #                       text = question_title + question_content + best_answer)
    #   - "trec" ("CogComp/trec", 6 classes: coarse question type)