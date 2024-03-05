# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_wikitext2(model, nsamples=128, seed=100, seqlen=2048, load=True):
    set_seed(seed)

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print("wikitext2:")
    print("\t--TRAIN:", len(traindata), [len(x) for x in traindata["text"][:20]])
    print("\tmean:", np.mean([len(x) for x in traindata["text"]]))
    print("\tstd: ", np.std([len(x) for x in traindata["text"]]))
    print(
        "\t--TEST: ",
        sum([len(x) for x in testdata]),
        len(testdata),
        [len(x) for x in testdata[:20]],
    )
    print("\tmean:", np.mean([len(x) for x in testdata["text"]]))
    print("\tstd: ", np.std([len(x) for x in testdata["text"]]))

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["text"]))["input_ids"]
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # get random idx along sequence
    rand_idx = random.sample(range(0, len(trainenc) - seqlen), nsamples)

    trainencs = []
    for rand_id in rand_idx:
        substr = trainenc[rand_id : rand_id + seqlen]
        trainencs.append(substr)

    return trainencs, testenc


def get_wikipedia(
    model,
    lang,
    nsamples=128,
    seed=100,
    seqlen=2048,
    split=0.1,
    subsets=(3000, 400),
    olm=True,
    load=True,
):
    print(
        f"Loading [{lang}]... (nsamples={nsamples}, seed={seed}, seqlen={seqlen}, split={split}, subsets={subsets})"
    )
    set_seed(seed)

    import json

    with open("wikipedia_datasets_config.json", "r") as f:
        conf = json.load(f)
    dates, langs = conf["dates"], conf["langs"]

    try:
        if olm:
            print(f"loading dataset... olm/wikipedia[{lang}]")
            data = load_dataset("olm/wikipedia", language=lang, split="train", date=dates[lang])
        else:
            print(f"loading dataset... wikipedia[{lang}]")
            data = load_dataset("wikipedia", langs[lang], split="train", beam_runner="DirectRunner")
    except FileNotFoundError:
        raise FileNotFoundError(
            "This version of wikipedia is no longer online for this language. "
            "Please update wikipedia_dataset_config.json with an available date "
            "(see https://dumps.wikimedia.org/[lang]wiki)"
        )

    data = data.train_test_split(test_size=0.1, seed=seed, shuffle=False)
    traindata, testdata = data["train"], data["test"]

    trainsubset, testsubset = subsets
    print(f"TRAIN SUBSETS: {trainsubset}/{len(traindata)}")
    print(f"TEST  SUBSETS: {testsubset}/{len(testdata)}")

    traindata = traindata.select(range(trainsubset))
    testdata = testdata.select(range(testsubset))

    print(f"parsing dataset... train: {len(traindata)}, test: {len(testdata)}")
    traindata, testdata = traindata["text"], testdata["text"]

    traindata = [word for line in traindata for word in line.split("\n\n")]
    testdata = [word for line in testdata for word in line.split("\n\n")]

    print("dataset size:")
    print("\t--TRAIN:", len(traindata), [len(x) for x in traindata[:20]])
    print("\tmean:", np.mean([len(x) for x in traindata]))
    print("\tstd: ", np.std([len(x) for x in traindata]))
    print(
        "\t--TEST: ",
        sum([len(x) for x in testdata]),
        len(testdata),
        [len(x) for x in testdata[:20]],
    )
    print("\tmean:", np.mean([len(x) for x in testdata]))
    print("\tstd: ", np.std([len(x) for x in testdata]))

    print("tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata))["input_ids"]
    testenc = tokenizer("\n\n".join(testdata), return_tensors="pt")

    # get random idx along sequence
    rand_idx = random.sample(range(0, len(trainenc) - seqlen), nsamples)

    trainencs = []
    for rand_id in rand_idx:
        substr = trainenc[rand_id : rand_id + seqlen]
        trainencs.append(substr)

    return trainencs, testenc


def get_c4(model, nsamples=128, seed=100, seqlen=2048):
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_code_search_net(
    model, lang, nsamples=128, seed=100, seqlen=2048, split=0.1, subsets=(6000, 800), load=True
):
    set_seed(seed)

    traindata = load_dataset("code_search_net", split="train")
    testdata = load_dataset("code_search_net", split="test")
    traindata = traindata.filter(lambda x: x["language"] == lang)
    testdata = testdata.filter(lambda x: x["language"] == lang)

    trainsubset, testsubset = subsets
    print(f"TRAIN SUBSETS: {trainsubset}/{len(traindata)}")
    print(f"TEST  SUBSETS: {testsubset}/{len(testdata)}")

    traindata = traindata.select(range(trainsubset))
    testdata = testdata.select(range(testsubset))

    print("parsing dataset...")
    traindata = [x["whole_func_string"] for x in traindata]
    testdata = [x["whole_func_string"] for x in testdata]

    traindata = [word for line in traindata for word in line.split("\n\n")]
    testdata = [word for line in testdata for word in line.split("\n\n")]

    print("dataset size:")
    print("\t--TRAIN:", len(traindata), [len(x) for x in traindata[:20]])
    print("\tmean:", np.mean([len(x) for x in traindata]))
    print("\tstd: ", np.std([len(x) for x in traindata]))
    print(
        "\t--TEST: ",
        sum([len(x) for x in testdata]),
        len(testdata),
        [len(x) for x in testdata[:20]],
    )
    print("\tmean:", np.mean([len(x) for x in testdata]))
    print("\tstd: ", np.std([len(x) for x in testdata]))

    print("tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata))["input_ids"]
    testenc = tokenizer("\n\n".join(testdata), return_tensors="pt")

    # get random idx along sequence
    rand_idx = random.sample(range(0, len(trainenc) - seqlen), nsamples)

    trainencs = []
    for rand_id in rand_idx:
        substr = trainenc[rand_id : rand_id + seqlen]
        trainencs.append(substr)

    return trainencs, testenc
