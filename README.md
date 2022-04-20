# UniTE

This is the repository of UniTE: Unified Translation Evaluation, which is build on COMET.
Datasets can be found at COMET repository.

# Usage

1. Download this repository via git;
2. Download the checkpoints from google drive: https://drive.google.com/file/d/1TzWUt2pqiY45wbbRII0SgVx8Kk6wzE2H/;
3. Extract the zip file, and you can get checkpoints ('ckpt' as suffix) and corresponding setting files ('yaml' files, totally three);
4. Run score.py to score the files you want:
```
Source-Only
python score.py -s src.txt -r ref.txt -t trans.txt --model model.ckpt --to_json results.src.json --hparams_file_path hparams.src.yaml

Reference-Only
python score.py -s src.txt -r ref.txt -t trans.txt --model model.ckpt --to_json results.ref.json --hparams_file_path hparams.ref.yaml

Source-Reference-Combined
python score.py -s src.txt -r ref.txt -t trans.txt --model model.ckpt --to_json results.src_ref.json --hparams_file_path hparams.src_ref.yaml
```

where:
`src.txt` stores the source inputs
`ref.txt` stores the target reference
`trans.txt` stores the translation outputs (can also be named as candidates or hypotheses)
`model.ckpt` is the path of model checkpoint
`results.***.json` is the output path of json-formatted scores
`hparams.***.yaml` is the path of setting file

# Citation

Please cite our paper if you find useful:

```
@inproceedings{wan2021robleurt,
    title = "{{RoBLEURT Submission for WMT2021 Metrics Task}}",
    author = "Wan, Yu  and
      Liu, Dayiheng  and
      Yang, Baosong  and
      Bi, Tianchi  and
      Zhang, Haibo  and
      Chen, Boxing  and
      Luo, Weihua  and
      Wong, Derek F.  and
      Chao, Lidia S.",
    booktitle = "Proceedings of the Sixth Conference on Machine Translation (WMT)",
    year = "2021",
}

@inproceedings{wan2022unite,
    title = "{{UniTE: Unified Translation Evaluation}}",
    author = "Wan, Yu  and
      Liu, Dayiheng  and
      Yang, Baosong  and
      Zhang, Haibo  and
      Chen, Boxing  and
      Wong, Derek F.  and
      Chao, Lidia S.",
    booktitle = "Annual Meeting of the Association for Computational Linguistics (ACL)",
    year = "2022",
}
```
