# text2gloss_seq2seq
Sequence to Sequence models to serve the task of converting text to gloss (Sign Language Production pipeline).

## Approaches:
Implemented 3 approached from these papers:

1. Sequence to Sequence Learning with Neural Networks  
From this paper: [https://arxiv.org/abs/1409.3215](https://arxiv.org/abs/1409.3215)  
```bib
@misc{sutskever2014sequencesequencelearningneural,
      title={Sequence to Sequence Learning with Neural Networks}, 
      author={Ilya Sutskever and Oriol Vinyals and Quoc V. Le},
      year={2014},
      eprint={1409.3215},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1409.3215}, 
}
```

2. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation  
From this paper: [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)  
```bib
@misc{cho2014learningphraserepresentationsusing,
      title={Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation}, 
      author={Kyunghyun Cho and Bart van Merrienboer and Caglar Gulcehre and Dzmitry Bahdanau and Fethi Bougares and Holger Schwenk and Yoshua Bengio},
      year={2014},
      eprint={1406.1078},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1406.1078}, 
} 
```

3. Neural Machine Translation by Jointly Learning to Align and Translate  
From this paper: [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)  
```bib
@misc{bahdanau2016neuralmachinetranslationjointly,
      title={Neural Machine Translation by Jointly Learning to Align and Translate}, 
      author={Dzmitry Bahdanau and Kyunghyun Cho and Yoshua Bengio},
      year={2016},
      eprint={1409.0473},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1409.0473}, 
} 
```

## Dataset: aslg_pc12
Synthetic English-ASL Gloss Parallel Corpus 2012 (ASLG-PC12) dataset is used in the training process.  
* gloss: a string feature.  
* text: a string feature.   

Example:
```python
{
    "gloss": "WRITE STATEMENT AND DESC-ORAL QUESTION TABLE SEE MINUTE\n",  
    "text": "written statements and oral questions tabling see minutes\n"
}
```

Reference:  
```bib
@inproceedings{othman2012english,
  title={English-asl gloss parallel corpus 2012: Aslg-pc12},
  author={Othman, Achraf and Jemni, Mohamed},
  booktitle={5th Workshop on the Representation and Processing of Sign Languages: Interactions between Corpus and Lexicon LREC},
  year={2012}
}

```

## Implementation:  
The implementation is done using PyTorch.    

For each approach, a separate folder is created.  
* `main.py` is the main file to run the training process.  
* `models.py` contains the model architecture.  
* `support_funcs.py` contains the utility functions.  

Training results and models' weights files are saved in the same folder.  
Different results come from different batch_size value for the training process.  

#### Results:
| NMT Approach | Batch Size | BLEU | Precision (1-gram) | Precision (2-grams) | Precision (3-grams) | Precision (4-grams) |
|---|---|---|---|---|---|---|
| Seq2seq Learning with Neural Networks (LSTMs) | 32 | 0.574 | 0.766 | 0.626 | 0.522 | 0.441 |
| Seq2seq Learning with Neural Networks (LSTMs) | 64 | 0.534 | 0.737 | 0.587 | 0.477 | 0.395 |
| Learning Phrase Representations using RNN Enc-Dec for SMT (GRUS) | 32 | 0.596 | 0.806 | 0.650 | 0.537 | 0.449 |
| Learning Phrase Representations using RNN Enc-Dec for SMT (GRUS) | 64 | 0.594 | 0.803 | 0.645 | 0.535 | 0.450 |
| Learning Phrase Representations using RNN Enc-Dec for SMT (GRUS) | 128 | 0.569 | 0.787 | 0.625 | 0.509 | 0.420 |
| NMT by Jointly Learning to Align and Translate (Attention) | 32 | 0.906 | 0.937 | 0.915 | 0.895 | 0.877 |
| NMT by Jointly Learning to Align and Translate (Attention) | 64 | 0.892 | 0.924 | 0.901 | 0.881 | 0.863 |
| NMT by Jointly Learning to Align and Translate (Attention) | 128 | 0.896 | 0.933 | 0.908 | 0.886 | 0.865 |

