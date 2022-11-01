## RoMe: A Robust Metric for Evaluating Natural Language Generation
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

PyTorch code for **ACL 2022** paper: RoMe: A Robust Metric for Evaluating Natural Language Generation [[PDF]](https://aclanthology.org/2022.acl-long.387/).

### Installation (anaconda)
```commandline
conda create -n rome -y python=3.8 && conda activate rome
pip install -r requirements.txt
chmod +x setup.sh
./setup.sh
```

### Run
```
python rome.py
```

**NB:** The components of RoMe are highly parameter sensitive. We recommend users to try different parameters when adapting the code for different domain or dataset.


### Citation
If you use the code, please cite the following paper.
```
@inproceedings{rony-etal-2022-rome,
    title = "{R}o{M}e: A Robust Metric for Evaluating Natural Language Generation",
    author = "Rony, Md Rashad Al Hasan  and
      Kovriguina, Liubov  and
      Chaudhuri, Debanjan  and
      Usbeck, Ricardo  and
      Lehmann, Jens",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.387",
    pages = "5645--5657",
    abstract = "Evaluating Natural Language Generation (NLG) systems is a challenging task. Firstly, the metric should ensure that the generated hypothesis reflects the reference{'}s semantics. Secondly, it should consider the grammatical quality of the generated sentence. Thirdly, it should be robust enough to handle various surface forms of the generated sentence. Thus, an effective evaluation metric has to be multifaceted. In this paper, we propose an automatic evaluation metric incorporating several core aspects of natural language understanding (language competence, syntactic and semantic variation). Our proposed metric, RoMe, is trained on language features such as semantic similarity combined with tree edit distance and grammatical acceptability, using a self-supervised neural network to assess the overall quality of the generated sentence. Moreover, we perform an extensive robustness analysis of the state-of-the-art methods and RoMe. Empirical results suggest that RoMe has a stronger correlation to human judgment over state-of-the-art metrics in evaluating system-generated sentences across several NLG tasks.",
}
```

### License
[MIT](https://github.com/rashad101/RoMe/blob/main/LICENSE.md)

### Contact
For further information, contact the corresponding author Md Rashad Al Hasan Rony ([email](mailto:rashad.research@gmail.com)).
