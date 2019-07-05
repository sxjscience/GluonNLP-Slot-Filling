import os, zipfile, io
import numpy as np
from mxnet.gluon.utils import download, check_sha1
from mxnet.gluon.data import SimpleDataset
import gluonnlp
from gluonnlp.vocab import Vocab
from gluonnlp.data import TSVDataset
from gluonnlp.data.utils import Splitter


__DATADIR__ = os.path.join(os.path.dirname(__file__), 'data')


class _BaseIDSFDataset(SimpleDataset):
    """Base Class of Datasets for Joint Intent Detection and Slot Filling

    """
    def __init__(self, segment, root):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._segment = segment
        self._root = root
        self._intent_vocab = None
        self._slot_vocab = None
        self._get_data()
        super(_BaseIDSFDataset, self).__init__(self._read_data(segment))

    @property
    def _download_info(self):
        """

        Returns
        -------
        filename : str
        url : str
        sha1_hash : str
        """
        raise NotImplementedError

    @property
    def intent_vocab(self):
        if self._intent_vocab is None:
            with open(os.path.join(self._root, 'intent_vocab.json'), 'r') as f:
                self._intent_vocab = Vocab.from_json(f.read())
        return self._intent_vocab

    @property
    def slot_vocab(self):
        if self._slot_vocab is None:
            with open(os.path.join(self._root, 'slot_vocab.json'), 'r') as f:
                self._slot_vocab = Vocab.from_json(f.read())
        return self._slot_vocab

    def _get_data(self):
        filename, url, sha1_hash = self._download_info
        data_filename = os.path.join(self._root, filename)
        if not os.path.exists(data_filename) or not check_sha1(data_filename, sha1_hash):
            download(url, path=data_filename, sha1_hash=sha1_hash, verify_ssl=True)
            with zipfile.ZipFile(data_filename, 'r') as zf:
                zf.extractall(self._root)

    def _read_data(self, segment):
        sentences = TSVDataset(os.path.join(self._root, '{}_sentence.txt'.format(segment)),
                               field_separator=Splitter(' '))
        tags = TSVDataset(os.path.join(self._root, '{}_tags.txt'.format(segment)),
                          field_separator=Splitter(' '))
        with io.open(os.path.join(self._root, '{}_intent.txt'.format(segment)), 'r',
                     encoding='utf-8') as f:
            intents = []
            for line in f:
                line = line.strip()
                intents.append(np.array([self.intent_vocab[ele] for ele in line.split(';')],
                                        dtype=np.int32))
        ret = []
        for sentence, seq_tag, intent in zip(sentences, tags, intents):
            ret.append((sentence, seq_tag, intent))
        return ret


@gluonnlp.data.register(segment=['train', 'dev', 'test'])
class ATISDataset(_BaseIDSFDataset):
    """Converted from https://github.com/sz128/slot_filling_and_intent_detection_of_SLU

    Refer:

    [Hakkani-Tur et al. (2016)]
    JointSLU: Joint Semantic Parsing for Spoken/Natural Language Understanding


    """
    def __init__(self, segment='train', root=os.path.join(__DATADIR__, 'atis')):
        super(ATISDataset, self).__init__(segment, root)

    @property
    def _download_info(self):
        return 'atis.zip', 'https://www.dropbox.com/s/w3afmp41t1jtna4/atis.zip?dl=1',\
               'fb75a1b595566d5c5ec06ee6f2296d6629b8c225'


@gluonnlp.data.register(segment=['train', 'dev', 'test'])
class SNIPSDataset(_BaseIDSFDataset):
    """Converted from https://github.com/sz128/slot_filling_and_intent_detection_of_SLU

    Refer:

    [Alice Coucke et al. (2018)]
    Snips Voice Platform: an embedded Spoken Language Understanding system
    for private-by-design voice interfaces

    """
    def __init__(self, segment='train', root=os.path.join(__DATADIR__, 'snips')):
        super(SNIPSDataset, self).__init__(segment, root)

    @property
    def _download_info(self):
        return 'snips.zip', 'https://www.dropbox.com/s/3ashuawsyzpkx0m/snips.zip?dl=1',\
               'f22420cc0f2a26078337dc375606be46a4cc8c51'


