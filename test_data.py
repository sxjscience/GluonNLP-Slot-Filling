from data import ATISDataset, SNIPSDataset
import numpy as np

def test_atis():
    train_data = ATISDataset('train')
    dev_data = ATISDataset('dev')
    test_data = ATISDataset('test')
    assert len(train_data) == 4478
    assert len(dev_data) == 500
    assert len(test_data) == 893
    assert len(train_data.intent_vocab) == 18
    assert len(train_data.slot_vocab) == 127
    all_slot_types = set()
    for ele in train_data.slot_vocab.idx_to_token:
        if ele.startswith('B') or ele.startswith('I'):
            all_slot_types.add(ele[2:])
        else:
            all_slot_types.add(ele)
    assert len(all_slot_types) == 84
    slot_count = np.zeros(shape=(len(train_data.slot_vocab)), dtype=np.int32)
    for dataset in [train_data, test_data, dev_data]:
        for word_tokens, tags, intent_label in dataset:
            tag_ids = train_data.slot_vocab[tags]
            slot_count[np.array(tag_ids)] += 1
            assert len(word_tokens) ==  len(tags)
    assert (slot_count > 0).all()


def test_snips():
    train_data = SNIPSDataset('train')
    dev_data = SNIPSDataset('dev')
    test_data = SNIPSDataset('test')
    assert len(train_data) == 13084
    assert len(dev_data) == 700
    assert len(test_data) == 700
    assert len(train_data.intent_vocab) == 7
    assert len(train_data.slot_vocab) == 72
    all_slot_types = set()
    for ele in train_data.slot_vocab.idx_to_token:
        if ele.startswith('B') or ele.startswith('I'):
            all_slot_types.add(ele[2:])
        else:
            all_slot_types.add(ele)
    assert len(all_slot_types) == 40
    slot_count = np.zeros(shape=(len(train_data.slot_vocab)), dtype=np.int32)
    for dataset in [train_data, test_data, dev_data]:
        for word_tokens, tags, intent_label in dataset:
            tag_ids = train_data.slot_vocab[tags]
            slot_count[np.array(tag_ids)] += 1
            assert len(word_tokens) == len(tags)
    assert (slot_count > 0).all()


if __name__ == "__main__":
    test_atis()
    test_snips()