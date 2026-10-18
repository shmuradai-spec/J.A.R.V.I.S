import json

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3

    @property
    def vocab_size(self):
        return len(self.char_to_id)

    def build_vocab(self, texts):
        chars = sorted(set(''.join(texts)))
        tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'] + chars
        self.char_to_id = {c: i for i, c in enumerate(tokens)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}

    def encode(self, text, add_bos=False, add_eos=False):
        ids = ([self.bos_id] if add_bos else [])
        ids += [self.char_to_id.get(c, self.unk_id) for c in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        chars = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.bos_id, self.unk_id):
                continue
            chars.append(self.id_to_char.get(idx, ''))
        return ''.join(chars)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'char_to_id': self.char_to_id}, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, encoding='utf-8') as f:
            self.char_to_id = json.load(f)['char_to_id']
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
