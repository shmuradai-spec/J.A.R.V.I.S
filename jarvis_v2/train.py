import json
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.tokenizer import CharTokenizer
from model.transformer import JarvisTransformer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
OUT = ROOT / 'checkpoints'
OUT.mkdir(exist_ok=True)
MAX_EPOCHS = 60
PATIENCE = 25  # остановка при отсутствии улучшения обучающей ошибки
BATCH_SIZE = 16
LR = 3e-4
MAX_LENGTH = 384
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| GPU:', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'нет', flush=True)

# Load all Q&A JSON datasets. Repeat personality/dialogue examples to balance topics.
weights = {'standard': 3, 'dialogue': 2, 'chemistry': 1, 'math': 1, 'film': 2}
pairs = []
counts = {}
for path in sorted(DATA_DIR.glob('*.json')):
    with path.open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'{path.name}: expected JSON list')
    count = 0
    for item in data:
        q = str(item.get('question', '')).strip()
        a = str(item.get('answer', '')).strip()
        if not q or not a:
            continue
        variants = [(q, a)]
        if q.endswith('?'):
            variants.append((q[:-1], a))
        for _ in range(weights.get(path.stem, 1)):
            pairs.extend(variants)
        count += 1
    counts[path.name] = count
if not pairs:
    raise ValueError('No nonempty Q&A examples found in data/*.json')
print('Datasets:', counts, '| Training variants:', len(pairs), flush=True)

tokenizer = CharTokenizer()
all_texts = [f'Вопрос: {q}\nОтвет: {a}' for q, a in pairs]
tokenizer.build_vocab(all_texts)
tokenizer.save(OUT / 'tokenizer.json')

class QADataset(Dataset):
    def __init__(self):
        self.samples = []
        for q, a in pairs:
            prompt = f'Вопрос: {q}\nОтвет: '
            p = tokenizer.encode(prompt, add_bos=True)
            answer = tokenizer.encode(a, add_eos=True)
            full = p + answer
            if len(full) > MAX_LENGTH:
                raise ValueError('Слишком длинный пример: ' + q)
            # Сдвиг: logits в позиции len(p)-1 предсказывает первый символ ответа.
            labels = [-100] * (len(p) - 1) + answer
            assert len(labels) == len(full) - 1
            self.samples.append((full[:-1], labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

def collate(batch):
    maxlen = max(len(x) for x, _ in batch)
    x = torch.full((len(batch), maxlen), tokenizer.pad_id, dtype=torch.long)
    y = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(labels)] = torch.tensor(labels)
    return x, y

loader = DataLoader(QADataset(), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
model = JarvisTransformer(vocab_size=tokenizer.vocab_size, d_model=128, nhead=4,
                          num_layers=4, dim_feedforward=512, max_length=MAX_LENGTH, dropout=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
best, stale = float('inf'), 0
print('Vocabulary:', tokenizer.vocab_size, '| Parameters:', sum(p.numel() for p in model.parameters()), flush=True)
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        count += 1
    avg = total_loss / count
    if avg < best - 0.0001:
        best, stale = avg, 0
        torch.save(model.state_dict(), OUT / 'jarvis.pt')
        mark = 'SAVED'
    else:
        stale += 1
        mark = f'patience {stale}/{PATIENCE}'
    print(f'Epoch {epoch:03d}/{MAX_EPOCHS} | answer loss {avg:.5f} | {mark}', flush=True)
    if stale >= PATIENCE:
        print('Early stopping: training loss did not improve.')
        break
print('DONE. Best training loss:', best)
print('Важно: малая модель может ошибаться даже при низкой обучающей ошибке.')
