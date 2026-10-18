from pathlib import Path
import torch
from model.tokenizer import CharTokenizer
from model.transformer import JarvisTransformer

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = CharTokenizer()
tokenizer.load(ROOT / 'checkpoints' / 'tokenizer.json')
model = JarvisTransformer(vocab_size=tokenizer.vocab_size, d_model=128, nhead=4,
                          num_layers=4, dim_feedforward=512, max_length=384, dropout=0.1).to(DEVICE)
checkpoint = torch.load(ROOT / 'checkpoints' / 'jarvis.pt', map_location=DEVICE, weights_only=True)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    checkpoint = checkpoint['model_state_dict']
model.load_state_dict(checkpoint)
model.eval()
print('JARVIS 2.2 готов |', DEVICE)

@torch.inference_mode()
def generate(question, max_new_tokens=48):
    prompt = f'Вопрос: {question.strip()}\nОтвет: '
    ids = tokenizer.encode(prompt, add_bos=True)
    if len(ids) >= model.max_length:
        return 'Слишком длинный вопрос, сэр.'
    prompt_len = len(ids)
    for _ in range(min(max_new_tokens, model.max_length - len(ids))):
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        next_id = int(model(x)[0, -1].argmax())
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    answer = tokenizer.decode(ids[prompt_len:]).strip()
    # Защита от бесконечного повторения, не исправляет неверные химические факты.
    if 'сэр.' in answer:
        answer = answer.split('сэр.', 1)[0] + 'сэр.'
    return answer or '(нет ответа)'

if __name__ == '__main__':
    while True:
        question = input('Ты: ').strip()
        if question.lower() in ('выход', 'exit', 'quit'):
            break
        print('Джарвис:', generate(question))
