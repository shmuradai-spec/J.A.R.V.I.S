from chat import generate
for q in ['Какой символ у железа?', 'Какой символ у хлора?', 'Какой символ у фосфора?', 'Какой атомный номер у золота?']:
    print(q, '->', generate(q), flush=True)
