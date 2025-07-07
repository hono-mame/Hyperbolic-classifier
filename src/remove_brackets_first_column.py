input_file = '/Users/honokakobayashi/dev/Univ/Research/data/MLP/entity_vector.model.small.txt'
output_file = '/Users/honokakobayashi/dev/Univ/Research/data/MLP/entity_vector.model.small_new.txt'

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        parts = line.rstrip('\n').split()  # 空白区切りで分割
        if len(parts) == 0:
            fout.write('\n')
            continue
        
        first_col = parts[0]
        # 先頭が [ で末尾が ] なら外す
        if first_col.startswith('[') and first_col.endswith(']'):
            first_col = first_col[1:-1]
        
        parts[0] = first_col
        fout.write(' '.join(parts) + '\n')  # 元の区切りはスペースなのでここも' 'に
