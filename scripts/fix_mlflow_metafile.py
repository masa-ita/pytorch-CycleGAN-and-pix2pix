import os
import fileinput

# スキャン対象のディレクトリパス
target_directory = '/raid/itagaki/mlruns'  # ディレクトリの実際のパスに置き換えてください

# 置換前の文字列と置換後の文字列
search_text = 'file:///home/itagaki/pytorch-CycleGAN-and-pix2pix/mlruns'
replace_text = 'file:///raid/itagaki/mlruns'

# ディレクトリ内を再帰的にスキャン
for root, dirs, files in os.walk(target_directory):
    for file in files:
        if file == 'meta.yaml':
            file_path = os.path.join(root, file)
            
            # ファイル内のテキストを置換
            with fileinput.FileInput(file_path, inplace=True) as file:
                for line in file:
                    print(line.replace(search_text, replace_text), end='')

print('置換が完了しました。')
