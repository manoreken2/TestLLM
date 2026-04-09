import re
import io

in_text_list = []

with io.open("999_太平記.txt", mode="r", encoding="utf-8") as r:
    # 1行づつ読み、1個の文字列whole_textに連結。
    whole_text = ""
    for line in r.readlines():
        whole_text += line.strip() + "\n"

    # ファイル最後の空行除去。
    whole_text = whole_text.strip("\r\n ")

    for t in re.split(rf"太平記\n", whole_text):
        t = t.strip()
        if 0 == len(t):
            # 空文字列追加しない。
            continue

        in_text_list.append("太平記\n" + t)

for i in range(len(in_text_list)):
    t = in_text_list[i]
    with open(f"{i+1:03}_太平記_巻{i+1}.txt", "w", encoding="utf-8") as f:
        f.write(t)
