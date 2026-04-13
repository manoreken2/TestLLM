def format(filename):
    whole_text = ""

    import io

    with io.open(filename, mode="r", encoding="utf-8") as r:
        # 1行づつ読み、1個の文字列whole_textに連結。
        whole_text = ""
        for line in r.readlines():
            whole_text += line.strip() + "\n"

        # 単一の改行を除去。
        whole_text = whole_text.strip("\r\n ")
        import re

        whole_text = re.sub(r"(?<!\n)\n(?!\n)", "", whole_text)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(whole_text)


import glob

for fn in glob.iglob("史記/*.txt"):
    format(fn)
