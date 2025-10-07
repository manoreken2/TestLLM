import argparse
import glob
import io
import os
import re

def declutter(in_filepath, out_dir):
    filename = os.path.basename(in_filepath)
    with io.open(os.path.join(out_dir, filename), mode="w", encoding="utf-8") as w:
        with io.open(in_filepath, mode="r", encoding="utf-8") as r:
            for orig_line in r.readlines():
                orig_line = orig_line.strip()
                if orig_line == "":
                    continue

                # Wiki文庫の脚注記号除去。[123]的なもの。
                # カギ括弧で囲まれた部分(最小一致)を削除。
                out_line = re.sub(re.escape('[') + '.+?' + re.escape(']'), '', orig_line)

                # \u300を除去。
                out_line = re.sub('　', ' ', out_line)

                w.write(out_line + '\n')


def main():
    parser = argparse.ArgumentParser("translate")
    parser.add_argument("--input_dir",          help="Directory contains original plain text files to perform decluttering.", type=str, default="shijitsugan")
    parser.add_argument("--output_dir",        help="Output directory.",   type=str, default="out")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for in_filepath in glob.glob(args.input_dir + '/*.txt'):
        declutter(in_filepath, args.output_dir)

if __name__ == "__main__":
    main()

