import argparse


def run(args):
    lines = []
    with open(args.input_html_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            # 空行を除去
            if 3 < len(s):
                lines.append(s)

    trans_cnt = 0
    dup = 0
    prev_txt_bgn = ""

    for i in range(len(lines)):
        line = lines[i]
        if i < len(lines) - 1 and "</span></td><td>" in line:
            trans_cnt += 1

            translated_txt = lines[i + 1]
            txt_bgn = translated_txt[: args.inspect_len]
            if args.print_all_translation_beginning:
                print(f"      {txt_bgn}")
            if prev_txt_bgn == txt_bgn:
                print(f"    line {i+1} : possible duplicate {txt_bgn}")
                dup += 1
            prev_txt_bgn = txt_bgn

    print(
        f'"{args.input_html_path}" translation_count {trans_cnt}, possible_duplicate {dup}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_html_path", type=str, help="input html path")
    parser.add_argument(
        "--inspect_len", type=int, default=25, help="compare characters count"
    )
    parser.add_argument("--print_all_translation_beginning", type=bool, default=False)
    args = parser.parse_args()

    run(args)
