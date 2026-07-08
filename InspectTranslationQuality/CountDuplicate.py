import argparse


def gather_text(lines, idx, count):
    text_out = ""
    while idx < len(lines) - 1:
        t = lines[idx]
        text_out = text_out + t[:count]

        idx += 1
        count -= len(text_out)
        if count <= len(text_out):
            break

    return text_out


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
    in_text_begin = ""

    for i in range(len(lines)):
        line = lines[i]
        if i == len(lines) - 1:
            break

        if '<span style="white-space: pre-wrap;">' in line:
            in_text_begin = gather_text(lines, i + 1, args.inspect_len)

        elif "</span></td><td>" in line:
            trans_cnt += 1

            out_txt_bgn = gather_text(lines, i + 1, args.inspect_len)
            if args.print_in_out:
                print(f"{in_text_begin}\n{out_txt_bgn}\n")
            elif args.print_all_translation_beginning:
                print(f"{out_txt_bgn}")
            # if prev_txt_bgn == txt_bgn:
            #    print(f"    line {i+1} : possible duplicate {txt_bgn}")
            #    dup += 1
            prev_txt_bgn = out_txt_bgn

    print(
        f'"{args.input_html_path}" translation_count {trans_cnt}, possible_duplicate {dup}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_html_path", type=str, help="input html path")
    parser.add_argument(
        "--inspect_len", type=int, default=40, help="compare characters count"
    )
    parser.add_argument("--print_all_translation_beginning", type=bool, default=True)
    parser.add_argument("--print_in_out", type=bool, default=True)
    args = parser.parse_args()

    run(args)
