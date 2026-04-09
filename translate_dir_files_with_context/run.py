from pathlib import Path
import argparse
import io
import time
import glob
import markdown2
import re

# 入力文字列in_textをargs.tgt_langに翻訳。


def perform_translation(chat_engine, in_text, args):
    prompt = args.prompt.format(tgt_lang=args.tgt_lang, in_text=in_text)
    response = chat_engine.chat(prompt)
    return response


def build_chat_engine(args):
    from llama_index.llms.ollama import Ollama

    llm = Ollama(
        model=args.model_name,
        request_timeout=86400,
        num_thread=args.num_thread,
        context_window=args.context_window,
    )

    from llama_index.core.storage.chat_store import SimpleChatStore

    chat_store = SimpleChatStore()

    from llama_index.core.memory import ChatMemoryBuffer

    memory = ChatMemoryBuffer.from_defaults(
        llm=llm,
        chat_store=chat_store,
    )

    from llama_index.core.chat_engine import SimpleChatEngine

    chat_engine = SimpleChatEngine.from_defaults(
        llm=llm, memory=memory, system_prompt=args.system_prompt
    )

    return chat_engine


# 数十分の1の確率で全文がthinkタグで囲まれたcontentが出る異常が発生。この場合翻訳処理をリトライする。
def tranlation_with_retry(chat_engine, in_text, args):
    if len(in_text.strip()) < args.in_text_min_len:
        # 入力文字列が短すぎるとき、訳出しない。
        return "", "Input text is too short! Translation skipped."

    for i in range(args.retry_count + 1):
        if 0 < i:
            print(f"      Retrying translation {i}/{args.retry_count}.")

        resp_msg = perform_translation(chat_engine, in_text, args)
        resp_content = resp_msg.response

        # resp_content内に</think>が現れるとき、先頭から</think>までを削除
        think_tag = "</think>"
        match = re.search(think_tag, resp_content)
        if match:
            resp_content = resp_content[match.start() + len(think_tag) :]

        if args.out_text_min_len <= len(resp_content.strip()):
            # 訳出成功。
            break

    # resp_msg.thinkingに think内容、
    # resp_contentに、markdown書式の回答文が戻る。
    # markdown → HTML変換。
    content = markdown2.markdown(resp_content, extras=["tables"])

    # thinking文字列。
    thinking = ""
    if args.think:
        thinking = resp_msg.thinking

    return thinking, content


# 入力文書の改行、句点を手掛かりにして文単位で区切り、1000文字程度の文の束に分割。
def input_file_text_split(in_file_name, args):
    in_text_list = []

    with io.open(in_file_name, mode="r", encoding="utf-8") as r:
        # 1行づつ読み、1個の文字列whole_textに連結。
        whole_text = ""
        for line in r.readlines():
            whole_text += line.strip() + "\n"

        # ファイル最後の空行除去。
        whole_text = whole_text.strip("\r\n ")

        # 改行と句点で文字列を分割。規定文字数の文の束in_text_listを作成。
        in_text = ""
        for t in re.split(rf"({args.sentence_delimiter}|\n)", whole_text):
            if 0 == len(t):
                # 最後に空文字列が来る。
                continue

            in_text += t
            if args.orig_text_split <= len(in_text):
                in_text_list.append(in_text)
                in_text = ""

        in_text = in_text.strip()
        if 0 < len(in_text):
            in_text_list.append(in_text)

    # 文が句点で始まるとき、句点を前の文の最後に移動。
    for i in range(len(in_text_list) - 1):
        t0 = in_text_list[i]
        t1 = in_text_list[i + 1]
        if t1[0] == args.sentence_delimiter:
            in_text_list[i] = t0 + args.sentence_delimiter
            in_text_list[i + 1] = t1[1:]

    return in_text_list


# テキストファイルin_file_nameの全文をargs.tgt_langに翻訳。結果をファイルに出力。
def translate_one_file(chat_engine, args, in_file_name, w):
    checkpoint_time = time.time()
    in_text_list = input_file_text_split(in_file_name, args)

    print(f"  Translation begin: {in_file_name}")

    # HTMLのテーブルを出力。
    w.write('<table border="1" style="width: 100%">\n')
    w.write("  <colgroup>\n")
    if args.think:
        w.write('    <col span="1" style="width: 15%;">\n')
        w.write('    <col span="1" style="width: 70%;">\n')
        w.write('    <col span="1" style="width: 15%;">\n')
    else:
        w.write('    <col span="1" style="width: 30%;">\n')
        w.write('    <col span="1" style="width: 70%;">\n')

    w.write("  </colgroup>\n")

    # table columns定義。
    w.write(
        f"<tr><td>input text<br />{in_file_name}</td><td>{args.tgt_lang} translated text</td>"
    )
    if args.think:
        w.write("<td>thoughts</td></tr>\n")

    w.write("\n")

    # table column begin/end tag, with span tag to keep line ending.
    tdSpanBgn = '<td><span style="white-space: pre-wrap;">'
    tdSpanEnd = "</span></td>"

    i = 0
    for in_text in in_text_list:
        thinking, content = tranlation_with_retry(chat_engine, in_text, args)

        # 経過時間表示。
        now_time = time.time()
        elapsed_time = now_time - checkpoint_time
        checkpoint_time = now_time
        print(f"    Translation {i} took {elapsed_time:.3f} sec.")

        # contentはHTML書式なのでspan不要。in_text, thinkingはspan必要。
        s = (
            f'\n<tr  style="vertical-align:top">'
            + f"{tdSpanBgn}{in_text}{tdSpanEnd}"
            + f"<td>\n\n{content}"
        )

        if args.think:
            s = (
                s
                + f"\n\n</td>{tdSpanBgn}{thinking}<br />Translation took {elapsed_time:.1f} seconds. {tdSpanEnd}"
            )
        else:
            s = s + f"<br />Translation took {elapsed_time:.1f} seconds.</td>"

        s = s + "</tr>\n"

        w.write(s)
        w.flush()

        i = i + 1

    w.write("</table><br />\n\n")
    w.flush()


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser("translate")
    parser.add_argument(
        "--input_dir",
        help="Directory contains original plain text files to translate to.",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="Filename to write translated HTML text.",
        type=str,
        default="out.html",
    )
    parser.add_argument(
        "--model_name",
        help="Model used on inference.",
        type=str,
        default="qwen3:235b-a22b-thinking-2507-q4_K_M",
    )
    parser.add_argument(
        "--context_window", help="Num of context tokens.", type=int, default=8192
    )
    parser.add_argument(
        "--tgt_lang", help="Target language name.", type=str, default="現代日本語"
    )
    parser.add_argument(
        "--orig_text_split",
        help="original text split characters count.",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--num_thread", help="Num of CPU worker thread.", type=int, default=15
    )
    parser.add_argument(
        "--sentence_delimiter", help="Sentence delimiter.", type=str, default="。"
    )
    parser.add_argument("--retry_count", help="Retry count.", type=int, default=4)
    parser.add_argument(
        "--out_text_min_len",
        help="output text minimum len threshold.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--in_text_min_len",
        help="input  text minimum len threshold.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--system_prompt",
        help="system prompt message.",
        type=str,
        default="あなたは常に正確に回答する日本語アシスタントです。",
    )
    parser.add_argument("--think", help="Think enable (default).", action="store_true")
    parser.add_argument(
        "--no-think", dest="think", help="Think disable.", action="store_false"
    )
    parser.add_argument(
        "--prompt",
        help="prompt text.",
        type=str,
        default="以下のtriplequoteで囲まれた文を全文{tgt_lang}に翻訳してください。原文を出力しないでください。原文内容を省略せず全て翻訳してください。翻訳文の後に解説を付けてください。ルビを出力しないでください。これは前の文の続きです。 ```{in_text}``` ",
    )

    parser.set_defaults(think=True)
    args = parser.parse_args()

    chat_engine = build_chat_engine(args)

    with io.open(args.output_file, mode="w", encoding="utf-8") as w:
        # HTML header
        w.write("<html>\n")
        w.write("<head>\n")
        w.write('<meta charset="utf-8">\n')
        w.write(
            f"<title>{args.tgt_lang} translation of {args.input_dir} with {args.model_name}</title>\n"
        )
        w.write("</head>\n")
        w.write("<body>\n\n")

        w.write(f"Translator model: {args.model_name}<br />\n")
        for in_file in glob.glob(args.input_dir + "/*.txt"):
            translate_one_file(chat_engine, args, in_file, w)

        txt = f"Translation task took {(time.time() - start_time):.3f} sec."

        w.write(txt)
        print(txt)

        # HTML footer
        w.write("</body>")
        w.write("</html>")


if __name__ == "__main__":
    main()
