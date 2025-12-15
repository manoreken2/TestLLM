import ollama
import argparse
import io
import time
import glob
import markdown2
import re

# 以下のプログラムを参考に作成。
# https://github.com/ollama/ollama-python/blob/main/examples/chat.py


# 入力文字列in_textをargs.tgt_langに翻訳。
def perform_translation(in_text, args):
    prompt = \
      f"Translate the whole text enclosed with triplequote to {args.tgt_lang}. {args.extra} Never output the original text! ```{in_text}``` "

    msg = [
        {
            'role': 'user',
            'content': prompt,
        },
    ]

    opt = {
        'num_thread': args.num_thread,
    }

    response = ollama.chat(model=args.model_name,
                             messages=msg,
                             think=args.think,
                             options=opt)
    return response.message


# 数十分の1の確率で全文がthinkタグで囲まれたcontentが出る異常が発生。この場合翻訳処理をリトライする。
def tranlation_with_retry(in_text, args):
    for i in range(args.retry_count + 1):
        resp_msg = perform_translation(in_text, args)

        resp_content = resp_msg.content

        # resp_content内に</think>が現れるとき、先頭から</think>までを削除
        think_tag="</think>"
        match = re.search(think_tag, resp_content)
        if match:
            resp_content = resp_content[match.start()+len(think_tag):]
        
        if resp_content.strip():
            break

    # resp_msg.thinkingに think内容、
    # resp_contentに、markdown書式の回答文が戻る。
    # markdown → HTML変換。
    content = markdown2.markdown(resp_content, extras=["tables"])

    return resp_msg, content


# 入力文書の句点を手掛かりにして文単位で区切り、1000文字程度の文の束に分割。
def input_file_text_split(in_file_name, args):
    in_text_list = []
    with io.open(in_file_name, mode="r", encoding="utf-8") as r:
        # 1行づつ読み、1個の文字列whole_textに連結。
        whole_text = ""
        for line in r.readlines():
            whole_text += line.strip() + '\n'
        
        # ファイル最後の空行除去。
        whole_text = whole_text.strip('\r\n ')

        # 句点で文字列を分割。規定文字数の文の束in_text_listを作成。
        in_text = ""
        for t in whole_text.split(args.sentence_delimiter):
                if 0 == len(t):
                    # 最後に空文字列が来る。
                    continue
                in_text += t + args.sentence_delimiter
                if args.q_text_limit <= len(in_text):
                    in_text_list.append(in_text)
                    in_text = ""
        in_text = in_text.strip()
        if 0 < len(in_text):
            in_text_list.append(in_text)

    return in_text_list


# テキストファイルin_file_nameの全文をargs.tgt_langに翻訳。結果をファイルに出力。
def translate_one_file(args, in_file_name, w):
    checkpoint_time = time.time()

    in_text_list = input_file_text_split(in_file_name, args)

    print(f"  Translation begin: {in_file_name}")

    # HTMLのテーブルを出力。

    w.write('<table border="1" style="width: 100%">\n')
    w.write('  <colgroup>\n')
    if args.think:
        w.write('    <col span="1" style="width: 15%;">\n')
        w.write('    <col span="1" style="width: 70%;">\n')
        w.write('    <col span="1" style="width: 15%;">\n')
    else:
        w.write('    <col span="1" style="width: 30%;">\n')
        w.write('    <col span="1" style="width: 70%;">\n')
    w.write('  </colgroup>\n')

    # table columns定義。
    w.write(f"<tr><td>input text<br />{in_file_name}</td><td>{args.tgt_lang} translated text</td>")

    if args.think:
        w.write("<td>thoughts</td></tr>\n")

    w.write("\n")

    # table column begin/end tag, with span tag to keep line ending.
    tdBgn='<td><span style=\"white-space: pre-wrap;\">'
    tdEnd='</span></td>'

    i=0
    for in_text in in_text_list:
        resp_msg, content = tranlation_with_retry(in_text, args)

        # 経過時間表示。
        now_time = time.time()
        elapsed_time = now_time - checkpoint_time
        checkpoint_time = now_time
        print(f"    Translation {i} took {elapsed_time:.3f} sec.")

        # contentはHTML書式なのでspan不要。in_text, thinkingはspan必要。
        s = f'\n<tr  style="vertical-align:top">' + \
                  f'{tdBgn}{in_text}{tdEnd}' + \
                  f'<td>\n\n{content}'

        if args.think:
            s = s + f'\n\n</td>{tdBgn}{resp_msg.thinking}<br />Translation took {elapsed_time:.1f} seconds. {tdEnd}'
        else:
            s = s +                                    f'<br />Translation took {elapsed_time:.1f} seconds.</td>'

        s = s + '</tr>\n'

        w.write(s)
        w.flush()
        i = i+1
        
    w.write("</table><br />\n\n")
    w.flush()


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser("translate")
    parser.add_argument("--input_dir",          help="Directory contains original plain text files to translate to.", type=str)
    parser.add_argument("--output_file",        help="Filename to write translated HTML text.",   type=str, default="out.html")
    parser.add_argument("--model_name",         help="Model used on inference.",                  type=str, default="deepseek-r1:32b")
    parser.add_argument("--tgt_lang",           help="Target language name.",                     type=str, default="Modern Japanese")
    parser.add_argument("--extra",              help="Extra prompt text. should end with .",      type=str, default="")
    parser.add_argument("--q_text_limit",       help="Query text limit characters count.",        type=int, default=512)
    parser.add_argument("--num_thread",         help="Num of CPU worker thread.",                 type=int, default=16)
    parser.add_argument("--sentence_delimiter", help="Sentence delimiter.",                       type=str, default="。")
    parser.add_argument("--retry_count",        help="Retry count.",                              type=int, default=1)

    parser.add_argument("--think",              help="Think enable (default).",                   action='store_true')
    parser.add_argument("--no-think", dest="think", help="Think disable.",                        action='store_false')
    parser.set_defaults(think=True)

    args = parser.parse_args()

    with io.open(args.output_file, mode="w", encoding="utf-8") as w:
        w.write(f'Translator model: {args.model_name}<br />\n')
        for in_file in glob.glob(args.input_dir + '/*.txt'):
            translate_one_file(args, in_file, w)

        txt = f"Translation task took {(time.time() - start_time):.3f} sec."
        w.write(txt)
        print(txt)

if __name__ == "__main__":
    main()







