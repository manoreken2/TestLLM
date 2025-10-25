import ollama
import argparse
import io
import time
import glob
import markdown2

# 以下のプログラムを参考に作成。
# https://github.com/ollama/ollama-python/blob/main/examples/chat.py

def query(in_text, model_name, tgt_lang):
    prompt = \
      f"Translate the whole text enclosed with triplequote to {tgt_lang}. Never output the original text! Think with Japanese language. ```{in_text}``` "

    messages = [
        {
            'role': 'user',
            'content': prompt,
        },
    ]

    response = ollama.chat(model=model_name,
                             messages=messages,
                             think=True)
    return response.message

def translate_one_file(args, in_file_name, w):
    checkpoint_time = time.time()

    # 入力文書を文単位で区切って、2048文字程度の文の束に分割。
    # q_text_list: the list of original text sentences.
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

    print(f" Inference begin. {in_file_name}")

    # 翻訳実行、結果をHTML形式で保存する。
    i=0

    w.write('<table border="1" style="width: 100%">\n')
    w.write('  <colgroup>\n')
    w.write('    <col span="1" style="width: 15%;">\n')
    w.write('    <col span="1" style="width: 70%;">\n')
    w.write('    <col span="1" style="width: 15%;">\n')
    w.write('  </colgroup>\n')

    s = f"<tr><td>input text<br />{in_file_name}</td><td>{args.tgt_lang} translated text</td><td>thoughts</td></tr>\n"
    w.write(s)

    # table column begin/end tag, with span tag to keep line ending.
    tdBgn='<td><span style=\"white-space: pre-wrap;\">'
    tdEnd='</span></td>'

    for in_text in in_text_list:
        resp_msg = query(in_text, args.model_name, args.tgt_lang)

        # 経過時間表示。
        now_time = time.time()
        elapsed_time = now_time - checkpoint_time
        checkpoint_time = now_time
        print(f"  Translation {i} took {elapsed_time:.3f} sec.")

        # resp_msg.thinkingに think内容、
        # resp_msg.contentに、markdown書式の回答文が戻る。
        # markdown → HTML変換。
        content = markdown2.markdown(resp_msg.content, extras=["tables"])

        # contentはHTML書式なのでspan不要。in_text, thinkingはspan必要。
        s = f'\n<tr  style="vertical-align:top">' + \
                  f'{tdBgn}{in_text}{tdEnd}' + \
                  f'<td>{content}</td>' + \
                  f'{tdBgn}{resp_msg.thinking}<br />Translation took {elapsed_time:.1f} seconds. {tdEnd}' + \
              f'</tr>\n'
        w.write(s)
        w.flush()
        i = i+1
        
    w.write("</table><br />\n")
    w.flush()

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser("translate")
    parser.add_argument("--input_dir",          help="Directory contains original plain text files to translate to.", type=str)
    parser.add_argument("--output_file",        help="Filename to write translated HTML text.",   type=str, default="out.html")
    parser.add_argument("--model_name",         help="Model used on inference.",                  type=str, default="deepseek-r1:32b")
    parser.add_argument("--tgt_lang",           help="Target language name.",                     type=str, default="Modern Japanese")
    parser.add_argument("--q_text_limit",       help="Query text limit characters count.",        type=int, default=1024)
    parser.add_argument("--sentence_delimiter", help="Sentence delimiter.",                       type=str, default="。")

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







