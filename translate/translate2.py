import ollama
import argparse
import io
import time    

def query(in_text, model_name, tgt_lang):
    prompt = \
      f'Translate the text enclosed with triplequote to {tgt_lang}. Enclose the translated text with triplequote. ```{in_text}``` ' \
      f'Translate the text enclosed with triplequote to {tgt_lang}. Enclose the translated text with triplequote.'

    # print(f"{prompt}\n")

    result = ollama.generate(model=model_name, \
                             prompt=prompt)
    out_text = result['response']

    print(f"{out_text}\n")
    return out_text

def main():
    # print(ollama.list().get('models', []))
    start_time = int(time.time())

    parser = argparse.ArgumentParser("translate")
    parser.add_argument("--input",              help="Original plain text file to translate to.", type=str)
    parser.add_argument("--output",             help="Filename to write translated HTML text.",   type=str, default="out.html")
    parser.add_argument("--model_name",         help="Model used on inference.",                  type=str, default="deepseek-r1:32b")
    parser.add_argument("--tgt_lang",           help="Target language name.",                     type=str, default="Modern Japanese")
    parser.add_argument("--q_text_limit",       help="Query text limit characters count.",        type=int, default=2048)
    parser.add_argument("--sentence_delimiter", help="Sentence delimiter.",                       type=str, default=".")

    args = parser.parse_args()

    # 入力文書を文単位で区切って、2048文字程度の文の束に分割。
    # q_text_list: the list of original text sentences.
    in_text_list = []
    with io.open(args.input, mode="r", encoding="utf-8") as r:        
        # 1行づつ読み、1個の文字列whole_textに連結。
        whole_text = ""
        for line in r.readlines():
            whole_text += line.strip() + '\n'

        # 句点で文字列を分割。規定文字数の文の束in_text_listを作成。
        in_text = ""
        for t in whole_text.split(args.sentence_delimiter):
                in_text += t + args.sentence_delimiter
                if args.q_text_limit <= len(in_text):
                    in_text_list.append(in_text)
                    in_text = ""
        if 0 < len(in_text):
            in_text_list.append(in_text)

    begin_time = int(time.time())
    print(f" Preprocessing took {begin_time - start_time} sec.")
    print(f" Inference begin.")

    # 翻訳実行、結果をHTML形式で保存する。
    with io.open(args.output, mode="w", encoding="utf-8") as w:
        i=0

        w.write(f'Input text file: {args.input}<br>\nTranslater model: {args.model_name}<br>\n')
        w.write('<table border="1"><br>\n')

        s = f"<tr><td>input text</td><td>thoughts</td><td>{args.tgt_lang} translated text</td><td>extra comments</td></tr>\n"
        w.write(s)

        for in_text in in_text_list:
            out_text = query(in_text, args.model_name, args.tgt_lang)

            # column begin/end tag
            tdBgn='<td><span style=\"white-space: pre-wrap;\">'
            tdEnd='</span></td>'

            # out_textは、<think>考察</think> '''訳文'''考察2
            # の形式で戻る。トリプルクォートの字が数種類ある。
            # 稀に考察の中にトリプルクォートが現れることがあり、その場合列が増えてズレが発生するため
            # 出力ファイルを観察し手動修正する必要あり。
            # 考察2は省略される場合があるがこれは問題ない。
            out_text = out_text.replace("<think>", "")
            out_text = out_text.replace("</think>", "")
            out_text = out_text.replace("'''", f"\n{tdEnd}{tdBgn}\n")
            out_text = out_text.replace('"""', f"\n{tdEnd}{tdBgn}\n")
            out_text = out_text.replace("```", f"\n{tdEnd}{tdBgn}\n")

            s = f"\n<tr>{tdBgn}\n{in_text}\n{tdEnd}" + \
                      f"{tdBgn}\n{out_text}\n{tdEnd}\n"
            w.write(s)

            # 経過時間表示。
            now_time = int(time.time())
            print(f" Inference {i} took {now_time-begin_time} sec.")
            begin_time = now_time

            # 動作テストのため、数回推論し終了。
            #if 3 <= i:
            #    break

            w.flush()
            i = i+1
            
        w.write("</table>\n")
    
    finish_time = int(time.time())
    print(f" Elapsed time: {finish_time - start_time} sec.")


if __name__ == "__main__":
    main()







