import ollama
import argparse
import io

def query(in_text, model_name, tgt_lang):
    prompt = \
      f'Translate the text enclosed with triplequote to {tgt_lang}. Enclose the translated text with triplequote. ```{in_text}``` ' \
      f'Translate the text enclosed with triplequote to {tgt_lang}. Enclose the translated text with triplequote.'

    print(f"{prompt}\n")

    result = ollama.generate(model=model_name, \
                             prompt=prompt)
    out_text = result['response']

    print(f"{out_text}\n")
    return out_text

def main():

    # print(ollama.list().get('models', []))

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
        whole_text = ""
        for line in r.readlines():
            whole_text += line.strip() + ' '

        in_text = ""
        for t in whole_text.split('.'):
                in_text += t + '. '
                if args.q_text_limit <= len(in_text):
                    in_text_list.append(in_text)
                    in_text = ""
        if 0 < len(in_text):
            in_text_list.append(in_text)

    # 翻訳実行、結果をHTML形式で保存する。
    with io.open(args.output, mode="w", encoding="utf-8") as w:
        i=0

        w.write(f'Input text file: {args.input}<br>\nTranslater model: {args.model_name}<br>\n')

        w.write('<table border="1"><br>\n')
        w.write(f"<tr><td>input text</td><td>{args.tgt_lang} translated text</td></tr>\n")

        for in_text in in_text_list:
            out_text = query(in_text, args.model_name, args.tgt_lang)

            w.write(f"\n<tr><td>\n{in_text}\n</td><td>\n{out_text}\n</td></tr>\n")

            # 動作テストのため、数個推論し終了。
            #i = i+1
            #if 3 < i:
            #    return
            w.flush()
            
        w.write("</table>\n")

if __name__ == "__main__":
    main()







