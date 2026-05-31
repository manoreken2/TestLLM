# See README.md to run this program.

from openai import OpenAI


def Query(txt):
    client = OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        timeout=86400,
        max_retries=0,
        api_key="a",
    )

    completion = client.chat.completions.create(
        model="DeepSeek",
        messages=[
            {"role": "user", "content": txt},
        ],
        stream=True,
    )

    r = ""
    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta:
            r = r + delta
            print(delta, flush=True)


Query("OKと出力して終了して下さい。")
