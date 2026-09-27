# llama-cpp vulkanで、2個のIntel Arc Pro B60を動かし、OpenCodeで使用する

## PCの構成

- CPU: AMD Ryzen 7950x
- メインメモリ容量: 192GB
- GPU: Intel Arc Pro B60 を2台接続

## 手順

### Intel Arc Pro ドライバーをインストール

https://www.intel.com/content/www/us/en/download/741626/intel-arc-pro-graphics-windows.html

Intel Graphics Softwareを起動して、Resize BARが2台とも有効になっていることを確認

### 作業フォルダー作成

エクスプローラーで

- C:\appフォルダー作成
- C:\hfフォルダー作成

### llama.cpp取得

Releaseページから
b11205のWindows x64(Vulkan)をダウンロード。

https://github.com/ggml-org/llama.cpp/releases

zipを展開して、フォルダをC:\app\llama.cpp
にリネーム

環境変数PATHに、C:\app\llama.cppを追加

CMDを開いて、llama-server --list-devicesすると
Vulkan1とVulkan2が、Intel Arc B60
と表示された。
起動するたびにVulkan0とVulkan2など変わります。

### Miniforge3インストール

リリースページから
Miniforge3-26.7.2-0-Windows-x86_64.exe 取得

https://github.com/conda-forge/miniforge/releases

- All Users
- インストール先はC:\miniforge3にインストールした

### DeepSeekのパラメーターファイル取得

スタートメニューのMiniforgeプロンプトを選択して起動、hf_download取得

```
conda create -y -n hf python=3.12
conda activate hf
conda install pip
pip install hf_download
```

続けてDeepSeek-V4-flashダウンロードし、C:\hf\DeepSeek-V4-Flash-0731-UD-Q8_K_XL.gguf 作成。

```
cd /d C:\hf
for %x in (00001 00002 00003 00004 00005) do hf download hf://unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-%x-of-00005.gguf --local-dir C:/hf/
llama-gguf-split --merge UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf DeepSeek-V4-Flash-0731-UD-Q8_K_XL.gguf 
```

### llama-serverでDeepSeek起動

Vulkanのデバイス名を調べる

```
llama-server --list-devices
```

起動パラメーターはPCの状態に合わせて調整します。Vulkan1とVulkan2がIntel Arc GPUの場合、-dev Vulkan1,Vulkan2を指定。
-tと -tbは、CPUコア数引く1の値にする。このPCは16コアなので15を指定。

llama-server起動

```
llama-server --model C:/hf/DeepSeek-V4-Flash-0731-UD-Q8_K_XL.gguf --jinja  --ctx-size 262144 --load-mode none  --flash-attn on  --batch-size 4096 --ubatch-size 4096 --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8888 -t 15 -tb 15 --parallel 1 -dev Vulkan1,Vulkan2  -sm layer  
```
コンソールの標準出力に
```
llama_server: listening on http://0.0.0.0:8888
```

と出たら、起動成功。

### Webブラウザで動作テストする

http://127.0.0.1:8888
を表示。チャットできることを確認。


### OpenCodeをインストール

OpenCode Desktop Windows (x64)をダウンロード
https://opencode.ai/download

インストールし、OpenCode起動したら、いったん終了

設定ファイル

%USERPROFILE%\.config\opencode\opencode.jsonc

を以下の内容で作成

```
{
  "$schema": "https://opencode.ai/config.json",
  "disabled_providers": [],
  "model": "deepseek-v4-flash",
  "provider": {
    "llama.cpp": {
      "name": "LocalPC",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8888/v1",
        "timeout": 3600000,
        "chunkTimeout": 3600000
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DSV4FL0731",
          "reasoning": true,
          "limit": {
           "context": 262144,
           "output": 131072
         }
        }
      }
    }
  }
}
```

これは、LocalPC プロバイダーが、DSV4FL0731 モデルを提供するという設定。

### OpenCodeを起動

OpenCodeの設定画面で、LocalPCプロバイダーのDSV4FL0731を使用する設定をする

OpenCodeをいったん終了

OpenCodeを起動

LocalPCプロバイダーのDSV4FL0731が選択されていることを確認

挨拶のメッセージなどを入力し、ローカルPCのllama.cppが動くことを確認。

### 実行速度

```
prompt processing: 50 tok/s
token generation: 2.5～3.0 tok/s
```
