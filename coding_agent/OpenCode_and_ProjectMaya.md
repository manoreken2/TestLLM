# OpenCode動かし方 (Project Maya ローカルAI実行)

## 用意するもの

 - OS: Windows 11 Pro
 - CPU: IntelまたはAMDのCPU
 - GPU: NVIDIA Geforce RTX 5090
 - メモリ: 192 GB以上

## Windowsの設定

設定→システム→詳細設定画面で
 - 長いパスを有効にします
 - 開発者モードを有効にします

エクスプローラーを開いて、隠しファイル、隠しフォルダーを表示します
 - フォルダー オプション→表示タブ→詳細設定のファイルとフォルダーの表示→隠しファイル、隠しフォルダー、および隠しドライブを表示するをチェック

## NVIDIAドライバのインストール

NVIDIAグラフィックスドライバーをインストール。
https://www.nvidia.com/en-us/drivers/

## Project Mayaのセットアップ

Project MayaのReleasesページを訪れ、Project Maya v1.1.27
のSource Code (zip)をダウンロードします。
https://github.com/mw00/project-maya

zipファイルを展開し、フォルダを
C:\app\project-maya-1.1.27に置きます。

MiniForgeをダウンロードしてインストールします。 https://github.com/conda-forge/miniforge/releases/

MiniForge Promptを開いて、condaのmaya環境を作成します。

```
conda create -y -n maya python=3.12
```

## Project Maya実行方法

MiniForge Promptを開いて、condaのmaya環境を有効化します。
その後、C:\app\project-maya-1.1.27に移動してSTART-MAYA.bat
を実行します。


```
conda activate maya
cd /d C:\app\project-maya-1.1.27
.\START-MAYA.bat
```
あとは、案内に従って、キーボード操作でYES等を選択します。
context sizeは128KB。
モデルはGLM-5.3-Flash-Maya-Lを選択しました。

二度目以降に起動するときも上記の手順でOK

## OpenCodeの設定

OpenCodeのGitHubのReleasesページに行って、opencode-desktop-win-x64.exe を取得、インストール。 https://github.com/anomalyco/opencode/

インストール完了するとOpenCodeが起動し、新しいセッションが作成される。入力欄の下のほうにLLMのモデル名をクリックしモデルを管理、その他70以上のプロバイダーを見るをクリック。

 - OpenAI互換のカスタムプロバイダーを選択
 - プロバイダーIDをlocal_pc
 - 表示名をlocal_pc
 - ベースURLを `http://127.0.0.1:8080/v1`
 - APIキーを空欄
 - モデルをlocal
 - 表示名をGLM53fMayaL
 - ヘッダー(オプション)を空欄
 
 以上の設定を行い送信ボタンを押す。
 
 OpenCodeをいったん終了(重要)。
 
 もう一度OpenCodeを起動すると、モデル一覧の下のほうに local_pcのGLM53fMayaLが現れ、ローカルのProject MayaでOpenCodeが動きます。

 ## コンテキストサイズの設定

Project Mayaセットアップ時に、 コンテキストサイズ128KB(131072)と指定しました。これをopencodeの設定にも記述します。
 
エクスプローラーで `%USERPROFILE%/.config/opencode` を開き、以下のように編集します。

 ```
{
  "$schema": "https://opencode.ai/config.json",
  "disabled_providers": [],
  "compaction": {
    "auto": true,
    "keep": {
      "tokens": 15000
    },
    "buffer": 40000
  },
  "model": "localmodel",
  "provider": {
    "llama.cpp": {
      "name": "LocalPC",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "timeout": 3600000,
        "chunkTimeout": 3600000
      },
      "models": {
        "localmodel": {
          "name": "GLM53fMayaL",
          "reasoning": true,
          "limit": {
           "context": 130000,
           "output": 32768
         }
        }
      }
    }
  }
}
```
