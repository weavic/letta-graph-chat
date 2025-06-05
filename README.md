## 環境設定手順

## 初回

```bash
uv init memory-adaptor-demo
```

## 開発環境要Package

```bash
uv add "black" "ruff" "pytest" "anyio" "mypy" "coverage" --dev
```

## Package

```bash
uv add "fastapi", "uvicorn", "pydantic", "streamlit", "langchain", "langchain-core", "langchain-community", "langchain-openai",
```


例：pyproject.tomlに下記のように記載される

```toml
dependencies = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "streamlit",
    "langchain",
    "langchain-core",
    "langchain-community",
    "langchain-openai",
]

[dependency-groups]
dev = [
    "anyio>=4.9.0",
    "black>=25.1.0",
    "coverage>=7.8.2",
    "mypy>=1.16.0",
    "pre-commit>=4.2.0",
    "pytest>=8.4.0",
    "ruff>=0.11.12",
]

```

package をもしマニュアルでいじったら uv sync
```bash
uv sync
Using CPython 3.12.9
Creating virtual environment at: .venv
Resolved 49 packages in 1.02s
Prepared 43 packages in 1.41s
Installed 46 packages in 100ms
 + altair==5.5.0
 + annotated-types==0.7.0
 + anyio==4.9.0
 + attrs==25.3.0
 + blinker==1.9.0
 + cachetools==5.5.2
 + certifi==2025.4.26
 + charset-normalizer==3.4.2
 + click==8.2.1
 + fastapi==0.115.12
 + gitdb==4.0.12
 + gitpython==3.1.44
 + h11==0.16.0
 + idna==3.10
 + jinja2==3.1.6
 + jsonschema==4.24.0
 + jsonschema-specifications==2025.4.1
 + markupsafe==3.0.2
 + narwhals==1.41.0
 + numpy==2.2.6
 + packaging==24.2
 + pandas==2.2.3
 + pillow==11.2.1
 + protobuf==6.31.1
 + pyarrow==20.0.0
 + pydantic==2.11.5
 + pydantic-core==2.33.2
 + pydeck==0.9.1
 + python-dateutil==2.9.0.post0
 + pytz==2025.2
 + referencing==0.36.2
 + requests==2.32.3
 + rpds-py==0.25.1
 + six==1.17.0
 + smmap==5.0.2
 + sniffio==1.3.1
 + starlette==0.46.2
 + streamlit==1.45.1
 + tenacity==9.1.2
 + toml==0.10.2
 + tornado==6.5.1
 + typing-extensions==4.14.0
 + typing-inspection==0.4.1
 + tzdata==2025.2
 + urllib3==2.4.0
 + uvicorn==0.34.3
```

### ターミナルで開発環境に入る場合

以下は任意

```bash
uv venv
. .venv/bin/activate.fish  
```

## 実行方法

```bash
uv streamlit run main.py   
```

## pre-commit の初回実行

.pre-commit-config.yaml を変更したら要チェック

```bash
 uv run pre-commit run --all-files
```

## TODO

次のステップ

### 2. PineconeやChromaと差し替えるためのAdapterクラス設計に拡張

### 3. Session単位のTTLや削除設計を追加してセッション管理強化
