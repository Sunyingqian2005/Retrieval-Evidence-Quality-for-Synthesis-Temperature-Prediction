# GitHub 上传步骤

## 上传前检查

1. 打开 `CITATION.cff`，确认 GitHub 用户名为 `Sunyingqian2005`。
2. 检查作者姓名和 MIT License 是否得到合作者同意。
3. 不要把完整数据、API key 或 `outputs/` 上传。
4. 在仓库根目录运行：

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
git status --short
```

## 在 GitHub 网页创建空仓库

1. 登录 GitHub，点击右上角 `+` → `New repository`。
2. Repository name 填 `Retrieval-Evidence-Quality-for-Synthesis-Temperature-Prediction`。
3. 投稿前建议先选 `Private`；合作者核对完成后再改为 `Public`。
4. 不要勾选网页端的 README、`.gitignore` 或 License，因为本地已有。
5. 点击 `Create repository`，复制 GitHub 给出的 HTTPS 地址。

## 从终端上传

进入本目录：

```bash
cd "/path/to/Retrieval-Evidence-Quality-for-Synthesis-Temperature-Prediction"
```

首次上传：

```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial reproducibility release"
git remote add origin https://github.com/Sunyingqian2005/Retrieval-Evidence-Quality-for-Synthesis-Temperature-Prediction.git
git push -u origin main
```

如果 GitHub 要求登录，按提示使用浏览器授权或 Personal Access Token；
GitHub 账户密码不能再直接作为命令行密码使用。

## 上传后核对

1. 在 GitHub 网页确认 `data/full/`、`.env`、`outputs/` 都不存在。
2. 打开 `README.md`，检查命令和目录树是否正常显示。
3. 点击仓库右侧 `Releases` → `Create a new release`。
4. Tag 建议用 `v1.0.0`，标题可写 `Manuscript submission release`。
5. 若文章返修时改代码，使用新 tag（例如 `v1.0.1`），不要覆盖旧 release。

## 生成永久 DOI（推荐）

1. 用 GitHub 账户登录 Zenodo。
2. 在 Zenodo 的 GitHub 设置中启用该仓库。
3. 回到 GitHub 创建 release；Zenodo 会归档该 release 并生成 DOI。
4. 将 DOI 回填到 `CITATION.cff` 和论文 Data and Software Availability。
5. 再发布一个只修正元数据的小版本，使归档内容与论文链接一致。
