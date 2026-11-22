# hakken-skill

論文 **[Hakken: Predicting future discoveries to fill the gaps in today's knowledge](https://arxiv.org/abs/2609.04494)**
(Besold et al., Sony AI, [arXiv:2609.04494](https://arxiv.org/abs/2609.04494), 2026) の方法論を、**どの研究分野にも適用できる
エージェントスキル**として実装したものです。Claude Code、Codex、Cursor、Gemini CLI、
OpenCode など、[`skills` CLI](https://github.com/vercel-labs/skills) が対応するエージェントに
`pnpx skills` でインストールできます。

- **GPU も学習基盤も不要**です。Python 3.8 以上の標準ライブラリだけで動きます。
- **推論中心の構成**です。グラフ側の予測は、時系列ネットワーク特徴量のロジスティックモデルを、そのグラフ自身の過去データから数秒で当てはめます。言語側の評価は、エージェント自身またはそのサブエージェントが担います。
- **知識データは利用者のプロジェクト内の作業領域（`hakken/`）で管理**します。スキルのディレクトリには一切書き込みません。

## 論文との対応

| Hakken | このスキル |
|---|---|
| 時系列知識グラフ（文献から抽出した三つ組 + 初出年） | 作業領域の台帳 `hakken/ledger/observations.csv`（追記のみ、出典と抽出者を記録） |
| THiGER のグラフブランチ（GraphSAGE + 時系列 Transformer） | 時系列ネットワーク特徴量 + 関係パターン特徴量のロジスティックモデル（Science4Cast で特徴量ベースの方法が最良だった知見に基づく） |
| THiGERLLM の言語ブランチ（Mistral-7B） | エージェントとサブエージェント（`judge-queue` → `record` → `fuse`）。証拠は概念ごとに分けて、カットオフ以前の文献から集める |
| 関係ラベルごとのアンサンブルとキャリブレーション | `fuse`（logit を重み付きで合成） |
| PHELInE（十分性・必要性による説明） | `explain`（モデルに直接問い合わせて計算するので、代理モデルの再学習は不要） |
| 仮説選定パイプライン | `novelty` + `select` |
| 過去データでの検証 | `backtest`（null ベースラインとの比較、区間ごとの再現率、nDCG） |

フルモデルである [SonyResearch/HakkenOSS](https://github.com/SonyResearch/HakkenOSS) は、
学習済みの重みが公開されておらず LLM ブランチも含まれていません。また学習には GPU、PyG、
ZenML、S3 が必要です。このためこのスキルには取り込んでいません。将来重みが公開されたときは、
`predict --import-scores` で外部モデルのスコアを取り込めます。

## インストール

```bash
pnpx skills add HyodaKazuaki/hakken-skill                 # GitHub から
pnpx skills add ./hakken-skill                            # ローカルのクローンから
pnpx skills add HyodaKazuaki/hakken-skill --list          # 中身の確認だけ
pnpx skills add HyodaKazuaki/hakken-skill --skill hakken-discovery -a claude-code -a codex -g -y
```

## 使い方

インストール後、エージェントにこう頼むだけで使えます。

- 「老化関連遺伝子について、まだ報告されていない関係の仮説を Hakken 方式で出して」
- 「ペロブスカイト太陽電池の安定性で、次に見つかりそうな材料と特性の関係を予測して」
- 「この triples.csv を取り込んで、2012年をカットオフにしたバックテストをして」

コマンドを直接使う場合（`H` は `python3 <skill>/scripts/hakken.py` の略）:

```bash
H init --goal "aging genes"                    # ./hakken/ を作成
H fetch-pubtator --query @GENE_SIRT1 --year-from 1995 --year-to 2025 --per-year 30
H stats
H backtest --cutoff 2014                       # 特徴量モデルとヒューリスティックの両方を null と比較
H predict                                      # runs/<id>/predictions.csv
H novelty --top 200 --mark-pending             # 台帳での新規性確認。外部検索は承認があるときだけ
H judge-queue --top 50                         # サブエージェント用の判定バッチを出力
H record judgments out.jsonl && H fuse
H select && H explain && H report              # runs/<id>/report.md
```

`examples/toy_triples.csv` は、将来の発見が起きる仕組みを埋め込んだ合成データです。
`H ingest examples/toy_triples.csv --extractor synthetic` で取り込めば、すぐに一連の流れを試せます。

## 作業領域とポリシー

```
hakken/
├── hakken.json         目的、時間単位、git ポリシー、外部検索ポリシー
├── config/             関係の定義、同義語、除外するハブ、シード（人が編集する）
├── ledger/             正本となる観測の台帳（追記のみ）
├── judgments/          言語ブランチの判定（追記のみ）
├── novelty/            新規性チェックの記録（保留中のものを含む）
└── runs/<id>/          派生物（manifest.json があり、再現できる）
```

- **git ポリシー**：既定は `private` で、設定と台帳だけを git で管理します。生成した仮説は未発表の研究アイデアなので、外に出さない設定にしています。
- **外部検索ポリシー**：概念の組み合わせを外部で検索すると、仮説の内容が第三者に伝わります。そのため既定は `ask` です。対話中なら利用者の承認を得てから実行します。自律モードやヘッドレス実行では検索せずに「保留」として記録し、レポートに一覧を残します。
- キャッシュは `~/.cache/hakken` に保存されます。

詳しくは `skills/hakken-discovery/references/workspace.md` を参照してください。

## 構成

```
skills/hakken-discovery/
├── SKILL.md                    エージェント向けの手順（ワークフロー、動作モードごとの方針、ガードレール）
├── scripts/hakken.py           CLI の入口
├── scripts/hakkenlib/          標準ライブラリのみの実装（graph, models, sources, workspace, cli）
├── references/                 workspace / language-branch / method / data-pipeline /
│                               domain-adaptation / evaluation / report-template
└── examples/                   toy_triples.csv, make_toy_data.py,
                                biomedical_relations.json（論文 Table E2/E3）, pubtator_relations.json
```

## 注意

- スコアは「既存文献がその関係をどれだけ支持しているか」を表すもので、関係が真である確率ではありません。未観測の関係は「偽」ではなく「未ラベル」として扱います。
- グラフ側のモデルは THiGERLLM の代わりとなる特徴量モデルです。論文に載っている性能をそのまま期待しないでください。
- LLM の事前学習データに将来の文献が含まれている場合、情報の漏れを完全には防げません（`leak_suspected` フラグで検出し、`fuse` で除外します）。

## 参考文献

- T. R. Besold, U. Akujuobi, P. Sanchez, et al. "Hakken: Predicting future discoveries to fill the gaps in today's knowledge." arXiv:2609.04494, 2026. <https://arxiv.org/abs/2609.04494>
- M. Krenn, L. Buffoni, B. Coutinho, et al. "Predicting the Future of AI with AI: High-quality link prediction in an exponentially growing knowledge network." arXiv:2210.00881, 2022. <https://arxiv.org/abs/2210.00881>（Science4Cast。特徴量ベースのグラフ側モデルの根拠）
- X. Gu, M. Krenn. "Forecasting high-impact research topics via machine learning on evolving knowledge graphs." arXiv:2402.08640, 2024. <https://arxiv.org/abs/2402.08640>
- SonyResearch/HakkenOSS（論文著者による公開実装、BSD-3-Clause）: <https://github.com/SonyResearch/HakkenOSS>
- PubTator3: <https://www.ncbi.nlm.nih.gov/research/pubtator3/>

このリポジトリは論文著者や Sony とは関係のない独立した実装です。

## License

MIT（スキルとコード）。論文の内容は原著者に帰属します。
