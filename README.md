# hakken-skill

論文 **"Hakken: Predicting future discoveries to fill the gaps in today's knowledge"**
(Besold et al., Sony AI, arXiv:2609.04494, 2026) の方法論を、**どの研究分野にも適用できる
エージェントスキル**として実装したものです。Claude Code、Codex、Cursor、Gemini CLI、
OpenCode など、[`skills` CLI](https://github.com/vercel-labs/skills) が対応するエージェントに
`pnpx skills` でインストールできます。

## 何ができるか

研究テーマを与えると、エージェントが次のサイクルを回します。

1. **Frame**: 目的・エンティティ型・関係語彙・時間単位・シード集合を決める
2. **Build**: 文献から `(subject, relation, object, 初出年)` の時系列知識グラフを構築
3. **Clean**: 同義語統合、受動態関係の反転、極性矛盾（増加/減少）の解消、初出のみ保持
4. **Validate**: 過去のカットオフで「その後10年の発見」をどれだけ事前に当てられたかをバックテスト（ランダム経験スコアの null ベースラインと比較）
5. **Predict**: まだ報告されていない関係を予測し、直近ウィンドウでキャリブレーション
6. **Fuse**: エージェント自身が論文の「言語ブランチ」役として意味的妥当性を評価し、アンサンブル
7. **Select**: 新規性フィルタ、エンティティごと top-k、recency、最短パス長で絞り込み
8. **Explain**: PHELInE 方式で根拠パスを列挙し、十分性・必要性スコアと多様性で再ランク
9. **Report**: 仮説カード（根拠・リスク・最小検証実験デザイン）を出力

同梱の `scripts/hakken_kg.py` は **Python 標準ライブラリのみ**で動く軽量実装です
（`stats / clean / predict / backtest / explain / select`）。論文の THiGERLLM（GNN + LLM）
そのものではなく、同じ帰納バイアス（時間的なエビデンス蓄積、マルチホップ構造、関係タイプの
パターン、positive–unlabeled 設定）を持つ透明な代理モデルです。言語ブランチはエージェントが担います。
フルモデルは論文が公開先として挙げている
[SonyResearch/HakkenOSS](https://github.com/SonyResearch/HakkenOSS) を参照してください。

## インストール

```bash
# GitHub に push した後（<owner> は自分のアカウント名）
pnpx skills add <owner>/hakken-skill

# ローカルのクローンから
pnpx skills add ./hakken-skill

# 中身の確認だけ
pnpx skills add ./hakken-skill --list

# 対象エージェントを指定して非対話インストール（例: Claude Code と Codex、ユーザー全体）
pnpx skills add <owner>/hakken-skill --skill hakken-discovery -a claude-code -a codex -g -y
```

`npx skills add ...` / `bunx skills add ...` でも同じです。

## 使い方の例

インストール後、エージェントにこう頼むだけで起動します。

- 「老化関連遺伝子の間で、まだ報告されていない制御関係の仮説を Hakken 方式で出して」
- 「ペロブスカイト太陽電池の安定性について、次に見つかりそうな材料–特性の関係を予測して」
- 「この triples.csv で 2010 年カットオフのバックテストをして、null と比較して」
- 「(TP53, AFFECTS_EXPRESSION_OF, BAMBI) の根拠パスを説明して」

スクリプト単体でも使えます。

```bash
cd skills/hakken-discovery
python3 scripts/hakken_kg.py backtest examples/toy_triples.csv --cutoff 2008
python3 scripts/hakken_kg.py predict  examples/toy_triples.csv --cutoff 2015 --calibrate 5 -o pred.csv
python3 scripts/hakken_kg.py select   pred.csv --triples examples/toy_triples.csv --cutoff 2015 --top-k 3
python3 scripts/hakken_kg.py explain  examples/toy_triples.csv -s GENE08 -r IS_A_BIOMARKER_FOR -O DISEASE03 --cutoff 2015 --latent
```

## 構成

```
skills/hakken-discovery/
├── SKILL.md                     # エージェント向け手順（9ステップのワークフローとガードレール）
├── scripts/hakken_kg.py         # 依存ゼロのツールキット
├── references/
│   ├── method.md                # 論文の詳細解説（THiGERLLM / PHELInE / 選定 / 結果 / 限界）
│   ├── data-pipeline.md         # データソース、抽出プロンプト、clean 設定の書式
│   ├── domain-adaptation.md     # 生物医学・材料・化学・農学・生態・社会科学・CS などの関係スキーマ
│   ├── evaluation.md            # 時系列分割、指標、null ベースライン、結果の読み方
│   └── report-template.md       # 最終レポートと仮説カードの雛形
└── examples/
    ├── toy_triples.csv          # 発見メカニズムを埋め込んだ合成データ
    ├── make_toy_data.py         # その生成スクリプト
    └── biomedical_clean_config.json  # 論文 Table E2/E3 に基づく clean 設定
```

## 注意

- スコアは「既存文献がその関係をどれだけ支持しているか」であり、真である確率ではありません。
- 未観測の関係は「偽」ではなく「未ラベル」です。精度（precision）は評価に使いません。
- LLM の事前学習データによる将来情報のリークは完全には排除できません。

## License

MIT（スキルとコード）。論文の内容は原著者に帰属します。
