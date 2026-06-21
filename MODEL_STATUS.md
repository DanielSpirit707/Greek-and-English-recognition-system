# 目前模型版本

> 此檔案由 `tools/export_model.py` 自動產生，記錄目前部署的模型資訊。

| 項目 | 內容 |
|---|---|
| 模型時間戳記 | `20260619_231452` |
| 訓練日期 | 2026-06-19 23:14:52 |
| 模型架構 | simple |
| 訓練模式 | full |
| 訓練樣本數 | 126,327 |
| 訓練輪數 | 45 |
| 最佳驗證準確率 | 96.76% |
| 測試準確率 | 96.43% |
| 匯出時間 | 2026-06-22 07:08:42 |

## 如何更新

```powershell
$env:PYTHONIOENCODING='utf-8'; python github_pages_static/tools/export_model.py
```

更新後請將變更推送到 GitHub，GitHub Pages 會自動部署新版本。
