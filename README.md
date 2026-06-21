# GreekLens GitHub Pages Static Build

這個資料夾是純靜態版本，可以直接放到 GitHub Pages。

它不需要 Flask、Render、Python、PyTorch 伺服器，也不會有雲端閒置冷啟動問題。模型會在使用者的瀏覽器中載入並推論。

## 檔案內容

- `index.html`：網站頁面
- `style.css`：樣式
- `script.js`：Canvas 繪圖、前處理、CNN 推論、結果顯示
- `model/manifest.json`：類別資訊與權重索引
- `model/weights.bin`：已固定的最佳模型權重
- `.nojekyll`：讓 GitHub Pages 直接提供所有靜態檔案

## 使用方式

1. 建立 GitHub repository。
2. 把這個資料夾內的所有檔案放到 repository 根目錄。
3. 到 repository 的 `Settings` -> `Pages`。
4. Source 選 `Deploy from a branch`。
5. Branch 選 `main`，資料夾選 `/root`。
6. 儲存後等待 GitHub Pages 產生網址。

完成後，使用者只要點 GitHub Pages 網址就能直接使用。

## 注意

直接雙擊 `index.html` 用 `file://` 開啟時，瀏覽器可能會阻擋 `fetch()` 載入模型檔。放到 GitHub Pages 後會正常。

目前內建模型：

- timestamp: `20260619_231452`
- best validation accuracy: `96.76%`
- classes: `50`