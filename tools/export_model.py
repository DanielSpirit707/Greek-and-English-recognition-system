# -*- coding: utf-8 -*-  # 指定檔案編碼為 utf-8，確保程式碼能正確處理中文字符
"""  # 多行字串，做為此 Python 腳本的模組說明文件
把主專案訓練好的 PyTorch 模型匯出為 GitHub Pages 靜態版可用的格式。  # 說明此腳本的作用
會產生：  # 說明此腳本產生的檔案
  model/manifest.json — 類別清單、模型架構、權重索引  # 說明 manifest.json 的用途
  model/weights.bin   — 所有權重的連續二進位檔  # 說明 weights.bin 的用途
  MODEL_STATUS.md     — 人類可讀的目前模型版本資訊  # 說明 MODEL_STATUS.md 的用途
"""  # 結束模組說明文件

from __future__ import annotations  # 從 __future__ 匯入 annotations，允許在型別提示中使用推遲求值

import argparse  # 匯入 argparse 模組，用於解析命令列參數
import json  # 匯入 json 模組，用於讀寫 JSON 格式的資料
import shutil  # 匯入 shutil 模組，提供高階檔案操作
import struct  # 匯入 struct 模組，用於將 Python 資料打包成 C 語言相容的二進位格式
import sys  # 匯入 sys 模組，用於修改系統路徑
from datetime import datetime  # 從 datetime 模組匯入 datetime 類別，用於取得和格式化時間
from pathlib import Path  # 匯入 Path 類別，用於跨平台路徑操作

STATIC_ROOT = Path(__file__).resolve().parents[1]  # 取得此腳本往上兩層目錄的絕對路徑 (github_pages_static/)
MODEL_DIR = STATIC_ROOT / "model"  # 設定靜態版 model 資料夾的路徑

# ── 把主專案的 src/ 加入 Python 路徑，這樣才能匯入 config 和 models ──
PROJECT_ROOT = STATIC_ROOT.parent  # 取得整個專案的根目錄
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # 把 src/ 加入匯入搜尋路徑的最前面


def load_model_and_info(source_dir: Path):  # 定義函式，載入 PyTorch 模型與模型資訊
    """從來源資料夾載入 best_model.pth 與 model_info.json。"""  # 函式說明
    import torch  # 匯入 PyTorch (放在函式內部以加快沒有 torch 時的錯誤提示)
    from models import get_model  # 從 src/models.py 匯入 get_model 工廠函式
    from config import (  # 從 src/config.py 匯入需要的設定
        NUM_CLASSES, CLASS_NAMES, CLASS_SYMBOLS, CLASS_LANGUAGES,  # 類別相關設定
        DISPLAY_SYMBOL_BY_NAME, IMG_SIZE,  # 顯示符號與圖片大小
    )

    model_path = source_dir / "best_model.pth"  # 設定模型權重檔的路徑
    info_path = source_dir / "model_info.json"  # 設定模型資訊檔的路徑

    if not model_path.exists():  # 若模型檔不存在
        raise SystemExit(f"找不到模型檔: {model_path}")  # 結束程式並報錯
    if not info_path.exists():  # 若資訊檔不存在
        raise SystemExit(f"找不到模型資訊: {info_path}")  # 結束程式並報錯

    with open(info_path, "r", encoding="utf-8") as f:  # 以 UTF-8 開啟模型資訊檔
        model_info = json.load(f)  # 將 JSON 內容解析為 Python 字典

    model_name = model_info.get("model_name", "simple")  # 取得模型架構名稱，預設為 simple
    model = get_model(model_name, num_classes=NUM_CLASSES)  # 用工廠函式建立對應的模型物件
    state = torch.load(model_path, map_location="cpu", weights_only=True)  # 載入模型權重到 CPU
    model.load_state_dict(state)  # 將權重套用到模型物件上
    model.eval()  # 將模型設定為評估模式 (關閉 Dropout 與 BN 的訓練行為)

    # ── 組裝類別資訊列表 ──
    classes = []  # 建立空列表來存放類別資訊
    for name, symbol, language in zip(CLASS_NAMES, CLASS_SYMBOLS, CLASS_LANGUAGES):  # 同時遍歷名稱、符號、語言
        classes.append({  # 將每個類別包裝成字典並加入列表
            "name": name,  # 類別系統名稱
            "symbol": symbol,  # 類別對應的顯示符號
            "display_symbol": DISPLAY_SYMBOL_BY_NAME.get(name, symbol),  # 覆蓋後的顯示符號
            "language": language,  # 類別所屬語言 (greek 或 english)
        })

    return model, model_info, classes  # 回傳模型物件、模型資訊字典、類別列表


def export_weights(model, output_dir: Path):  # 定義函式，匯出模型權重為二進位格式
    """把模型 state_dict 匯出為 manifest.json + weights.bin。"""  # 函式說明
    import torch  # 匯入 PyTorch
    output_dir.mkdir(parents=True, exist_ok=True)  # 確保輸出目錄存在

    state_dict = model.state_dict()  # 取得模型的所有權重和 BN 統計量
    tensors_meta = {}  # 建立空字典，記錄每個張量的位置資訊
    all_bytes = bytearray()  # 建立空的位元組陣列，用來累積所有權重的二進位資料

    for name, param in state_dict.items():  # 遍歷 state_dict 中的每個張量
        arr = param.detach().cpu().float().numpy().flatten()  # 將張量轉成 CPU 上的 Float32 一維 numpy 陣列
        offset = len(all_bytes)  # 記錄這個張量在 weights.bin 中的起始位元組位置
        packed = struct.pack(f"<{len(arr)}f", *arr)  # 將陣列中的每個浮點數打包成小端序 float32 二進位
        all_bytes.extend(packed)  # 把這段二進位資料附加到總二進位陣列後面
        tensors_meta[name] = {  # 記錄該張量的中繼資訊
            "shape": list(param.shape),  # 張量原始的維度形狀
            "offset": offset,  # 在 weights.bin 中的位元組偏移量
            "length": len(arr),  # 元素數量 (不是位元組數)
        }

    weights_path = output_dir / "weights.bin"  # 設定 weights.bin 的完整路徑
    with open(weights_path, "wb") as f:  # 以二進位寫入模式開啟檔案
        f.write(all_bytes)  # 寫入全部權重的二進位資料

    return tensors_meta  # 回傳所有張量的中繼資訊字典


def write_manifest(tensors_meta, classes, model_info, output_dir: Path):  # 定義函式，寫出 manifest.json
    """組裝並寫出 manifest.json。"""  # 函式說明
    from config import IMG_SIZE  # 匯入圖片大小設定

    manifest = {  # 組裝完整的 manifest 字典
        "tensors": tensors_meta,  # 權重索引資訊
        "classes": classes,  # 類別清單
        "model_info": model_info,  # 模型訓練資訊
        "preprocess": {  # 前處理參數 (與 script.js 中的邏輯對齊)
            "img_size": IMG_SIZE,  # 模型輸入圖片大小 (28)
            "target_size": 24,  # 裁切後目標大小
            "threshold_min": 10,  # 二值化最小閾值
            "threshold_ratio": 0.15,  # 自適應閾值比率
            "zero_below": 80,  # 低於此值設為零
            "normalize_mean": 0.5,  # 正規化平均值
            "normalize_std": 0.5,  # 正規化標準差
        },
    }

    manifest_path = output_dir / "manifest.json"  # 設定 manifest.json 的完整路徑
    with open(manifest_path, "w", encoding="utf-8") as f:  # 以 UTF-8 寫入模式開啟檔案
        json.dump(manifest, f, ensure_ascii=False)  # 以 JSON 格式寫出，允許 Unicode 字元直接顯示

    return manifest_path  # 回傳檔案路徑


def write_model_status(model_info, output_dir: Path):  # 定義函式，產生人類可讀的模型狀態文件
    """在目標資料夾產生 MODEL_STATUS.md，記錄目前使用的模型版本。"""  # 函式說明
    timestamp = model_info.get("timestamp", "unknown")  # 取得模型的時間戳記
    date = model_info.get("date", "unknown")  # 取得模型的日期
    val_acc = model_info.get("best_val_acc", 0)  # 取得最佳驗證準確率
    test_acc = model_info.get("test_acc", 0)  # 取得測試準確率
    epochs = model_info.get("total_epochs", 0)  # 取得訓練輪數
    train_samples = model_info.get("train_samples", 0)  # 取得訓練樣本數
    model_name = model_info.get("model_name", "unknown")  # 取得模型架構名稱
    training_mode = model_info.get("training_mode", "unknown")  # 取得訓練模式

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 取得目前時間並格式化

    content = f"""# 目前模型版本

> 此檔案由 `tools/export_model.py` 自動產生，記錄目前部署的模型資訊。

| 項目 | 內容 |
|---|---|
| 模型時間戳記 | `{timestamp}` |
| 訓練日期 | {date} |
| 模型架構 | {model_name} |
| 訓練模式 | {training_mode} |
| 訓練樣本數 | {train_samples:,} |
| 訓練輪數 | {epochs} |
| 最佳驗證準確率 | {val_acc:.2%} |
| 測試準確率 | {test_acc:.2%} |
| 匯出時間 | {now} |

## 如何更新

```powershell
$env:PYTHONIOENCODING='utf-8'; python github_pages_static/tools/export_model.py
```

更新後請將變更推送到 GitHub，GitHub Pages 會自動部署新版本。
"""

    status_path = output_dir / "MODEL_STATUS.md"  # 設定 MODEL_STATUS.md 的完整路徑
    with open(status_path, "w", encoding="utf-8") as f:  # 以 UTF-8 寫入模式開啟檔案
        f.write(content)  # 寫入內容

    return status_path  # 回傳檔案路徑


def update_readme(model_info, readme_path: Path):  # 定義函式，更新 README.md 中的模型資訊
    """更新 README.md 中的模型資訊區塊。"""  # 函式說明
    if not readme_path.exists():  # 如果 README.md 不存在
        return  # 直接回傳，不做任何事

    text = readme_path.read_text(encoding="utf-8")  # 讀取 README.md 的全部內容
    timestamp = model_info.get("timestamp", "unknown")  # 取得模型時間戳記
    val_acc = model_info.get("best_val_acc", 0)  # 取得最佳驗證準確率
    num_classes = model_info.get("num_classes", 50)  # 取得類別數量

    # ── 尋找並替換 "目前內建模型" 區塊 ──
    marker = "目前內建模型"  # 設定搜尋標記
    if marker in text:  # 如果 README 中有這個標記
        lines = text.split("\n")  # 將文字按行分割
        new_lines = []  # 建立新的行列表
        skip = False  # 用來標記是否正在跳過舊的模型資訊
        for line in lines:  # 遍歷每一行
            if marker in line:  # 如果找到標記行
                new_lines.append(line)  # 保留標記行
                new_lines.append("")  # 空行
                new_lines.append(f"- timestamp: `{timestamp}`")  # 寫入新的時間戳記
                new_lines.append(f"- best validation accuracy: `{val_acc:.2%}`")  # 寫入新的驗證準確率
                new_lines.append(f"- classes: `{num_classes}`")  # 寫入新的類別數
                skip = True  # 開始跳過舊的資訊行
            elif skip:  # 如果正在跳過
                if line.startswith("- ") or line.strip() == "":  # 如果還在舊的模型資訊區塊內
                    continue  # 跳過這一行
                else:  # 否則已經超出舊區塊
                    skip = False  # 停止跳過
                    new_lines.append(line)  # 正常加入新行
            else:  # 不在標記區塊內
                new_lines.append(line)  # 正常加入新行

        readme_path.write_text("\n".join(new_lines), encoding="utf-8")  # 將修改後的文字寫回 README.md


def main():  # 定義主程式函式
    parser = argparse.ArgumentParser(  # 建立參數解析器
        description="匯出 PyTorch 模型到 GitHub Pages 靜態版格式"  # 設定說明文字
    )
    parser.add_argument(  # 新增 --source 參數
        "--source",  # 參數名稱
        default=str((PROJECT_ROOT / "models").resolve()),  # 預設為主專案的 models 資料夾
        help="來源 models 資料夾，預設為主專案根目錄下的 models",  # 說明文字
    )
    args = parser.parse_args()  # 解析命令列參數

    source_dir = Path(args.source).resolve()  # 將來源路徑轉為絕對路徑
    if not source_dir.exists():  # 如果來源目錄不存在
        raise SystemExit(f"找不到來源資料夾: {source_dir}")  # 結束程式並報錯

    print(f"來源: {source_dir}")  # 印出來源路徑
    print(f"目標: {STATIC_ROOT}")  # 印出目標路徑
    print()  # 印出空行

    # 1. 載入模型
    print("載入模型...")  # 提示使用者
    model, model_info, classes = load_model_and_info(source_dir)  # 載入模型、資訊、類別
    print(f"  模型架構: {model_info.get('model_name', 'unknown')}")  # 印出模型架構
    print(f"  時間戳記: {model_info.get('timestamp', 'unknown')}")  # 印出時間戳記

    # 2. 匯出權重
    print("匯出權重...")  # 提示使用者
    tensors_meta = export_weights(model, MODEL_DIR)  # 匯出 weights.bin 並取得張量中繼資訊
    print(f"  張量數: {len(tensors_meta)}")  # 印出張量數量

    # 3. 寫出 manifest.json
    print("寫出 manifest.json...")  # 提示使用者
    manifest_path = write_manifest(tensors_meta, classes, model_info, MODEL_DIR)  # 寫出 manifest
    print(f"  {manifest_path}")  # 印出檔案路徑

    # 4. 產生 MODEL_STATUS.md
    print("產生 MODEL_STATUS.md...")  # 提示使用者
    status_path = write_model_status(model_info, STATIC_ROOT)  # 產生模型狀態文件
    print(f"  {status_path}")  # 印出檔案路徑

    # 5. 更新 README.md
    print("更新 README.md...")  # 提示使用者
    update_readme(model_info, STATIC_ROOT / "README.md")  # 更新 README 中的模型資訊

    print()  # 印出空行
    print("匯出完成！請將變更 push 到 GitHub 以更新 GitHub Pages。")  # 提示使用者完成


if __name__ == "__main__":  # 如果是直接執行此腳本
    main()  # 呼叫主函式
