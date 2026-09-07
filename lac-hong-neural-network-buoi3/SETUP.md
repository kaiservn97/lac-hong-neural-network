# HƯỚNG DẪN CÀI ĐẶT

Tài liệu này hướng dẫn cài đặt để chạy notebook trong VS Code.

## 1. Yêu cầu

- macOS, Linux hoặc Windows.
- Đã cài VS Code.
- Đã cài Python 3.11 hoặc 3.12 (khuyến nghị).

Lưu ý: Vì notebook hiện tại dùng scikit-learn, bạn không cần PyTorch.

## 2. Clone và mở project

```bash
git clone <repo-url>
cd lac-hong-neural-network
code .
```

## 3. Tạo virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 4. Cài package

```bash
pip install -r requirements.txt
```

## 5. Cấu hình VS Code

Project đã có sẵn file cấu hình trong thư mục `.vscode`:

- `.vscode/settings.json`: trỏ interpreter mặc định vào `.venv/bin/python`.
- `.vscode/extensions.json`: gợi ý extension Python, Pylance, Jupyter.

Nếu VS Code chưa nhận đúng interpreter:

1. Mở Command Palette.
2. Chọn Python: Select Interpreter.
3. Chọn interpreter trong `.venv`.

## 6. Chạy notebook

1. Mở file `notebooks/01_gia_nha_mlp.ipynb`.
2. Ở góc phải trên của notebook, chọn kernel là `.venv`.
3. Bấm Run All Cells (hoặc chạy từng ô từ trên xuống dưới).

## 7. Nếu gặp lỗi thường gặp

- Lỗi import pandas, sklearn, matplotlib:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

- VS Code vẫn báo lỗi import:
1. Python: Restart Language Server.
2. Developer: Reload Window.

- Kernel notebook không phải `.venv`:
1. Chọn lại kernel đúng `.venv`.
2. Chạy lại từ cell đầu.

## 8. Kiểm tra nhanh môi trường

```bash
python -c "import pandas, sklearn, matplotlib, jupyter; print('OK')"
```

Nếu in ra `OK` là môi trường đã sẵn sàng.
