# Environment Configuration Setup

Project đã được cấu hình với 2 cách để quản lý API keys:

## 1. File .env (Development)

File `.env` đã được tạo với nội dung:
```bash
GEMINI_API_KEY=AIzaSyAhmo11Er5YjU77_9y8FxBq_G0c38VO3NI
```

**Cách sử dụng:**
```bash
# Load .env và chạy app
set -a && source .env && set +a && streamlit run app_professional.py
```

hoặc đơn giản:
```bash
export GEMINI_API_KEY=AIzaSyAhmo11Er5YjU77_9y8FxBq_G0c38VO3NI
streamlit run app_professional.py
```

## 2. Streamlit Secrets (Production)

File `.streamlit/secrets.toml` đã được tạo:
```toml
GEMINI_API_KEY = "AIzaSyAhmo11Er5YjU77_9y8FxBq_G0c38VO3NI"
```

**Streamlit tự động load file này khi chạy app.**

## 3. File .env.example

File template `.env.example` được tạo để chia sẻ cấu trúc (không chứa API key thật).

## 🔒 Bảo mật

Đã thêm vào `.gitignore`:
- `.env`
- `.streamlit/secrets.toml`

**✅ An toàn:** API keys sẽ không bị commit lên GitHub!

## Chạy App

Bây giờ bạn có thể chạy đơn giản:
```bash
streamlit run app_professional.py
```

App sẽ tự động đọc API key từ:
1. Environment variable `GEMINI_API_KEY` (nếu có)
2. Hoặc từ `.streamlit/secrets.toml`
