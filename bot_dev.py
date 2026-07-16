import time
import requests
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.genai as genai
from google.genai import types


# ==========================================
# ⚙️ تنظیمات (CONFIG) - مقادیر از فایل .env خوانده می‌شوند
# ==========================================
load_dotenv()

BALE_TOKEN = os.getenv("BALE_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.prox.us.ci/v1")

if not BALE_TOKEN or not GEMINI_API_KEY or not OPENAI_API_KEY:
    raise ValueError("توکن‌های ضروری در فایل .env تنظیم نشده‌اند.")

BALE_API_URL = f"https://tapi.bale.ai/bot{BALE_TOKEN}/"


SYSTEM_PROMPT = (
    "تو یک استاد دانشگاه و متخصص علوم و مهندسی هستی. تصویر ارسالی یک سوال امتحانی است."
    "سوال را به دقت تحلیل کن و پاسخ را کاملاً گام‌به‌گام، تشریحی و با فرمول‌های دقیق بنویس"
    "مهم: بدون هیچ توضیح اضافه در خروجی فقط جواب سوال رو بنویس" 
)

# ==========================================
# 🧠 توابع هوش مصنوعی
# ==========================================
def ask_gemini(image_bytes):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, SYSTEM_PROMPT]
        )
        return response.text # 🟢 حذف متد replace برای جلوگیری از خرابی کدهای ریاضی
    except Exception as e:
        return f"خطا GGGGGدر دریافت پاسخ از جمنای: {str(e)}"

def ask_chatgpt(image_bytes):
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        
        response = client.chat.completions.create(
            model="gpt-5.5-openai-compact", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content # 🟢 حذف متد replace برای جلوگیری از خرابی کدهای ریاضی
    except Exception as e:
        return f"خطا در دریافت پاسخ از چت‌جی‌پتی: {str(e)}"

# ==========================================
# 🌐 تابع ساخت فایل HTML گرافیکی
# ==========================================
import re
import markdown

_MATH_PLACEHOLDER = "MATHJAXBLOCK{idx}END"

# الگوهای رایج LaTeX — ChatGPT اغلب \( \) و \[ \] می‌فرستد که markdown خراب می‌کند
_MATH_PATTERNS = [
    (r"\$\$([\s\S]+?)\$\$", "display"),
    (r"\\\[([\s\S]+?)\\\]", "display"),
    (
        r"\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|cases)\}"
        r"([\s\S]+?)\\end\{\1\}",
        "env",
    ),
    (r"```(?:latex|math)?\s*\n([\s\S]+?)```", "display"),
    (r"\\\(([\s\S]+?)\\\)", "inline"),
    (r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", "inline"),
]


def _stash_math(match, kind, placeholders):
    idx = len(placeholders)
    if kind == "env":
        env_name = match.group(1)
        body = match.group(2).strip()
        tex = f"$$\n\\begin{{{env_name}}}\n{body}\n\\end{{{env_name}}}\n$$"
    elif kind == "display":
        tex = f"$$\n{match.group(1).strip()}\n$$"
    else:
        tex = f"${match.group(1).strip()}$"
    placeholders.append(tex)
    return _MATH_PLACEHOLDER.format(idx=idx)


def _protect_math(text):
    placeholders = []
    for pattern, kind in _MATH_PATTERNS:
        text = re.sub(
            pattern,
            lambda m, k=kind: _stash_math(m, k, placeholders),
            text,
        )
    return text, placeholders


def _restore_math(html, placeholders):
    for idx, tex in enumerate(placeholders):
        html = html.replace(_MATH_PLACEHOLDER.format(idx=idx), tex)
    return html


def markdown_with_math(text):
    """فرمول‌ها را قبل از markdown محافظت می‌کند تا _ و \\ خراب نشوند."""
    protected, placeholders = _protect_math(text)
    html = markdown.markdown(protected, extensions=["nl2br"])
    return _restore_math(html, placeholders)


def create_html_report(gemini_ans, gpt_ans, output_filename="answer.html"):
    gemini_html = markdown_with_math(gemini_ans)
    gpt_html = markdown_with_math(gpt_ans)

    html_content = f"""
    <!DOCTYPE html>
    <html dir="rtl" lang="fa">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>پاسخ‌نامه هوش مصنوعی</title>
        
        <script>
        window.MathJax = {{
          tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true,
            packages: {{'[+]': ['ams']}}
          }},
          options: {{
            renderActions: {{
              addMenu: [0, '', '']
            }}
          }}
        }};
        </script>
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        
        <style>
            body {{ font-family: 'Tahoma', 'Arial', sans-serif; line-height: 1.8; padding: 20px; background-color: #f1f5f9; color: #1e293b; direction: rtl; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h2 {{ text-align: center; color: #1e3a8a; border-bottom: 3px solid #3b82f6; padding-bottom: 12px; margin-bottom: 30px; }}
            .columns {{
                display: flex;
                gap: 20px;
                align-items: stretch;
            }}
            .box {{
                flex: 1;
                min-width: 0;
                border: 1px solid #e2e8f0;
                padding: 20px;
                border-radius: 12px;
                background: white;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            .gemini {{ border-right: 6px solid #1a73e8; }}
            .gpt {{ border-right: 6px solid #10a37f; }}
            h3 {{ margin-top: 0; margin-bottom: 15px; font-size: 15pt; display: flex; align-items: center; }}
            .gemini h3 {{ color: #1a73e8; }}
            .gpt h3 {{ color: #10a37f; }}
            
            .content-area {{ 
                white-space: pre-wrap; 
                word-wrap: break-word; 
                font-size: 11pt;
            }}
            .content-area ul, .content-area ol {{
                padding-right: 20px;
                margin-top: 10px;
                margin-bottom: 10px;
            }}
            .content-area li {{
                margin-bottom: 8px;
            }}
            @media (max-width: 900px) {{
                .columns {{
                    flex-direction: column;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📄 پاسخ‌نامه تشریحی هوش مصنوعی</h2>
            <div class="columns">
                <div class="box gemini">
                    <h3>🤖 پاسخ مدل Gemini (Google)</h3>
                    <div class="content-area">{gemini_html}</div>
                </div>
                <div class="box gpt">
                    <h3>🧠 پاسخ مدل ChatGPT (OpenAI)</h3>
                    <div class="content-area">{gpt_html}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

# ==========================================
# 🤖 توابع ارتباط با بله
# ==========================================
def send_message(chat_id, text):
    requests.post(f"{BALE_API_URL}sendMessage", json={"chat_id": chat_id, "text": text})

def send_html_file(chat_id, filepath):
    with open(filepath, 'rb') as f:
        requests.post(
            f"{BALE_API_URL}sendDocument",
            data={"chat_id": chat_id, "caption": "🌐 پاسخ سوال شما آماده شد! فایل HTML بالا را باز کنید."},
            files={"document": f}
        )

def process_updates(updates):
    for update in updates:
        if 'message' not in update:
            continue
            
        message = update['message']
        chat_id = message.get('chat', {}).get('id')
        
        if not chat_id: 
            continue

        # اگر کاربر عکس فرستاد
        if 'photo' in message:
            send_message(chat_id, "⏳ تصویر دریافت شد. در حال پردازش... لطفاً صبور باشید.")
            print(f"📸 Photo received from user: {chat_id}")
            
            # ۱. گرفتن بالاترین کیفیت عکس
            file_id = message['photo'][-1]['file_id']
            
            # ۲. درخواست به بله برای گرفتن مسیر فایل
            file_info_res = requests.get(f"{BALE_API_URL}getFile", params={"file_id": file_id})
            file_info = file_info_res.json()
            
            if not file_info.get('ok'):
                print(f"❌ Failed to get file info from Bale: {file_info}")
                continue
                
            file_path = file_info['result']['file_path']
            
            # ۳. دانلود بایت‌های واقعی عکس از سرور بله
            download_url = f"https://tapi.bale.ai/file/bot{BALE_TOKEN}/{file_path}"
            img_res = requests.get(download_url)
            
            # مطمئن می‌شویم دانلود با موفقیت انجام شده است
            if img_res.status_code != 200:
                print(f"❌ Failed to download photo from Bale server. Status code: {img_res.status_code}")
                continue
                
            img_data = img_res.content
            print(f"🔹 Photo downloaded successfully. File size: {len(img_data)} bytes")

            # اگر حجم فایل کمتر از ۱ کیلوبایت باشد یعنی عکس نیست و متن خطاست
            if len(img_data) < 1000:
                print(f"⚠️ Warning: File size is very small! Downloaded content may be an error: {img_data[:100]}")
            
            # ۴. ارسال به هوش‌های مصنوعی
            print("→ Sending to Gemini...")
            gemini_ans = ask_gemini(img_data)
            
            print("→ Sending to ChatGPT...")
            gpt_ans = ask_chatgpt(img_data)
            
            # ۵. ساخت فایل HTML
            html_name = f"answer_{chat_id}.html"
            create_html_report(gemini_ans, gpt_ans, html_name)
            
            # ۶. ارسال فایل نهایی به بله
            send_html_file(chat_id, html_name)
            print("✅ HTML file sent to user.")
            
            if os.path.exists(html_name): 
                os.remove(html_name)

        # اگر کاربر متن فرستاد
        elif 'text' in message:
            send_message(chat_id, "سلام! 👋 لطفاً از سوال امتحانی خود یک عکس واضح بفرستید.")

# ==========================================
# 🚀 حلقه اصلی اجرا (Polling)
# ==========================================
def main():
    print("🤖 Bale bot is running (no PDF engine required)...")
    last_update_id = 0
    while True:
        try:
            res = requests.get(f"{BALE_API_URL}getUpdates", params={"offset": last_update_id + 1, "timeout": 10})
            if res.status_code == 200:
                data = res.json()
                if data['ok'] and data['result']:
                    process_updates(data['result'])
                    last_update_id = data['result'][-1]['update_id']
        except Exception as e:
            print(f"⚠️ Network error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()