import time
import traceback
import requests
import base64
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# ⚙️ تنظیمات (CONFIG) - مقادیر از فایل .env خوانده می‌شوند
# ==========================================
load_dotenv()

BALE_TOKEN = os.getenv("BALE_TOKEN")
COMETAPI_KEY = os.getenv("COMETAPI_KEY")
COMETAPI_BASE_URL = os.getenv("COMETAPI_BASE_URL", "https://api.cometapi.com/v1")
COMETAPI_GPT_MODEL = os.getenv("COMETAPI_GPT_MODEL", "gpt-5.6-sol")
COMETAPI_GEMINI_MODEL = os.getenv("COMETAPI_GEMINI_MODEL", "gemini-3-flash")

if not BALE_TOKEN or not COMETAPI_KEY:
    raise ValueError("توکن‌های ضروری در فایل .env تنظیم نشده‌اند.")

BALE_API_URL = f"https://tapi.bale.ai/bot{BALE_TOKEN}/"

print(f"🔑 BALE_TOKEN loaded: {'YES - ' + BALE_TOKEN[:6] + '...' + BALE_TOKEN[-4:] if BALE_TOKEN else 'NO (empty/None)'}")
print(f"🔑 COMETAPI_KEY loaded: {'YES - ' + COMETAPI_KEY[:8] + '...' if COMETAPI_KEY else 'NO (empty/None)'}")
print(f"🌐 BALE_API_URL: {BALE_API_URL}")

# یک کلاینت مشترک برای هر دو مدل، چون هر دو از طریق CometAPI صدا زده می‌شوند
comet_client = OpenAI(api_key=COMETAPI_KEY, base_url=COMETAPI_BASE_URL)


SYSTEM_PROMPT = (
    "تو یک استاد دانشگاه و متخصص علوم و مهندسی هستی. تصویر ارسالی یک سوال امتحانی است."
    "سوال را به دقت تحلیل کن و پاسخ را کاملاً گام‌به‌گام، تشریحی و با فرمول‌های دقیق بنویس"
    "مهم: بدون هیچ توضیح اضافه در خروجی فقط جواب سوال رو بنویس"
)

# ==========================================
# 🧠 توابع هوش مصنوعی
# ==========================================
def ask_gemini(image_bytes):
    """
    جمنای هم از طریق CometAPI (اندپوینت chat.completions سازگار با OpenAI) صدا زده می‌شود.
    مستندات: https://apidoc.cometapi.com/
    """
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        response = comet_client.chat.completions.create(
            model=COMETAPI_GEMINI_MODEL,
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
        return response.choices[0].message.content  # 🟢 حذف متد replace برای جلوگیری از خرابی کدهای ریاضی
    except Exception as e:
        return f"خطا در دریافت پاسخ از جمنای (CometAPI): {str(e)}"


def ask_chatgpt(image_bytes):
    """
    از CometAPI (سازگار با OpenAI Responses API) برای تحلیل تصویر استفاده می‌کند.
    مستندات: https://apidoc.cometapi.com/
    """
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        response = comet_client.responses.create(
            model=COMETAPI_GPT_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]
        )
        return response.output_text  # 🟢 حذف متد replace برای جلوگیری از خرابی کدهای ریاضی
    except Exception as e:
        return f"خطا در دریافت پاسخ از چت‌جی‌پتی (CometAPI): {str(e)}"

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
          chtml: {{
            scale: 1,
            minScale: 0.55,
            matchFontHeight: false
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
            * {{ box-sizing: border-box; }}
            html, body {{ max-width: 100%; overflow-x: hidden; }}
            body {{
                font-family: 'Tahoma', 'Arial', sans-serif;
                line-height: 1.8;
                padding: 16px;
                background-color: #f1f5f9;
                color: #1e293b;
                direction: rtl;
                font-size: 15px;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h2 {{
                text-align: center;
                color: #1e3a8a;
                border-bottom: 3px solid #3b82f6;
                padding-bottom: 12px;
                margin-bottom: 24px;
                font-size: 1.2rem;
            }}
            .columns {{
                display: flex;
                flex-wrap: wrap;
                gap: 16px;
                align-items: stretch;
            }}
            .box {{
                flex: 1 1 320px;
                min-width: 0;
                max-width: 100%;
                border: 1px solid #e2e8f0;
                padding: 16px;
                border-radius: 12px;
                background: white;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}
            .gemini {{ border-right: 6px solid #1a73e8; }}
            .gpt {{ border-right: 6px solid #10a37f; }}
            h3 {{
                margin-top: 0;
                margin-bottom: 15px;
                font-size: 1.05rem;
                display: flex;
                align-items: center;
                flex-wrap: wrap;
            }}
            .gemini h3 {{ color: #1a73e8; }}
            .gpt h3 {{ color: #10a37f; }}

            .content-area {{
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                word-break: break-word;
                font-size: 0.95rem;
                max-width: 100%;
            }}
            .content-area ul, .content-area ol {{
                padding-right: 20px;
                padding-left: 0;
                margin-top: 10px;
                margin-bottom: 10px;
            }}
            .content-area li {{
                margin-bottom: 8px;
            }}
            .content-area img {{
                max-width: 100%;
                height: auto;
            }}
            .content-area table {{
                display: block;
                max-width: 100%;
                overflow-x: auto;
                border-collapse: collapse;
                white-space: normal;
            }}
            .content-area code, .content-area pre {{
                overflow-x: auto;
                max-width: 100%;
                white-space: pre-wrap;
                word-break: break-word;
            }}

            /* فرمول‌های ریاضی طولانی: به‌جای زدن بیرون از کادر، اسکرول افقی داخل خودشون بگیرن */
            mjx-container {{
                max-width: 100%;
                overflow-x: auto;
                overflow-y: hidden;
            }}
            mjx-container[display="true"] {{
                display: block !important;
                margin: 0.6em 0 !important;
                padding-bottom: 4px;
            }}

            @media (max-width: 900px) {{
                .columns {{
                    flex-direction: column;
                }}
                .box {{
                    flex: 1 1 auto;
                }}
            }}

            @media (max-width: 480px) {{
                body {{ padding: 10px; font-size: 14px; }}
                h2 {{ font-size: 1.05rem; margin-bottom: 16px; }}
                .box {{ padding: 12px; border-radius: 10px; }}
                h3 {{ font-size: 0.95rem; margin-bottom: 10px; }}
                .content-area {{ font-size: 0.88rem; }}
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
                    <h3>🧠 پاسخ مدل ChatGPT (CometAPI)</h3>
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
    res = requests.post(f"{BALE_API_URL}sendMessage", json={"chat_id": chat_id, "text": text})
    print(f"🔹 sendMessage status: {res.status_code}, body: {res.text[:300]}")
    return res

def send_html_file(chat_id, filepath):
    with open(filepath, 'rb') as f:
        res = requests.post(
            f"{BALE_API_URL}sendDocument",
            data={"chat_id": chat_id, "caption": "🌐 پاسخ سوال شما آماده شد! فایل HTML بالا را باز کنید."},
            files={"document": f}
        )
    print(f"🔹 sendDocument status: {res.status_code}, body: {res.text[:300]}")
    return res

def process_updates(updates):
    print(f"📥 process_updates called with {len(updates)} update(s)")
    for update in updates:
        print(f"🔸 Raw update: {update}")

        if 'message' not in update:
            print("↪️ Skipped: no 'message' key in this update (probably an edited_message or other event)")
            continue

        message = update['message']
        chat_id = message.get('chat', {}).get('id')

        if not chat_id:
            print("↪️ Skipped: could not find chat_id in message")
            continue

        # اگر کاربر عکس فرستاد
        if 'photo' in message:
            try:
                print(f"📸 Photo received from user: {chat_id}")
                send_message(chat_id, "⏳ تصویر دریافت شد. در حال پردازش... لطفاً صبور باشید.")
                print("✅ Step 1/6 - 'processing' message sent to user")

                # ۱. گرفتن بالاترین کیفیت عکس
                file_id = message['photo'][-1]['file_id']
                print(f"🔹 Step 2/6 - file_id extracted: {file_id}")

                # ۲. درخواست به بله برای گرفتن مسیر فایل
                file_info_res = requests.get(f"{BALE_API_URL}getFile", params={"file_id": file_id})
                print(f"🔹 getFile response status: {file_info_res.status_code}, body: {file_info_res.text[:300]}")
                file_info = file_info_res.json()

                if not file_info.get('ok'):
                    print(f"❌ Failed to get file info from Bale: {file_info}")
                    continue

                file_path = file_info['result']['file_path']
                print(f"🔹 Step 3/6 - file_path from Bale: {file_path}")

                # ۳. دانلود بایت‌های واقعی عکس از سرور بله
                download_url = f"https://tapi.bale.ai/file/bot{BALE_TOKEN}/{file_path}"
                print(f"🔹 Downloading image from: {download_url}")
                img_res = requests.get(download_url)

                # مطمئن می‌شویم دانلود با موفقیت انجام شده است
                if img_res.status_code != 200:
                    print(f"❌ Failed to download photo from Bale server. Status code: {img_res.status_code}, body: {img_res.text[:300]}")
                    continue

                img_data = img_res.content
                print(f"🔹 Step 4/6 - Photo downloaded successfully. File size: {len(img_data)} bytes")

                # اگر حجم فایل کمتر از ۱ کیلوبایت باشد یعنی عکس نیست و متن خطاست
                if len(img_data) < 1000:
                    print(f"⚠️ Warning: File size is very small! Downloaded content may be an error: {img_data[:100]}")

                # ۴. ارسال هم‌زمان به هر دو هوش مصنوعی (موازی، برای سرعت بیشتر)
                print("→ Sending to Gemini AND ChatGPT in parallel (via CometAPI)...")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    gemini_future = executor.submit(ask_gemini, img_data)
                    gpt_future = executor.submit(ask_chatgpt, img_data)

                    gemini_ans = gemini_future.result()
                    gpt_ans = gpt_future.result()

                print(f"✅ Gemini responded ({len(gemini_ans) if gemini_ans else 0} chars). Preview: {str(gemini_ans)[:200]}")
                print(f"✅ ChatGPT responded ({len(gpt_ans) if gpt_ans else 0} chars). Preview: {str(gpt_ans)[:200]}")

                # ۵. ساخت فایل HTML
                print("🔹 Step 5/6 - Building HTML report...")
                html_name = f"answer_{chat_id}.html"
                create_html_report(gemini_ans, gpt_ans, html_name)
                print(f"✅ HTML report created: {html_name}")

                # ۶. ارسال فایل نهایی به بله
                print("🔹 Step 6/6 - Sending HTML file to user via Bale...")
                send_html_file(chat_id, html_name)
                print("✅ HTML file sent to user.")

                if os.path.exists(html_name):
                    os.remove(html_name)
                    print(f"🧹 Cleaned up local file: {html_name}")

            except Exception:
                print("🔥 EXCEPTION while processing photo:")
                traceback.print_exc()
                try:
                    send_message(chat_id, "❌ خطایی در پردازش تصویر رخ داد. لطفاً دوباره امتحان کنید.")
                except Exception:
                    print("🔥 Also failed to send error message to user:")
                    traceback.print_exc()

        # اگر کاربر متن فرستاد
        elif 'text' in message:
            print(f"💬 Text message received: {message['text']!r}")
            send_message(chat_id, "سلام! 👋 لطفاً از سوال امتحانی خود یک عکس واضح بفرستید.")
            print("✅ Greeting sent")

        else:
            print(f"↪️ Message type not handled (no photo, no text): {list(message.keys())}")

# ==========================================
# 🚀 حلقه اصلی اجرا (Polling)
# ==========================================
def main():
    print("🤖 Bale bot is running (Gemini + ChatGPT both via CometAPI)...")
    last_update_id = 0
    while True:
        try:
            print(f"⏳ Polling getUpdates with offset={last_update_id + 1} ...")
            res = requests.get(f"{BALE_API_URL}getUpdates", params={"offset": last_update_id + 1, "timeout": 10})
            print(f"🔹 getUpdates response status: {res.status_code}")

            if res.status_code != 200:
                print(f"❌ Unexpected status code from getUpdates: {res.status_code}, body: {res.text[:300]}")
                time.sleep(3)
                continue

            data = res.json()

            if not data.get('ok'):
                print(f"❌ Bale API returned ok=False: {data}")
                time.sleep(3)
                continue

            if data['result']:
                print(f"📬 Received {len(data['result'])} new update(s)")
                process_updates(data['result'])
                last_update_id = data['result'][-1]['update_id']
                print(f"🔹 last_update_id advanced to {last_update_id}")
            else:
                # چیزی نیامده - این طبیعیه، فقط برای دیباگ چاپ می‌کنیم
                print("… no new updates this cycle")

        except Exception as e:
            print(f"🔥 EXCEPTION in main polling loop: {e}")
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()