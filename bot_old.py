import time
import requests
import base64
import os
from openai import OpenAI
import google.genai as genai
from google.genai import types


# ==========================================
# ⚙️ تنظیمات (CONFIG) - توکن‌های خود را اینجا بگذارید
# ==========================================
BALE_TOKEN = "1055685913:8ffdDoHq3t4a4iAjmUlqIS51cdQmnZ4vknc"
#telegram_TOKEN = "8988146917:AAEFeycCw0ymWTb8rFrJVPNpDYCfT2a-r9M"
GEMINI_API_KEY = "AQ.Ab8RN6JQzZ0BdJcfyBb7jYaKMfvjl7lzFUS0Wn5k8gQf-fKHYA"
OPENAI_API_KEY = "sk-phUTkqfAix0YEtSXzl1Y13YC6Kqps7kt61VwNeTZOzx4lTzt"
DEEPSEEK_API_KEY = "توکن_دیپ_سیک" # اگر هنوز ندارید مهم نیست

BALE_API_URL = f"https://tapi.bale.ai/bot{BALE_TOKEN}/"
#BALE_API_URL = f"https://api.telegram.org/bot{telegram_TOKEN}/"
OPENAI_BASE_URL = "https://api.prox.us.ci/v1"
SYSTEM_PROMPT = (
    "تو یک استاد دانشگاه و متخصص علوم و مهندسی هستی. تصویر ارسالی یک سوال امتحانی است. "
    "سوال را به دقت تحلیل کن و پاسخ را کاملاً گام‌به‌گام، تشریحی و با فرمول‌های دقیق بنویس. "
    "از فرمت استانداردی برای فرمول‌ها استفاده کن (مثلا فرمت متنی تمیز یا علائم ریاضی واضح) تا در صفحه وب به خوبی رندر شود."
)

# ==========================================
# 🧠 توابع هوش مصنوعی
# ==========================================
def ask_gemini(image_bytes):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # 🟢 اصلاح این بخش: اجازه می‌دهیم گوگل خودش نوع تصویر را تشخیص دهد
        image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg"  # برگرداندن به حالت استاندارد
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, SYSTEM_PROMPT]
        )
        return response.text.replace('\n', '<br>')
    except Exception as e:
        return f"خطا در دریافت پاسخ از جمنای: {str(e)}"

def ask_chatgpt(image_bytes):
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # ست کردن کلاینت با آدرس سرور پروکسی شما
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        
        response = client.chat.completions.create(
            model="gpt-5.5-openai-compact", # ← نام دقیق مدل بر اساس پنل شما
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
        return response.choices[0].message.content.replace('\n', '<br>')
    except Exception as e:
        return f"خطا در دریافت پاسخ از چت‌جی‌پتی: {str(e)}"

# ==========================================
# 🌐 تابع ساخت فایل HTML گرافیکی
# ==========================================
def create_html_report(gemini_ans, gpt_ans, output_filename="answer.html"):
    html_content = f"""
    <!DOCTYPE html>
    <html dir="rtl" lang="fa">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>پاسخ‌نامه هوش مصنوعی</title>
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body {{ font-family: 'Tahoma', 'Arial', sans-serif; line-height: 1.8; padding: 20px; background-color: #f1f5f9; color: #1e293b; direction: rtl; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            h2 {{ text-align: center; color: #1e3a8a; border-bottom: 3px solid #3b82f6; padding-bottom: 12px; margin-bottom: 30px; }}
            .box {{ border: 1px solid #e2e8f0; padding: 20px; margin-bottom: 25px; border-radius: 12px; background: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }}
            .gemini {{ border-right: 6px solid #1a73e8; }}
            .gpt {{ border-right: 6px solid #10a37f; }}
            h3 {{ margin-top: 0; margin-bottom: 15px; font-size: 15pt; display: flex; align-items: center; }}
            .gemini h3 {{ color: #1a73e8; }}
            .gpt h3 {{ color: #10a37f; }}
            p {{ margin: 0 0 10px 0; font-size: 11pt; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📄 پاسخ‌نامه تشریحی هوش مصنوعی</h2>
            <div class="box gemini">
                <h3>🤖 پاسخ مدل Gemini (Google)</h3>
                <div>{gemini_ans}</div>
            </div>
            <div class="box gpt">
                <h3>🧠 پاسخ مدل ChatGPT (OpenAI)</h3>
                <div>{gpt_ans}</div>
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
            print(f"📸 دریافت عکس از کاربر: {chat_id}")
            
            # ۱. گرفتن بالاترین کیفیت عکس
            file_id = message['photo'][-1]['file_id']
            
            # ۲. درخواست به بله برای گرفتن مسیر فایل
            file_info_res = requests.get(f"{BALE_API_URL}getFile", params={"file_id": file_id})
            file_info = file_info_res.json()
            
            if not file_info.get('ok'):
                print(f"❌ خطا در دریافت اطلاعات فایل از بله: {file_info}")
                continue
                
            file_path = file_info['result']['file_path']
            
            # ۳. دانلود بایت‌های واقعی عکس از سرور بله
            download_url = f"https://tapi.bale.ai/file/bot{BALE_TOKEN}/{file_path}"
            img_res = requests.get(download_url)
            
            # مطمئن می‌شویم دانلود با موفقیت انجام شده است
            if img_res.status_code != 200:
                print(f"❌ خطا در دانلود عکس از سرور بله. کد وضعیت: {img_res.status_code}")
                continue
                
            img_data = img_res.content
            print(f"🔹 عکس با موفقیت دانلود شد. حجم فایل: {len(img_data)} بایت")

            # اگر حجم فایل کمتر از ۱ کیلوبایت باشد یعنی عکس نیست و متن خطاست
            if len(img_data) < 1000:
                print(f"⚠️ هشدار: حجم فایل بسیار کم است! محتوای دانلود شده احتمالا خطا است: {img_data[:100]}")
            
            # ۴. ارسال به هوش‌های مصنوعی
            print("→ ارسال به جمنای...")
            gemini_ans = ask_gemini(img_data)
            
            print("→ ارسال به چت‌جی‌پتی...")
            gpt_ans = ask_chatgpt(img_data)
            
            # ۵. ساخت فایل HTML
            html_name = f"answer_{chat_id}.html"
            create_html_report(gemini_ans, gpt_ans, html_name)
            
            # ۶. ارسال فایل نهایی به بله
            send_html_file(chat_id, html_name)
            print("✅ فایل HTML برای کاربر ارسال شد.")
            
            if os.path.exists(html_name): 
                os.remove(html_name)

        # اگر کاربر متن فرستاد
        elif 'text' in message:
            send_message(chat_id, "سلام! 👋 لطفاً از سوال امتحانی خود یک عکس واضح بفرستید.")
# ==========================================
# 🚀 حلقه اصلی اجرا (Polling)
# ==========================================
def main():
    print("🤖 ربات بله فعال شد و بدون نیاز به موتور PDF کار می‌کند... (Bot is running)")
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
            print(f"⚠️ خطای شبکه: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()