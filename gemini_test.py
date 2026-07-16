import google.genai as genai
from google.genai import types

GEMINI_API_KEY = "AQ.Ab8RN6JQzZ0BdJcfyBb7jYaKMfvjl7lzFUS0Wn5k8gQf-fKHYA"

def test_standalone():
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # یک عکس تست کوچک در کنار فایل کد قرار دهید
        with open("test_image.jpg", "rb") as f:
            image_bytes = f.read()
            
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        
        print("⏳ در حال ارسال درخواست به جمنای...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, "سلام، این یک تست است. چه چیزی در تصویر می‌بینی؟"]
        )
        print("✅ پاسخ موفقیت‌آمیز بود:")
        print(response.text)
        
    except Exception as e:
        print("❌ خطا رخ داد:")
        print(str(e))

if __name__ == "__main__":
    test_standalone()