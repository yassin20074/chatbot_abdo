# 1. استخدام صورة بيثون خفيفة
FROM python:3.11-slim

# 2. تعيين مجلد العمل الأساسي
WORKDIR /app

# 3. إعداد متغيرات بيئة البيثون لتقليل الحجم وإظهار الـ Logs مباشرة
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 4. نسخ ملف المتطلبات أولاً لتسريع بناء الطبقات (Layer Caching)
COPY requirements.txt /app/requirements.txt

# 5. تحديث pip وتثبيت الاعتماديات
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 6. نسخ كل ملفات المشروع والمجلدات إلى داخل مجلد العمل /app
COPY . /app/

# 7. توثيق المنفذ
EXPOSE 8000

# 7. أمر التشغيل المستقر لـ Railway
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
