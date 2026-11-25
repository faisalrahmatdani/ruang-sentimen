FROM python:3.10-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user app.py .
COPY --chown=user ruang_sentimen ruang_sentimen
COPY --chown=user static static
COPY --chown=user assets assets
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
