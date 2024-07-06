FROM python:3.10

WORKDIR /app

COPY . /app/

# Upgrade pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.11 get-pip.py
RUN python3.11 -m pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0" , "--port" , "80" ]

