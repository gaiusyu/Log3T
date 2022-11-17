FROM python:3.7.3
 
WORKDIR /
 
ADD . .
 
RUN pip install -r requirements.txt

CMD ["python","/Log3T/classfy.py"]
 
