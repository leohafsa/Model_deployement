FROM public.ecr.aws/lambda/python:3.8
# Copy function code
COPY app.py botos3.py face_lib_complete.py requirements.txt ./
COPY face_detection ./face_detection
# Install the function's dependencies using file requirements.txt

RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"  -U --no-cache-dir
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]