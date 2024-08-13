# Start from the AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.12

# Copy function code and requirements file
COPY data_generation.py requirements.txt ./

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Set the CMD to your handler (app is the Python file, handler is the function)
CMD ["data_generation.lambda_handler"]
