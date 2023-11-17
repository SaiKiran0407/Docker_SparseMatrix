# Use the official Python image from Docker Hub as the base image
FROM python:3.8-slim
# Set working directory
WORKDIR /usr/src/sparse_matrix
# Copy
COPY sparse_recommender.py .
COPY test_sparse_recommender.py .

RUN pip install --progress-bar off numpy pytest

# run
CMD ["python", "sparse_recommender.py"]
CMD [ "pytest", "./test_sparse_recommender.py" ]