# Step 1: Use an official Python image as the base image
FROM python:3.12-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements.txt file into the container (if available)
# Make sure to create a `requirements.txt` file in your project directory with the necessary dependencies
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the FastAPI app code into the container
COPY . /app

# Step 6: Expose the port FastAPI will run on (in this case, port 8000)
EXPOSE 8000

# Step 7: Command to run the FastAPI app using Uvicorn when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
