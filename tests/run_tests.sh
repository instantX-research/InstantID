#!/bin/bash

echo "Starting the script to run tests..."

# Build the model
echo "Building the model with cog..."
sudo cog build -t test-model --use-cog-base-image
echo "Model build completed."

# Stop and remove the existing container if it's running
container_name='cog-test'
echo "Checking if the container '$container_name' is already running..."
if sudo docker inspect --format="{{.State.Running}}" "$container_name" &> /dev/null; then
    echo "Container '$container_name' is running. Stopping and removing..."
    sudo docker stop "$container_name"
    sudo docker rm "$container_name"
    echo "Container '$container_name' stopped and removed successfully."
else
    echo "Container '$container_name' not found or not running. Proceeding to run a new instance."
fi

# Run the container
echo "Running the container '$container_name'..."
sudo docker run -d -p 5000:5000 --gpus all --name "$container_name" test-model
echo "Container '$container_name' is now running."

# Wait for the server to be ready
echo "Waiting for the server to be ready..."
sleep 10
echo "Server should be ready now."

# Set the environment variable for local testing
echo "Setting environment variable for local testing..."
export TEST_ENV=local
echo "Environment variable set."

# Run the specific test case
echo "Running the test case: test_seeded_prediction..."
pytest -vv tests/test_predict.py::test_seeded_prediction
echo "Test case execution completed."

# Stop the container
echo "Stopping the container '$container_name'..."
sudo docker stop "$container_name"
echo "Container '$container_name' stopped. Script execution completed."