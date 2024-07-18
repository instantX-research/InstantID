import os
import gdown
import zipfile

def download_and_extract(url, output_dir, zip_file_name):
    """
    Downloads a zip file from the specified URL and extracts it to the given directory.
    
    Parameters:
        url (str): The URL from which to download the zip file.
        output_dir (str): The directory where the zip file will be saved and extracted.
        zip_file_name (str): The name of the zip file to be downloaded.
    
    This function performs several tasks:
    - Ensures the output directory exists.
    - Downloads the zip file using gdown if it does not already exist.
    - Extracts the zip file into the specified directory.
    """
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Full path for the downloaded zip file
    zip_path = os.path.join(output_dir, zip_file_name)

    # Check if the zip file already exists
    if not os.path.exists(zip_path):
        # Download the file
        try:
            print(f"Downloading file to: {zip_path}")
            gdown.download(url=url, output=zip_path, quiet=False, fuzzy=True)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download the file: {e}")
            return  # Exit if download fails
    else:
        print(f"File already exists: {zip_path}")

    # Extract the zip file
    try:
        print(f"Extracting file to: {output_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("File extracted successfully.")
    except Exception as e:
        print(f"Failed to extract the file: {e}")

# Example usage
download_and_extract(
    url="https://drive.google.com/uc?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8",
    output_dir="./models",
    zip_file_name="antelopev2.zip"
)
