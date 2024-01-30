# for checkpoints d, cd in, tar it up, then move up a d, then push to gcp
# e.g.
# cd checkpoints/models--stablediffusionapi--nightvision-xl-0791
# sudo tar -cvf ../models--stablediffusionapi--nightvision-xl-0791.tar *
# cd ..
# gcloud storage cp models--stablediffusionapi--nightvision-xl-0791.tar gs://replicate-weights/InstantID/models--stablediffusionapi--nightvision-xl-0791.tar

# TODO

import os
import subprocess

# Get the list of directories in the checkpoints directory
dirs = [
    # "checkpoints/models--stablediffusionapi--juggernaut-xl-v8",
    # "checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
    # "checkpoints/models--stablediffusionapi--afrodite-xl-v2",
    # "checkpoints/models--stablediffusionapi--albedobase-xl-20",
    # "checkpoints/models--stablediffusionapi--albedobase-xl-v13",
    # "checkpoints/models--stablediffusionapi--animagine-xl-30",
    # "checkpoints/models--stablediffusionapi--anime-art-diffusion-xl",
    # "checkpoints/models--stablediffusionapi--anime-illust-diffusion-xl",
    # "checkpoints/models--stablediffusionapi--dreamshaper-xl",
    # "checkpoints/models--stablediffusionapi--duchaiten-real3d-nsfw-xl",
    # "checkpoints/models--stablediffusionapi--dynavision-xl-v0610",
    # "checkpoints/models--stablediffusionapi--guofeng4-xl",
    # "checkpoints/models--stablediffusionapi--hentai-mix-xl",
    # "checkpoints/models--stablediffusionapi--juggernaut-xl-v8",
    # "checkpoints/models--stablediffusionapi--nightvision-xl-0791",
    # "checkpoints/models--stablediffusionapi--omnigen-xl",
    # "checkpoints/models--stablediffusionapi--pony-diffusion-v6-xl",
    # "checkpoints/models--stablediffusionapi--protovision-xl-high-fidel",
    "checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
]

# Iterate over each directory
for d in dirs:
    # Construct the tar file name
    tar_file_name = f"{d}.tar"
    print(f"[!] Starting the process for directory: {d}")
    print(f"[!] Step 1: Constructing tar file name as '{tar_file_name}'")

    # Construct the full path to the tar file
    full_tar_path = os.path.join(
        "..", tar_file_name
    )  # Adjusted to account for script's new location
    print(f"[!] Step 2: The full path for the tar file is '{full_tar_path}'")

    # Remove 'checkpoints/' from tar_file_name for gcloud destination
    gcloud_tar_file_name = tar_file_name.replace("checkpoints/", "")
    # Construct the gcloud destination
    gcloud_destination = f"gs://replicate-weights/InstantID/{gcloud_tar_file_name}"
    print(
        f"[!] Step 3: The destination path on GCloud is set to '{gcloud_destination}'"
    )

    # Adjust the shell command string to account for the script's new location
    cmd = f"cd ../{d} && tar -cvf ../../{tar_file_name} * && gcloud storage cp ../../{tar_file_name} {gcloud_destination}"
    print(
        f"[!] Step 4: The shell command constructed to perform the operations is: {cmd}"
    )

    # Run the shell command
    print(f"[!] Step 5: Executing the shell command for directory: {d}")
    subprocess.run(cmd, shell=True)
    print(f"[!] Step 6: The shell command execution for directory '{d}' has completed.")
    print(f"[!] Process completed for directory: {d}")
