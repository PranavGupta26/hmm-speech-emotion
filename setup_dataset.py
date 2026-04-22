import os
import shutil

source = "Audio_Speech_Actors_01-24"
target = "dataset"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

for root, dirs, files in os.walk(source):
    print("Checking folder:", root)   # DEBUG

    for file in files:
        if file.endswith(".wav"):
            print("Found file:", file)   # DEBUG

            parts = file.split("-")
            if len(parts) < 3:
                continue

            emotion_code = parts[2]
            print("Emotion code:", emotion_code)  # DEBUG

            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]

                dest_folder = os.path.join(target, emotion)
                os.makedirs(dest_folder, exist_ok=True)

                src_path = os.path.join(root, file)
                dst_path = os.path.join(dest_folder, file)

                shutil.copy(src_path, dst_path)
                print("Copied to:", dst_path)  # DEBUG

print("\nDataset created successfully!")