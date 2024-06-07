from CentroidTracker import CentroidTracker
import cv2
import os
import sys
import face_recognition


INPUT_FOLDER = "./frames"
OUTPUT_FOLDER = "./output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_image_path(idx):
    return f"{INPUT_FOLDER}/{str(idx).zfill(4)}.png"


# bounding box is in the format (top, right, bottom, left)
def draw_bbox(image, bb):
    cv2.rectangle(image, (bb[3], bb[0]), (bb[1], bb[2]), (0, 255, 0), 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <model>")
        sys.exit(1)

    if sys.argv[1] == "cnn":
        model = "cnn"

    # get the max image id from the frames folder
    image_names = os.listdir(INPUT_FOLDER)
    max_img_id = max([int(name.split(".")[0]) for name in image_names])
    start_img_id = 1

    tracker = CentroidTracker()

    # find the initial object we want to track (Jim Carrey's face)
    # based on the first image
    image = face_recognition.load_image_file(get_image_path(start_img_id))
    face_locations = face_recognition.face_locations(image)

    # this will register Jim Carrey's face
    face_id = tracker.register(face_locations[0])

    # iterate through the rest of the images
    for idx in range(start_img_id + 1, max_img_id + 1):
        image = face_recognition.load_image_file(get_image_path(idx))
        face_locations = face_recognition.face_locations(image, model=model)

        # update the tracker with the new locations we found
        # (we do not know which location corresponds to which face, so we must update the tracker with all the locations)
        center = tracker.update(face_locations)

        # the tracker has updated based on the new locations
        # and it should now give us the bounding box of the face we want to track
        bb = tracker.objects_bb[face_id]

        # change the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        draw_bbox(image, bb)

        cv2.imwrite(f"./output/{str(idx - start_img_id).zfill(4)}.png", image)
        print(f"Processed image {idx - start_img_id}/{max_img_id - start_img_id}")
