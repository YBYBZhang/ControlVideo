import cv2
def video_to_images(input_vid):
    video = cv2.VideoCapture(input_vid)

    images = []
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # if count % 10 == 0:
        images.append(frame)
        count += 1
    video.release()

    return images

if __name__ == "__main__":
    print(len(video_to_images("/Users/evan_kim/urop/controlvideo/outputs/Walking over stairs, first-person view, no background, no tree, blue sky and sun. smalller.mp4")))