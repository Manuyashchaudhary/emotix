import os
import cv2
import dlib
import numpy as np
import argparse
from wide_resnet import WideResNet


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def main():
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file

    if not weight_file:
        weight_file = os.path.join("pretrained_models", "weights.18-4.06.hdf5")

    # for face detection
    #detector = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    cascade_path = "face_cascades/haarcascade_profileface.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)


    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    # capture video
    cap = cv2.VideoCapture("people.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        #get video frame
        ret, img = cap.read()
	print img.shape

        if not ret:
            print("error: failed to capture image")
            return -1

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        #detected = detector.compute_face_descriptor(input_img, 1)
	detected = face_cascade.detectMultiScale(input_img, 1.1, 2, 0, (20, 20) )
        faces = np.empty((len(detected), img_size, img_size, 3))

        for (x1,y1,w,h) in detected:
	    x2=x1+w
	    y2=y1+h
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        if len(detected) > 0:
            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(predicted_ages[i]),
                                    "F" if predicted_genders[i][0] > 0.5 else "M")
            #draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
	#cv2.imwrite('detected.png',img)
        key = cv2.waitKey(30)

        if key == 27:
            break


if __name__ == '__main__':
    main()
