def amb_start():
    from keras.preprocessing.image import img_to_array
    from keras.models import load_model
    import time
    import numpy as np
    import imutils
    import cv2



    model = load_model("ambulance_final_32.model")
    # cap = cv2.VideoCapture(0)

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Capture frame-by-frame
            ret, frame = cap.read()
            cv2.imshow("Output2", frame)

            image = frame
            orig = image.copy()
            image = cv2.resize(image, (32, 32))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # classify the input image
            (notAmbulance, ambulance) = model.predict(image)[0]
            # build the label
            label = "ambulance" if (ambulance > notAmbulance and ambulance > 0.7) else "Not Ambulance"
            proba = ambulance if (ambulance > notAmbulance and ambulance > 0.7) else notAmbulance

            label = "{}: {:.2f}%".format(label, proba * 100)

            # draw the label on the image
            output = imutils.resize(orig, width=400)
            cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            #rawCapture.truncate(0)
            # show the output image
            #cv2.imshow("Output", output)
            #cv2.imshow("Output2", frame)

            if proba > .90:
                print("Ambulance")
            else:
                pass
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()