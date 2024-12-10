import cv2
import time

is_active = False
is_active_prev = False
time_sleep_start = None
time_sleep_end = None
time_active_start = None
time_active_end = None
activity_intervals = []
sleep_intervals = []
first_face = False
time_control_start = None


blinking_frequency = None
# read input image
cap = cv2.VideoCapture(0)


while True:
  ret, frame = cap.read()
# convert to grayscale of each frames
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cl1 = cv2.createCLAHE(clipLimit=2.0)  # set grid size
  clahe = cl1.apply(gray)  # clahe


# read the haarcascade to detect the faces in an image
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

  # read the haarcascade to detect the eyes in an image
  eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

  # detects faces in the input image
  faces = face_cascade.detectMultiScale(clahe, 1.2, 6)
  #print('Number of detected faces:', len(faces))

  if len(faces) != 0 and not first_face:
    time_control_start = time.time()
    first_face = True

  if time_control_start is not None and time.time() - time_control_start > 20:
    blinking_frequency = len(sleep_intervals) / (sum(sleep_intervals) + sum(activity_intervals))
    coefficient_active = sum(sleep_intervals) / sum(activity_intervals)
    coefficient_active_5_last = sum(sleep_intervals[-5:]) / sum(activity_intervals[-5:])
    frame = cv2.putText(frame, str(coefficient_active), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, str(coefficient_active_5_last), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if coefficient_active * 1.3 < coefficient_active_5_last:
      print("DON`T SLEEP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

  # loop over the detected faces
  for (x, y, w, h) in faces:
    roi_gray = clahe[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # detects eyes of within the detected face area (roi)
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 7)
    if len(eyes) != 0:
      is_active = True
      if not is_active_prev:
        time_sleep_end = time.time()
        time_active_start = time.time()
        time_active_end = None

    else:
      is_active = False

      if is_active_prev:
        time_sleep_start = time.time()
        time_sleep_end = None
        time_active_end = time.time()

    if time_sleep_start is not None and time_sleep_end is not None:
      time_sleep_delta = time_sleep_end - time_sleep_start
      sleep_intervals.append(time_sleep_delta)
      print("Sleep ", time_sleep_delta)
      time_sleep_end = None
      time_sleep_start = None

    if time_active_start is not None and time_active_end is not None:
      time_active_delta = time_active_end - time_active_start
      print("Active ", time_active_delta)
      activity_intervals.append(time_active_delta)
      time_active_start = None
      time_active_end = None

    # draw a rectangle around eyes
    for (ex, ey, ew, eh) in eyes:
      cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

    is_active_prev = is_active


# display the image with detected eyes
  cv2.imshow('Eyes Detection', frame)
  cv2.imshow('Gray', gray)
  cv2.imshow('Clahe', clahe)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.release()
cv2.destroyAllWindows()

