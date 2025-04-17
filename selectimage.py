import cv2

video_path = "data/Video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Don't open video.")
    exit()
# Chỉ định frame bạn muốn lấy
frame_number = 150  # Ví dụ, lấy frame thứ 100

# Thiết lập chỉ số frame trong video
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()

if ret:
    cv2.imshow('Frame', frame)
    cv2.imwrite('Frame.jpg', frame)
    cv2.waitKey(0)
else:
    print("Don't read frame.")

cap.release()
cv2.destroyAllWindows()