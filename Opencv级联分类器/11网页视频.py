import cv2


face_detect = cv2.CascadeClassifier('C:/anaconda3/envs/pytorch/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
class CaptureVideo(object):
	def net_video(self):
		# 获取网络视频流
		#cam = cv2.VideoCapture("rtmp://192.168.0.10/live/test")
		cam = cv2.VideoCapture('./test2.mp4')
		while cam.isOpened():
			sucess, frame = cam.read()
			if sucess:
				gary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				face = face_detect.detectMultiScale(gary, 1.01, 5, 0, (100, 100), (300, 300))
				for x, y, w, h in face:
					cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
				cv2.imshow("Network", frame)
				cv2.waitKey(1)
if __name__ == "__main__":
	capture_video = CaptureVideo()
	capture_video.net_video()