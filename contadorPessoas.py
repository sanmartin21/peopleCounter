import cv2
import numpy as np

# Definindo constantes
X, Y, W, H = 490, 230, 30, 150
THRESHOLD = 4000
KERNEL_SIZE = (8, 8)
ITERATIONS = 2

def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Erro: Não foi possível abrir o arquivo de vídeo.")
        return None
    return video

def process_frame(frame):
    frame = cv2.resize(frame, (1100, 720))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    frame_threshold = cv2.adaptiveThreshold(frame_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 12)
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    frame_dilated = cv2.dilate(frame_threshold, kernel, iterations=ITERATIONS)

    return frame, frame_dilated

def count_white_pixels(frame):
    crop = frame[Y:Y+H, X:X+W]
    return cv2.countNonZero(crop)

def draw_rectangle(frame, liberado):
    color = (0, 255, 0) if not liberado else (255, 0, 255)
    cv2.rectangle(frame, (X, Y), (X + W, Y + H), color, 4)
    return frame

def main():
    video = load_video('video.mp4')
    if video is None:
        return

    contador = 0
    liberado = False

    while True:
        ret, frame = video.read()
        if not ret:
            print("Erro: Não foi possível ler o frame.")
            break
        
        frame, frame_dilated = process_frame(frame)
        white_pixels = count_white_pixels(frame_dilated)

        if white_pixels > THRESHOLD and liberado == True:
            contador +=1
        if white_pixels < THRESHOLD:
            liberado = True
        else:
            liberado = False

        frame = draw_rectangle(frame, liberado)
        cv2.putText(frame, str(white_pixels), (X-30, Y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (575,155), (575 + 88, 155 + 85), (255, 255, 255), -1)
        cv2.putText(frame, str(contador), (X+100, Y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

        cv2.imshow('video original', frame)
        cv2.imshow('Vídeo Processado', cv2.resize(frame_dilated, (600, 500)))

        key = cv2.waitKey(20)
        if key == 27:  # Interromper o loop se a tecla 'Esc' for pressionada
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
