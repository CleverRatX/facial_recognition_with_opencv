import cv2

def face_capture():
    cascade_path = 'filters\haarcascade_frontalface_default.xml'
    
    # классификатор каскада - загружаем обученный каскад из xml
    clf = cv2.CascadeClassifier(cascade_path)
    if clf.empty():
        raise FileNotFoundError(f'Не удалось загрузить каскад: {cascade_path}')
    
    # берем видео из файла
    # video = cv2.VideoCapture('Путь к файлу')
    # берем видео с камеры
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        raise RuntimeError('Не удалось открыть видеоисточник')
    
    win_name = "Faces"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)  # создаём окно заранее
    
    #цикл работает до тех пор пока мы не нажмем на клавишу (в случае оконячания видео так же выходим из цикла)
    while cv2.waitKey(1) < 0 and not (cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1):
        # читаем один кадр
        ret, frame = video.read()
        if not ret or frame is None:
            break

        # Для Haar-детектора работаем в градациях серого (так быстрее и так он обучен)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Запускаем поиск лиц, параметры можно настроить.
        faces = clf.detectMultiScale(
            gray,
            scaleFactor = 1.1,   # шаг изменения масштаба; меньше — точнее, но медленнее (обычно 1.05–1.2)
            minNeighbors = 5,    # «жесткость» фильтрации; больше — меньше ложных, но больше пропусков
            minSize = (30, 30),  # минимальный размер детектируемого лица в пикселях
        )
        
        # Рисуем рамки вокруг всех найденных лиц
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 4)

        # Показываем текущий кадр с разметкой.
        cv2.imshow('Faces', frame)
        
    # Освобождаем ресурсы и закрываем окна.   
    video.release()
    cv2.destroyAllWindows()


def main():
    face_capture()
    
    
if __name__ == '__main__':
    main()
