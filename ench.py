import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os 
import sys
def adjust_face_and_background(input_image_path, output_image_path, brightness_factor=1.0, saturation_factor=1.0, expansion_factor=0.3):
    """
    Parlaklık ve canlılık değerlerini yüz bölgesine uygular ve arka planı beyaz yapar.
    :param input_image_path: Girdi görüntüsünün dosya yolu.
    :param output_image_path: Çıktı görüntüsünün dosya yolu.
    :param brightness_factor: Parlaklık faktörü (0.0 - 1.0 arasında azaltır, >1.0 artırır).
    :param saturation_factor: Canlılık faktörü (0.0 - 1.0 arasında azaltır, >1.0 artırır).
    :param expansion_factor: Yüz bölgesini genişletme oranı (0.1: %10, 0.3: %30 genişletir).
    """
    # Yüz algılama modeli (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Görüntüyü yükleme
    image = cv2.imread(input_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Beyaz arka plan oluşturma
    white_background = np.ones_like(image, dtype=np.uint8) * 255

    # Boş maske oluşturma
    mask = np.zeros_like(gray, dtype=np.uint8)

    # Yüz bölgesi işlemleri
    for (x, y, w, h) in faces:
        # Yüz bölgesini genişletme
        x_expanded = max(0, int(x - w * expansion_factor))  # Sol kenar
        y_expanded = max(0, int(y - h * expansion_factor))  # Üst kenar
        w_expanded = min(image.shape[1] - x_expanded, int(w + 2 * w * expansion_factor))  # Genişlik
        h_expanded = min(image.shape[0] - y_expanded, int(h + 2 * h * expansion_factor))  # Yükseklik

        # Genişletilmiş yüz bölgesine maske uygula
        mask[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded] = 255

        # Yüz bölgesini al
        face = image[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]

        # Yüz bölgesini parlaklık ve canlılık için düzenle
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Parlaklık ayarı
        brightness_enhancer = ImageEnhance.Brightness(face_pil)
        face_pil = brightness_enhancer.enhance(brightness_factor)

        # Canlılık ayarı
        saturation_enhancer = ImageEnhance.Color(face_pil)
        face_pil = saturation_enhancer.enhance(saturation_factor)

        # Görüntüyü geri OpenCV formatına çevir
        face = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)

        # Düzenlenen yüzü orijinal görüntüye yerleştir
        image[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded] = face

    # Maske ile ön plan ve beyaz arka planı birleştirme
    foreground = cv2.bitwise_and(image, image, mask=mask)
    inverted_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)
    final_image = cv2.add(foreground, background)

    # Görüntüyü kaydetme
    cv2.imwrite(output_image_path, final_image)
    print(f"Görüntü başarıyla kaydedildi: {output_image_path}")

# Kullanıcı parametreleri ile deneme
input_image = "girdi.jpeg"
output_image = "sonuc.jpg"
brightness = 0.8  # Parlaklık faktörü
saturation = 1.5  # Canlılık faktörü
expansion = 0.4   # Yüz bölgesini genişletme oranı
# Çalıştır
if not os.path.exists(input_image):
    print(f"Girdi dosyası bulunamadı: {input_image}")
    sys.exit(1)

adjust_face_and_background(input_image, output_image, brightness_factor=brightness, saturation_factor=saturation, expansion_factor=expansion)
print(f"Görüntü işleme tamamlandı! Çıktı: {output_image}")