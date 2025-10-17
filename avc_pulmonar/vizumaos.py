import cv2
import mediapipe as mp
import time
import math

def visualize_hand_tracking_pro():
    """
    Inicializa a webcam, rastreia as mãos usando MediaPipe,
    implementa controle de zoom por pinça, e exibe em tela cheia com aparência profissional.
    """
    # --- Configuração do MediaPipe Hands ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Focamos em uma mão para o gesto de pinça
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    mp_drawing = mp.solutions.drawing_utils

    # --- Configuração da Câmera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera. Verifique se ela está conectada e disponível.")
        print("Certifique-se de que nenhum outro aplicativo esteja usando a câmera.")
        return

    # Obter as dimensões originais da câmera (pode ser útil para mensagens iniciais, mas não essencial para o loop)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Removido, pois h,w serão obtidos por frame
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Removido, pois h,w serão obtidos por frame

    # --- Configuração da Janela OpenCV ---
    window_name = "Rastreamento de Mãos Profissional (Pressione 'q' para sair)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Permite redimensionar a janela
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Define tela cheia

    print(f"Câmera iniciada. Faça o gesto de pinça para zoom. Pressione 'q' para sair.")
    print("Para sair do modo tela cheia, pressione Esc. Para fechar o programa, pressione 'q'.")

    # --- Variáveis de Controle de Zoom ---
    p_time = 0  # Tempo anterior para cálculo de FPS
    initial_pinch_distance = None  # Distância inicial da pinça para referência
    current_zoom_level = 1.0  # Nível de zoom atual (1.0 = sem zoom)
    zoom_sensitivity = 0.008  # Sensibilidade do zoom (ajuste conforme necessário)
    min_zoom = 0.5  # Zoom mínimo
    max_zoom = 3.0  # Zoom máximo

    while True:
        success, img = cap.read()
        if not success:
            print("Erro: Não foi possível ler o frame da câmera.")
            break

        # Espelha a imagem para que pareça um espelho
        img = cv2.flip(img, 1)
        # Converte para RGB (formato esperado pelo MediaPipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- ONDE A CORREÇÃO É APLICADA: Obtenha as dimensões do frame AQUI ---
        h, w, c = img.shape  # Agora h e w são sempre definidos para cada frame

        # Processa a imagem para detectar as mãos
        results = hands.process(img_rgb)

        # --- Lógica de Rastreamento e Zoom ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os landmarks no frame original
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                                          # Conexões verdes
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4,
                                                                 circle_radius=5))  # Pontos vermelhos

                # Obter as coordenadas das pontas do polegar e indicador
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Converter coordenadas normalizadas (0-1) para pixels
                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                # Desenhar um círculo maior e linha na pinça para destaque
                cv2.circle(img, (x1, y1), 12, (255, 100, 0), cv2.FILLED)  # Laranja
                cv2.circle(img, (x2, y2), 12, (255, 100, 0), cv2.FILLED)  # Laranja
                cv2.line(img, (x1, y1), (x2, y2), (255, 100, 0), 5)  # Laranja mais grossa

                # Calcular a distância euclidiana entre os dois pontos
                current_pinch_distance = math.hypot(x2 - x1, y2 - y1)

                # Se a distância da pinça for menor que um limiar, consideramos que o gesto de pinça está ativo
                pinch_active_threshold = 80  # Ajuste este valor conforme a necessidade (em pixels)

                if current_pinch_distance < pinch_active_threshold:
                    if initial_pinch_distance is None:
                        initial_pinch_distance = current_pinch_distance
                        # Exibir "Pinça Ativa"
                        cv2.putText(img, "PINCA ATIVA!", (w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                                    cv2.LINE_AA)
                    else:
                        # Calcular a mudança na distância da pinça
                        delta_distance = current_pinch_distance - initial_pinch_distance

                        # Ajustar o nível de zoom
                        current_zoom_level += delta_distance * zoom_sensitivity

                        # Limitar o nível de zoom dentro dos valores definidos
                        current_zoom_level = max(min_zoom, min(current_zoom_level, max_zoom))

                        # Atualizar a distância inicial para o próximo frame para um controle contínuo
                        initial_pinch_distance = current_pinch_distance

                else:
                    # Se a pinça não estiver ativa, resetar a distância inicial
                    initial_pinch_distance = None

        else:
            # Se nenhuma mão for detectada, resetar a distância inicial
            initial_pinch_distance = None

        # --- Aplica o Zoom Digital ---
        # Calcula as dimensões da região a ser cortada com base no zoom
        zoomed_width = int(w / current_zoom_level)
        zoomed_height = int(h / current_zoom_level)

        # Calcula as coordenadas de início para centralizar o zoom
        x_start = max(0, int((w - zoomed_width) / 2))
        y_start = max(0, int((h - zoomed_height) / 2))
        x_end = min(w, x_start + zoomed_width)
        y_end = min(h, y_start + zoomed_height)

        # Corta a imagem (que já contém os desenhos dos landmarks)
        cropped_img = img[y_start:y_end, x_start:x_end]

        # Redimensiona a imagem cortada de volta para o tamanho original do frame
        # Isso cria o efeito de "zoom in" preenchendo a tela
        if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
            final_img = cv2.resize(cropped_img, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Em caso de corte inválido (ex: zoom extremo que resulta em dimensão 0),
            # exibe a imagem original para evitar erros.
            final_img = img

        # --- Adiciona Elementos Visuais Profissionais ---
        # Calcula e exibe o FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        fps_text = f'FPS: {int(fps)}'
        zoom_text = f'Zoom: {current_zoom_level:.2f}x'

        # Adiciona um fundo para o texto para melhor legibilidade
        def put_text_with_background(img_to_draw_on, text, org, font, font_scale, text_color, bg_color, thickness):
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            # Adiciona um pequeno padding ao redor do texto
            padding = 10
            x_bg = org[0]
            y_bg = org[1] - text_height - baseline - padding
            w_bg = text_width + 2 * padding
            h_bg = text_height + baseline + 2 * padding

            # Desenha o retângulo de fundo
            # Certifica-se de que as coordenadas do retângulo estão dentro dos limites da imagem
            x_bg_end = min(img_to_draw_on.shape[1], x_bg + w_bg)
            y_bg_end = min(img_to_draw_on.shape[0], y_bg + h_bg)

            # Cria uma sub-imagem para o fundo e mistura com a imagem original
            overlay = img_to_draw_on.copy()
            cv2.rectangle(overlay, (x_bg, y_bg), (x_bg_end, y_bg_end), bg_color, cv2.FILLED)
            alpha = 0.6  # Transparência do fundo
            img_to_draw_on = cv2.addWeighted(overlay, alpha, img_to_draw_on, 1 - alpha, 0)

            # Desenha o texto
            cv2.putText(img_to_draw_on, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)
            return img_to_draw_on  # Retorna a imagem com o texto e fundo aplicados

        # Aplica o texto e o fundo na imagem final
        final_img = put_text_with_background(final_img, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                             (255, 255, 255), (0, 0, 0), 2)
        final_img = put_text_with_background(final_img, zoom_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                             (255, 255, 255), (0, 0, 0), 2)

        # Exibe o frame final
        cv2.imshow(window_name, final_img)

        # --- Controle de Saída ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Pressione 'q' para sair do programa
            break
        elif key == 27:  # Pressione 'Esc' para sair do modo tela cheia (opcional)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # Libera a câmera e fecha todas as janelas do OpenCV
    cap.release()
    cv2.destroyAllWindows()
    print("Programa encerrado.")


if __name__ == "__main__":
    visualize_hand_tracking_pro()
