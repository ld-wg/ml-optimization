import os
import cv2

def convert_wider_to_yolo(annotation_file, source_img_dir, output_label_dir):
    """
    Converte anotações do dataset WIDER FACE para o formato YOLOv8.

    Args:
        annotation_file (str): Caminho para o arquivo de anotações (ex: wider_face_train_bbx_gt.txt).
        source_img_dir (str): Caminho para o diretório que contém as imagens originais.
        output_label_dir (str): Caminho para o diretório onde os arquivos .txt do YOLO serão salvos.
    """
    # Cria o diretório de saída se ele não existir
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
        print(f"Diretório criado: {output_label_dir}")

    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    line_idx = 0
    num_lines = len(lines)
    processed_files = 0

    while line_idx < num_lines:
        # Pega o caminho do arquivo de imagem
        image_path_line = lines[line_idx].strip()
        if not image_path_line.endswith('.jpg'):
            line_idx += 1
            continue
        
        image_path = os.path.join(source_img_dir, image_path_line)
        
        # Lê a imagem para obter suas dimensões
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"AVISO: Não foi possível ler a imagem: {image_path}. Pulando.")
                # Pula a imagem e suas anotações
                num_boxes_line = lines[line_idx + 1].strip()
                num_boxes = int(num_boxes_line)
                line_idx += 2 + num_boxes
                continue

            img_h, img_w, _ = img.shape
        except Exception as e:
            print(f"ERRO ao processar {image_path}: {e}")
            line_idx += 1
            continue

        # Pega o número de bounding boxes
        line_idx += 1
        num_boxes_line = lines[line_idx].strip()
        num_boxes = int(num_boxes_line)
        line_idx += 1

        yolo_annotations = []
        
        # Itera sobre as bounding boxes da imagem
        for i in range(num_boxes):
            bbox_line = lines[line_idx + i].strip().split()
            # Formato WIDER: [x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose]
            x1, y1, w, h = [int(val) for val in bbox_line[:4]]
            invalid = int(bbox_line[7])

            # Ignora bounding boxes marcadas como inválidas ou com dimensões zero
            if invalid or w == 0 or h == 0:
                continue

            # Conversão para o formato YOLO
            # class_id é 0 para "face"
            class_id = 0
            
            # Calcula o centro da caixa
            x_center = x1 + w / 2
            y_center = y1 + h / 2

            # Normaliza as coordenadas pela largura e altura da imagem
            x_center_norm = x_center / img_w
            y_center_norm = y_center / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            yolo_annotations.append(f"{class_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}")

        # Salva o arquivo .txt de anotação se houver anotações válidas
        if yolo_annotations:
            # Cria o caminho para o arquivo de label
            label_filename = os.path.basename(image_path).replace('.jpg', '.txt')
            label_filepath = os.path.join(output_label_dir, label_filename)
            
            # Garante que o subdiretório existe (ex: '0--Parade')
            os.makedirs(os.path.dirname(label_filepath), exist_ok=True)

            with open(label_filepath, 'w') as label_file:
                label_file.write("\n".join(yolo_annotations))
            
            processed_files += 1
            if processed_files % 500 == 0:
                print(f"Processados {processed_files} arquivos de imagem...")


        # Move o índice para a próxima imagem
        line_idx += num_boxes

    print(f"Conversão concluída para {annotation_file}. Total de {processed_files} arquivos de label criados.")

# --- BLOCO PRINCIPAL PARA EXECUÇÃO ---
if __name__ == "__main__":
    # --- CONFIGURE OS CAMINHOS AQUI ---
    # Supondo que a sua estrutura de pastas é /data/WIDER_train, /data/wider_face_split, etc.
    # e que este script está em uma pasta na raiz do projeto.
    
    BASE_DATA_PATH = "data"

    # Caminhos para o conjunto de TREINO
    TRAIN_ANNOTATION_FILE = os.path.join(BASE_DATA_PATH, "wider_face_split", "wider_face_train_bbx_gt.txt")
    TRAIN_IMG_DIR = os.path.join(BASE_DATA_PATH, "WIDER_train", "images")
    TRAIN_OUTPUT_LABEL_DIR = os.path.join(BASE_DATA_PATH, "WIDER_train", "labels")

    # Caminhos para o conjunto de VALIDAÇÃO
    VAL_ANNOTATION_FILE = os.path.join(BASE_DATA_PATH, "wider_face_split", "wider_face_val_bbx_gt.txt")
    VAL_IMG_DIR = os.path.join(BASE_DATA_PATH, "WIDER_val", "images")
    VAL_OUTPUT_LABEL_DIR = os.path.join(BASE_DATA_PATH, "WIDER_val", "labels")

    print("Iniciando a conversão do conjunto de TREINO...")
    convert_wider_to_yolo(TRAIN_ANNOTATION_FILE, TRAIN_IMG_DIR, TRAIN_OUTPUT_LABEL_DIR)

    print("\nIniciando a conversão do conjunto de VALIDAÇÃO...")
    convert_wider_to_yolo(VAL_ANNOTATION_FILE, VAL_IMG_DIR, VAL_OUTPUT_LABEL_DIR)

    print("\nProcesso de conversão finalizado com sucesso!")