import pandas as pd
from tqdm import tqdm

def parse_wider_face_to_dataframe(annotation_path: str) -> pd.DataFrame:
    """
    Lê o arquivo de anotações do WIDER FACE e o converte para um DataFrame do pandas.

    Args:
        annotation_path (str): O caminho para o arquivo .txt de anotações 
                               (ex: 'wider_face_train_bbx_gt.txt').

    Returns:
        pd.DataFrame: Um DataFrame contendo todas as anotações.
    """
    # Abre o arquivo e lê todas as linhas para a memória
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    all_boxes_data = []
    line_idx = 0
    total_lines = len(lines)

    # Usa tqdm para mostrar uma barra de progresso, já que o arquivo é grande
    with tqdm(total=total_lines, desc="Processando anotações") as pbar:
        while line_idx < total_lines:
            # Pega o nome do arquivo da imagem
            file_name = lines[line_idx].strip()
            line_idx += 1
            pbar.update(1)

            # Pega o número de caixas delimitadoras
            try:
                num_boxes = int(lines[line_idx].strip())
                line_idx += 1
                pbar.update(1)
            except (ValueError, IndexError):
                # Se o número de caixas for inválido ou não existir, pula para a próxima imagem
                continue

            # Se não houver caixas, apenas avança o índice
            if num_boxes == 0:
                # O WIDER FACE tem uma linha extra para 0 caixas, que precisa ser pulada
                line_idx += 1
                pbar.update(1)
                continue

            # Processa cada caixa delimitadora
            for i in range(num_boxes):
                box_line = lines[line_idx].strip().split()
                
                # O formato tem 10 valores: x1, y1, w, h, blur, expression, 
                # illumination, invalid, occlusion, pose
                values = [int(v) for v in box_line]
                
                box_data = {
                    'file_name': file_name,
                    'x1': values[0],
                    'y1': values[1],
                    'w': values[2],
                    'h': values[3],
                    'blur': values[4],
                    'expression': values[5],
                    'illumination': values[6],
                    'invalid': values[7],
                    'occlusion': values[8],
                    'pose': values[9]
                }
                all_boxes_data.append(box_data)
                line_idx += 1
                pbar.update(1)

    # Cria o DataFrame a partir da lista de dicionários
    df = pd.DataFrame(all_boxes_data)
    return df

# --- BLOCO PRINCIPAL PARA EXECUÇÃO ---
if __name__ == "__main__":
    # Configure o caminho para o seu arquivo de anotações aqui
    ANNOTATION_FILE_PATH = 'data/wider_face_split/wider_face_train_bbx_gt.txt'

    print(f"Iniciando o parsing do arquivo: {ANNOTATION_FILE_PATH}")
    
    # Chama a função para criar o DataFrame
    wider_face_df = parse_wider_face_to_dataframe(ANNOTATION_FILE_PATH)

    if not wider_face_df.empty:
        print("\nParsing concluído com sucesso!")
        print("\n--- Informações do DataFrame ---")
        wider_face_df.info()

        print("\n--- 5 Primeiras Linhas do DataFrame ---")
        print(wider_face_df.head())

        print("\n--- Estatísticas Descritivas dos Valores Numéricos ---")
        # Mostra estatísticas para as colunas numéricas
        print(wider_face_df.describe())

        # --- SEÇÃO MODIFICADA PARA SALVAR COMO CSV ---
        output_path = 'wider_face_train_attributes.csv'  # <--- MUDANÇA 1: Extensão do arquivo
        print(f"\nSalvando o DataFrame em {output_path}...")
        wider_face_df.to_csv(output_path, index=False)   # <--- MUDANÇA 2: Função to_csv()
        print("DataFrame salvo com sucesso!")            # <--- MUDANÇA 3: Mensagem (opcional)
    else:
        print("\nO DataFrame resultante está vazio. Verifique o caminho do arquivo.")