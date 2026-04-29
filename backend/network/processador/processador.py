# -------------------------------------
# Pré-processamento de imagens de grãos (Versão Final Definitiva)
#
# Combina o fluxo de trabalho automatizado com as funções aprimoradas do usuário:
# - Recorte com fundo transparente (canal Alfa).
# - Comparação de similaridade (SSIM) inteligente, ignorando o fundo.
# -------------------------------------
import os
import cv2
import numpy as np
import rawpy
from shutil import move
from skimage.metrics import structural_similarity as ssim

# --- 1. CONFIGURAÇÃO DOS DIRETÓRIOS ---
BASE_DIR = os.getcwd()
ORIGINAIS_DIR = os.path.join(BASE_DIR, "1_imagens_originais")
TRATADAS_DIR = os.path.join(BASE_DIR, "2_imagens_tratadas")
RECORTADOS_DIR = os.path.join(BASE_DIR, "3_graos_recortados")
OUTLIERS_DIR = os.path.join(BASE_DIR, "4_outliers")
REFERENCIA_DIR = os.path.join(BASE_DIR, "5_grao_referencia")

# --- 2. FUNÇÕES DE PROCESSAMENTO ---

def ler_imagem(path):
    """Lê imagens nos formatos JPG, PNG ou NEF."""
    try:
        if path.lower().endswith(".nef"):
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            # Usa IMREAD_UNCHANGED para carregar também o canal alfa, se existir
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: raise FileNotFoundError
            return img
    except Exception:
        print(f"    [ERRO] Falha ao ler a imagem: {path}")
        return None

def tratar_imagem(imagem_original_path):
    """Cria a máscara binária usando a lógica de conversão CMYK."""
    print(f"  [Passo 1/3] Tratando a imagem: {os.path.basename(imagem_original_path)}")
    os.makedirs(TRATADAS_DIR, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(imagem_original_path))[0]
    output_path = os.path.join(TRATADAS_DIR, f"{nome_base}_mascara.png")
    
    # Lê a imagem como BGR (3 canais) para o tratamento
    rgb = cv2.imread(imagem_original_path, cv2.IMREAD_COLOR)
    if rgb is None: return None

    rgbdash = rgb.astype(np.float32) / 255.
    K = 1 - np.max(rgbdash, axis=2)
    K[K == 1] = 1 - 1e-6
    C = (1 - rgbdash[..., 2] - K) / (1 - K)
    M = (1 - rgbdash[..., 1] - K) / (1 - K)
    Y = (1 - rgbdash[..., 0] - K) / (1 - K)
    CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)
    _, _, Y_channel, _ = cv2.split(CMYK)
    _, mascara = cv2.threshold(Y_channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    cv2.imwrite(output_path, mascara)
    print(f"    Máscara salva em: {output_path}")
    return output_path

# --- SUAS NOVAS FUNÇÕES INTEGRADAS ---

def separar_graos(imagem_tratada_path, imagem_original_path, tipo_grao):
    """Recorta cada grão individual com fundo transparente (alfa)."""
    print(f"  [Passo 2/3] Separando todos os objetos detectados...")

    dir_graos_temp = os.path.join(RECORTADOS_DIR, tipo_grao, "_temp")
    os.makedirs(dir_graos_temp, exist_ok=True)

    mascara_geral = cv2.imread(imagem_tratada_path, cv2.IMREAD_GRAYSCALE)
    original = ler_imagem(imagem_original_path)
    if mascara_geral is None or original is None:
        return 0, 0

    contornos, _ = cv2.findContours(mascara_geral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    nome_base = os.path.splitext(os.path.basename(imagem_original_path))[0]
    contador_objetos = 0
    max_dim_local = 0

    for i, contorno in enumerate(contornos):
        if cv2.contourArea(contorno) < 100:
            continue
        x, y, w, h = cv2.boundingRect(contorno)
        grao_recortado = original[y:y+h, x:x+w]
        if grao_recortado.size == 0:
            continue
        mascara_local = np.zeros(grao_recortado.shape[:2], dtype=np.uint8)
        contorno_local = contorno - [x, y]
        cv2.drawContours(mascara_local, [contorno_local], -1, (255), thickness=cv2.FILLED)

        # Cria imagem com canal alfa
        b, g, r = cv2.split(grao_recortado)
        grao_transparente = cv2.merge((b, g, r, mascara_local))

        output_path = os.path.join(dir_graos_temp, f"{nome_base}_obj_{i+1}.png")
        cv2.imwrite(output_path, grao_transparente)
        contador_objetos += 1
        max_dim_local = max(max_dim_local, w, h)

    print(f"    [INFO] {contador_objetos} objetos recortados e salvos temporariamente.")
    return contador_objetos, max_dim_local

def pad_and_resize(image, global_max_dim, target_size=224):
    """
    Coloca a imagem em um canvas quadrado do tamanho de global_max_dim (preenchimento transparente)
    para centralizar de forma proporcional, depois redimensiona para target_size x target_size.
    Assume que a imagem e BGRA (4 canais).
    """
    h, w = image.shape[:2]
    # Cria canvas transparente BGRA global_max_dim x global_max_dim
    canvas = np.zeros((global_max_dim, global_max_dim, 4), dtype=np.uint8)
    
    # Evita erros se a dimensao for maior por arredondamentos anormais
    if h > global_max_dim or w > global_max_dim:
        scale = global_max_dim / float(max(h, w))
        image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        
    start_y = (global_max_dim - h) // 2
    start_x = (global_max_dim - w) // 2
    
    # Assegura que image tem 4 canais
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
    canvas[start_y:start_y+h, start_x:start_x+w] = image
    
    # Redimensiona mantendo a nova proporcao quadrada no tamanho final
    resized = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized

def ssim_masked(imgA, imgB):
    """Calcula SSIM apenas sobre os pixels visíveis (alpha > 0)."""
    if imgA is None or imgB is None: return 0.0

    # Garante que as imagens tenham 4 canais (BGRA)
    if imgA.shape[2] == 3:
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2BGRA)
    if imgB.shape[2] == 3:
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2BGRA)

    # Pega a máscara do canal alfa de cada imagem
    maskA = imgA[:, :, 3] > 0
    maskB = imgB[:, :, 3] > 0
    # A máscara final é a intersecção: onde AMBAS as imagens são visíveis
    final_mask = np.logical_and(maskA, maskB)

    # Converte as partes coloridas para escala de cinza
    grayA = cv2.cvtColor(imgA[:, :, :3], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB[:, :, :3], cv2.COLOR_BGR2GRAY)
    # Assegura que as imagens tenham o mesmo tamanho (função chamadora normalmente redimensiona)
    if grayA.shape != grayB.shape:
        # redimensiona B para A
        grayB = cv2.resize(grayB, (grayA.shape[1], grayA.shape[0]), interpolation=cv2.INTER_AREA)
        final_mask = cv2.resize(final_mask.astype(np.uint8), (grayA.shape[1], grayA.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Determina dinamicamente um win_size adequado (padrão 7)
    min_side = min(grayA.shape[0], grayA.shape[1])
    desired_win = 7
    if min_side >= desired_win:
        win_size = desired_win
    else:
        # win_size precisa ser ímpar e >=3
        win_size = min_side if (min_side % 2 == 1) else (min_side - 1)

    # Se win_size for menor que 3, a SSIM por janela não é confiável — usamos um fallback simples
    if win_size < 3:
        # Fallback: similaridade baseada em diferença média normalizada (0..1)
        diff = np.mean(np.abs(grayA.astype(np.float32) - grayB.astype(np.float32)))
        fallback_score = max(0.0, 1.0 - (diff / 255.0))
        return float(fallback_score)

    # Calcula SSIM apenas na área da máscara final, com tratamento de exceções caso a lib levante ValueError
    try:
        score, _ = ssim(grayA, grayB, win_size=win_size, full=True, data_range=255, mask=final_mask)
        return float(score)
    except ValueError:
        # Se ainda houver problemas (por exemplo versões diferentes da skimage), usar fallback
        diff = np.mean(np.abs(grayA.astype(np.float32) - grayB.astype(np.float32)))
        fallback_score = max(0.0, 1.0 - (diff / 255.0))
        return float(fallback_score)

def achar_outliers(tipo_grao, grao_base_controle, global_max_dim, debug=False):
    """Filtra por tamanho e similaridade (SSIM), faz pad & resize finais."""
    print(f"  [Passo 3/3] Verificando outliers e formatando na categoria '{tipo_grao}'...")

    dir_origem = os.path.join(RECORTADOS_DIR, tipo_grao, "_temp")
    dir_destino_outliers = os.path.join(OUTLIERS_DIR, tipo_grao)
    dir_destino_bons = os.path.join(RECORTADOS_DIR, tipo_grao)
    os.makedirs(dir_destino_outliers, exist_ok=True)
    os.makedirs(dir_destino_bons, exist_ok=True)

    if not os.path.isdir(dir_origem):
        print(f"    [AVISO] Diretório temporário não encontrado: {dir_origem}")
        return

    base = grao_base_controle
    if base is None:
        print("    [ERRO] Grão de referência inválido.")
        return

    dy, dx = base.shape[0], base.shape[1]
    THRESH_SSIM = 0.60  # Limiar de similaridade. Ajuste conforme necessário.

    for nome_arquivo in list(os.listdir(dir_origem)):
        caminho_origem = os.path.join(dir_origem, nome_arquivo)
        if not nome_arquivo.lower().endswith('.png'): continue

        grao_atual = cv2.imread(caminho_origem, cv2.IMREAD_UNCHANGED)
        if grao_atual is None: continue

        # Filtro de tamanho
        if grao_atual.shape[0] < 80 or grao_atual.shape[1] < 80:
            move(caminho_origem, os.path.join(dir_destino_outliers, nome_arquivo))
            if debug: print(f"    [Tamanho] {nome_arquivo} movido (muito pequeno).")
            continue

        # Redimensiona para comparar
        grao_redimensionado = cv2.resize(grao_atual, (dx, dy), interpolation=cv2.INTER_AREA)
        
        score = ssim_masked(base, grao_redimensionado)

        if debug: print(f"    [DEBUG] {nome_arquivo} -> SSIM_masked={score:.3f}")

        if score <= THRESH_SSIM:
            move(caminho_origem, os.path.join(dir_destino_outliers, nome_arquivo))
            if debug: print(f"    [Outlier] {nome_arquivo} movido (score={score:.3f}).")
        else:
            # Se nao for outlier, formata o grao (Preenchimento proporcional)
            grao_padronizado = pad_and_resize(grao_atual, global_max_dim, target_size=224)
            # Salva no dir final
            cv2.imwrite(os.path.join(dir_destino_bons, nome_arquivo), grao_padronizado)
            # Remove o temporario
            os.remove(caminho_origem)

    # Tenta remover a pasta temporária, que agora deve estar vazia
    try:
        os.rmdir(dir_origem)
    except OSError:
        print(f"    [AVISO] A pasta temporária '{dir_origem}' não está vazia. Verifique manualmente.")

# --- 3. FUNÇÃO PRINCIPAL DE EXECUÇÃO ---
def main():
    """Orquestra todo o processo de forma automatizada em duas etapas."""
    print("--- INICIANDO PROCESSAMENTO DE IMAGENS DE SOJA (VERSÃO COM PADDING E SCALING PROPORCIONAL) ---")

    caminho_base = os.path.join(REFERENCIA_DIR, "grao_base.png")
    if not os.path.exists(caminho_base):
        print(f"\n[ERRO] Imagem de referência não encontrada em: '{caminho_base}'")
        return
    # Carrega a imagem base com suporte a transparência
    grao_base_controle = cv2.imread(caminho_base, cv2.IMREAD_UNCHANGED)
    print(f"Grão de referência '{caminho_base}' carregado com sucesso.")

    if not os.path.isdir(ORIGINAIS_DIR):
        print(f"\n[ERRO] O diretório '1_imagens_originais' não existe ou está vazio.")
        return

    # A primeira fase vai mapear todos os recortes brutos a fim de estabelecer a escala referencial
    global_max_dim = max(grao_base_controle.shape[:2]) # comeca assumindo as dimensoes da ref
    print("\n--- FASE 1: Extraindo grãos isolados e determinando maior dimensão ---")
    
    for tipo_grao in os.listdir(ORIGINAIS_DIR):
        dir_categoria = os.path.join(ORIGINAIS_DIR, tipo_grao)
        if os.path.isdir(dir_categoria):
            print(f"\nProcessando extracoes da categoria: '{tipo_grao}'")
            
            arquivos_processados = set()
            todos_os_arquivos = os.listdir(dir_categoria)
            arquivos_nef = [f for f in todos_os_arquivos if f.lower().endswith('.nef')]
            outros_arquivos = [f for f in todos_os_arquivos if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for nome_imagem in arquivos_nef + outros_arquivos:
                nome_base = os.path.splitext(nome_imagem)[0]
                if nome_base in arquivos_processados: continue
                
                imagem_original_path = os.path.join(dir_categoria, nome_imagem)
                imagem_tratada_path = tratar_imagem(imagem_original_path)
                if imagem_tratada_path:
                    obj, m_dim = separar_graos(imagem_tratada_path, imagem_original_path, tipo_grao)
                    global_max_dim = max(global_max_dim, m_dim)
                
                arquivos_processados.add(nome_base)
                
    print(f"\n--- FASE 2: Filtrando Outliers e Padronizando ---")
    print(f"Maior dimensao identificada (*canvas padronizado*): {global_max_dim} px")
            
    for tipo_grao in os.listdir(ORIGINAIS_DIR):
        dir_categoria = os.path.join(ORIGINAIS_DIR, tipo_grao)
        if os.path.isdir(dir_categoria):
            # Chama a funcao de outlier aprimorada, agora com pad_and_resize (target 224x224)
            achar_outliers(tipo_grao, grao_base_controle, global_max_dim, debug=True)

    print("\n--- PROCESSAMENTO FINALIZADO ---")

if __name__ == "__main__":
    main()
