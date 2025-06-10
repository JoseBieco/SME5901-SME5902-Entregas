
import numpy as np
import re

def parse_model_from_txt(filepath):
    """
    Analisa um arquivo de texto (.txt) contendo um modelo de Programação Linear (PL)
    e o converte para um formato matricial adequado para o solver Simplex.

    A função extrai a função objetivo, as restrições e as especificações de
    domínio das variáveis (padrão >= 0, livre ou negativa). Ela transforma
    automaticamente o modelo para a forma padrão, onde todas as variáveis de
    decisão são não-negativas, e prepara um dicionário com todas as informações
    necessárias para a resolução e posterior interpretação dos resultados.

    Formato do Arquivo de Modelo Esperado:
    - A primeira linha deve ser a função objetivo (ex: "max 3x1 + 5x2").
    - A linha "s.t." (subject to) deve separar o objetivo das restrições.
    - Cada restrição deve estar em uma nova linha (ex: "x1 + 2x2 <= 10").
    - Sinais aceitos para restrições: '<=', '>=', '='.
    - Especificações de domínio são opcionais e podem vir antes ou depois das
      restrições (ex: "x1 free", "x3 negative"). Variáveis não especificadas
      são consideradas não-negativas (>= 0) por padrão.

    Args:
        filepath (str): O caminho para o arquivo .txt contendo o modelo de PL.

    Returns:
        dict: Um dicionário contendo a estrutura do problema de PL processado.
              Este dicionário está pronto para ser passado como argumento para a
              classe Simplex. A estrutura detalhada deste dicionário de retorno
              é explicada na seção "Análise do Dicionário de Retorno".
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    objective_line = lines[0]
    constraints_lines = []
    domain_spec_lines = [] # Para linhas como "x1 free", "x2 negative"

    parsing_constraints = True # Flag para diferenciar constraints de outras especificações
    # Identificar s.t. e separar as linhas
    s_t_found = False
    for line in lines[1:]:
        if 's.t.' in line.lower():
            s_t_found = True
            continue
        if not s_t_found: # Linhas antes de s.t. que não são objetivo (talvez comentários ou domínios globais)
            domain_spec_lines.append(line) # Ou tratar como erro/ignorar
            continue

        # Após s.t.
        if any(op in line for op in ['<=', '>=', '=']):
            constraints_lines.append(line)
        else:
            # Linhas após s.t. que não são restrições são consideradas especificações de domínio
            domain_spec_lines.append(line)

    is_max = 'max' in objective_line.lower()
    is_min = 'min' in objective_line.lower()
    if not (is_max or is_min):
        raise ValueError("A função objetivo deve começar com 'max' ou 'min'")

    # 1. Identificar todas as variáveis originais x_i mencionadas no modelo
    all_text = ' '.join(lines)
    original_var_names_set = set(re.findall(r'x\d+', all_text))
    if not original_var_names_set:
        raise ValueError("Nenhuma variável (ex: x1, x2) encontrada no modelo.")
    sorted_original_vars = sorted(list(original_var_names_set), key=lambda v: int(v[1:]))


    # 2. Processar linhas de especificação de domínio
    free_vars = set()
    negative_vars = set() # Para x_i <= 0

    for line in domain_spec_lines:
        vars_in_line = set(re.findall(r'x\d+', line))
        if not vars_in_line:
            continue
        
        line_lower = line.lower()
        if 'free' in line_lower:
            free_vars.update(vars_in_line)
        elif 'negative' in line_lower: # Palavra-chave para x_i <= 0
            negative_vars.update(vars_in_line)
        # Poderia adicionar "non-negative" ou "positive" se precisasse de declaração explícita,
        # mas o padrão é não-negativo.

    # Checagem de conflitos (ex: uma variável não pode ser free e negative)
    if free_vars.intersection(negative_vars):
        conflicting_vars = free_vars.intersection(negative_vars)
        raise ValueError(f"Variáveis declaradas como 'free' e 'negative' ao mesmo tempo: {conflicting_vars}")

    # 3. Mapear variáveis originais para variáveis do Simplex (todas >= 0)
    #    e determinar seus multiplicadores de coeficiente.
    
    # simplex_vars_map: {'x1': {'type':'non_negative', 'cols_parser': [0], 'mult': 1, 'simplex_names': ['x1']}, ...}
    simplex_vars_map = {}
    simplex_var_column_names = [] # Nomes das colunas no A_transformed, c_transformed
    current_simplex_col = 0

    for var_orig in sorted_original_vars:
        if var_orig in free_vars:
            p_name = f"{var_orig}_p"
            n_name = f"{var_orig}_n"
            simplex_var_column_names.extend([p_name, n_name])
            simplex_vars_map[var_orig] = {
                'type': 'free',
                'cols_parser': [current_simplex_col, current_simplex_col + 1],
                'mult': 1, # Não usado diretamente para 'free', pois os coefs são divididos
                'simplex_names': [p_name, n_name]
            }
            current_simplex_col += 2
        elif var_orig in negative_vars:
            prime_name = f"{var_orig}_prime" # x_orig = -x_prime ONDE x_prime >= 0
            simplex_var_column_names.append(prime_name)
            simplex_vars_map[var_orig] = {
                'type': 'negative',
                'cols_parser': [current_simplex_col],
                'mult': -1, # Coeficientes de x_orig são multiplicados por -1 para x_prime
                'simplex_names': [prime_name]
            }
            current_simplex_col += 1
        else: # Default: não-negativa
            simplex_var_column_names.append(var_orig)
            simplex_vars_map[var_orig] = {
                'type': 'non_negative',
                'cols_parser': [current_simplex_col],
                'mult': 1,
                'simplex_names': [var_orig]
            }
            current_simplex_col += 1
            
    num_simplex_vars = len(simplex_var_column_names)

    # 4. Construir vetor de custos `c_transformed` para as variáveis do Simplex
    c_transformed = np.zeros(num_simplex_vars)
    obj_expr = objective_line.lower().replace('max', '').replace('min', '').strip()
    
    for term_match in re.finditer(r'([+-]?\s*\d*\.?\d*)\s*(x\d+)', obj_expr):
        coeff_str = term_match.group(1).replace(' ', '')
        var_orig = term_match.group(2)

        if coeff_str == '' or coeff_str == '+':
            coeff_val = 1.0
        elif coeff_str == '-':
            coeff_val = -1.0
        else:
            coeff_val = float(coeff_str)

        if var_orig not in simplex_vars_map: continue # Variável no obj não reconhecida

        map_info = simplex_vars_map[var_orig]
        cols = map_info['cols_parser']

        if map_info['type'] == 'free':
            c_transformed[cols[0]] += coeff_val
            c_transformed[cols[1]] -= coeff_val # Para -x_n
        else: # non_negative ou negative
            # Para x_negativo (x = -x'), se o termo é C*x, vira C*(-x') = (-C)*x'
            # O multiplicador já cuida disso: coeff_val * map_info['mult']
            c_transformed[cols[0]] += coeff_val * map_info['mult']
            
    if is_min:
        c_transformed = -c_transformed # Converte min para max -Z

    # 5. Construir matriz de restrições `A_transformed`
    A_transformed = []
    b_transformed = []
    signs_transformed = []

    for constr_line in constraints_lines:
        # Separar LHS, sinal, e RHS
        match = re.match(r'(.+?)\s*(<=|>=|=)\s*([^<>=]+)', constr_line)
        if not match:
            raise ValueError(f"Formato de restrição inválido: {constr_line}")
        
        lhs_expr, sign, rhs_str = match.groups()
        signs_transformed.append(sign)
        b_transformed.append(float(rhs_str.strip()))
        
        row_coeffs = np.zeros(num_simplex_vars)
        for term_match in re.finditer(r'([+-]?\s*\d*\.?\d*)\s*(x\d+)', lhs_expr):
            coeff_str = term_match.group(1).replace(' ', '')
            var_orig = term_match.group(2)

            if coeff_str == '' or coeff_str == '+':
                coeff_val = 1.0
            elif coeff_str == '-':
                coeff_val = -1.0
            else:
                coeff_val = float(coeff_str)

            if var_orig not in simplex_vars_map: continue

            map_info = simplex_vars_map[var_orig]
            cols = map_info['cols_parser']

            if map_info['type'] == 'free':
                row_coeffs[cols[0]] += coeff_val
                row_coeffs[cols[1]] -= coeff_val
            else: # non_negative ou negative
                row_coeffs[cols[0]] += coeff_val * map_info['mult']
        A_transformed.append(list(row_coeffs))

    # Informações para reverter a solução para as variáveis originais
    interpretation_info = {
        'sorted_original_vars': sorted_original_vars,
        'simplex_vars_map': simplex_vars_map, # Contém tipo, colunas e multiplicador
        'simplex_var_column_names': simplex_var_column_names # Nomes das vars do simplex
    }

    return {
        'c': list(c_transformed), # Coeficientes para as variáveis do Simplex
        'A': A_transformed,       # Matriz A para as variáveis do Simplex
        'b': b_transformed,
        'signs': signs_transformed,
        'was_min': is_min,
        'interpretation_info': interpretation_info
    }