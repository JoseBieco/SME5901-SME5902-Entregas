
import numpy as np
import re

def parse_model_from_txt(filepath, to_standard_form=False):
    """
    Analisa um arquivo de texto (.txt) contendo um modelo de Programação Linear (PL)
    e o converte para um formato matricial.

    Args:
        filepath (str): O caminho para o arquivo .txt contendo o modelo de PL.
        to_standard_form (bool): Se True, converte o modelo para a forma padrão
                                 (Ax = b, x >= 0) adicionando variáveis de folga/excesso.
                                 Caso contrário, mantém as inequalidades.

    Returns:
        dict: Um dicionário contendo a estrutura do problema de PL processado.
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    objective_line = lines[0]
    constraints_lines = []
    domain_spec_lines = []

    s_t_found = False
    for line in lines[1:]:
        if 's.t.' in line.lower():
            s_t_found = True
            continue
        if not s_t_found: continue

        if any(op in line for op in ['<=', '>=', '=']):
            constraints_lines.append(line)
        else:
            domain_spec_lines.append(line)

    is_max = 'max' in objective_line.lower()
    is_min = 'min' in objective_line.lower()
    if not (is_max or is_min):
        raise ValueError("A função objetivo deve começar com 'max' ou 'min'")

    all_text = ' '.join(lines)
    original_var_names_set = set(re.findall(r'x\d+', all_text))
    if not original_var_names_set:
        raise ValueError("Nenhuma variável (ex: x1, x2) encontrada no modelo.")
    sorted_original_vars = sorted(list(original_var_names_set), key=lambda v: int(v[1:]))

    free_vars, negative_vars = set(), set()
    for line in domain_spec_lines:
        vars_in_line = set(re.findall(r'x\d+', line))
        if 'free' in line.lower():
            free_vars.update(vars_in_line)
        elif 'negative' in line.lower():
            negative_vars.update(vars_in_line)
    if free_vars.intersection(negative_vars):
        raise ValueError(f"Variáveis conflitantes: {free_vars.intersection(negative_vars)}")

    simplex_vars_map, simplex_var_column_names = {}, []
    current_simplex_col = 0
    for var_orig in sorted_original_vars:
        if var_orig in free_vars:
            p, n = f"{var_orig}_p", f"{var_orig}_n"
            simplex_var_column_names.extend([p, n])
            simplex_vars_map[var_orig] = {'type': 'free', 'cols_parser': [current_simplex_col, current_simplex_col + 1], 'mult': 1, 'simplex_names': [p, n]}
            current_simplex_col += 2
        elif var_orig in negative_vars:
            p = f"{var_orig}_prime"
            simplex_var_column_names.append(p)
            simplex_vars_map[var_orig] = {'type': 'negative', 'cols_parser': [current_simplex_col], 'mult': -1, 'simplex_names': [p]}
            current_simplex_col += 1
        else:
            simplex_var_column_names.append(var_orig)
            simplex_vars_map[var_orig] = {'type': 'non_negative', 'cols_parser': [current_simplex_col], 'mult': 1, 'simplex_names': [var_orig]}
            current_simplex_col += 1
    
    num_simplex_vars = len(simplex_var_column_names)
    c_transformed = np.zeros(num_simplex_vars)
    obj_expr = objective_line.lower().replace('max', '').replace('min', '').strip()
    for term in re.finditer(r'([+-]?\s*\d*\.?\d*)\s*(x\d+)', obj_expr):
        coeff_str, var_orig = term.group(1).replace(' ', ''), term.group(2)
        coeff = float(coeff_str) if coeff_str not in ['+', '', '-'] else (1.0 if coeff_str in ['+', ''] else -1.0)
        info = simplex_vars_map[var_orig]
        if info['type'] == 'free':
            c_transformed[info['cols_parser'][0]] += coeff
            c_transformed[info['cols_parser'][1]] -= coeff
        else:
            c_transformed[info['cols_parser'][0]] += coeff * info['mult']

    if is_min: c_transformed = -c_transformed

    A_transformed, b_transformed, signs_transformed = [], [], []
    for line in constraints_lines:
        match = re.match(r'(.+?)\s*(<=|>=|=)\s*([^<>=]+)', line)
        lhs_expr, sign, rhs_str = match.groups()
        signs_transformed.append(sign)
        b_transformed.append(float(rhs_str.strip()))
        row = np.zeros(num_simplex_vars)
        for term in re.finditer(r'([+-]?\s*\d*\.?\d*)\s*(x\d+)', lhs_expr):
            coeff_str, var_orig = term.group(1).replace(' ', ''), term.group(2)
            coeff = float(coeff_str) if coeff_str not in ['+', '', '-'] else (1.0 if coeff_str in ['+', ''] else -1.0)
            info = simplex_vars_map[var_orig]
            if info['type'] == 'free':
                row[info['cols_parser'][0]] += coeff
                row[info['cols_parser'][1]] -= coeff
            else:
                row[info['cols_parser'][0]] += coeff * info['mult']
        A_transformed.append(list(row))

    interpretation_info = {
        'sorted_original_vars': sorted_original_vars,
        'simplex_vars_map': simplex_vars_map,
        'simplex_var_column_names': simplex_var_column_names
    }

    if to_standard_form:
        num_constraints = len(A_transformed)
        slack_surplus_cols = []
        new_var_names = []
        
        for i, sign in enumerate(signs_transformed):
            if sign == '<=':
                col = np.zeros(num_constraints)
                col[i] = 1
                slack_surplus_cols.append(col)
                new_var_names.append(f"s{i+1}")
            elif sign == '>=':
                col = np.zeros(num_constraints)
                col[i] = -1
                slack_surplus_cols.append(col)
                new_var_names.append(f"e{i+1}")
        
        if slack_surplus_cols:
            slack_surplus_matrix = np.array(slack_surplus_cols).T
            A_transformed = np.hstack([np.array(A_transformed), slack_surplus_matrix])
            c_transformed = np.hstack([c_transformed, np.zeros(len(new_var_names))])
            interpretation_info['simplex_var_column_names'].extend(new_var_names)
        
        signs_transformed = ['='] * num_constraints

    return {
        'c': list(c_transformed),
        'A': A_transformed.tolist() if isinstance(A_transformed, np.ndarray) else A_transformed,
        'b': b_transformed,
        'signs': signs_transformed,
        'was_min': is_min,
        'interpretation_info': interpretation_info
    }