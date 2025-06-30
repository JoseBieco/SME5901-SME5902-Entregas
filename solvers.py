import numpy as np
from scipy.linalg import inv

class Simplex:
    def __init__(self, c_from_parser, A_from_parser, b, signs, was_min=False, interpretation_info=None):
        """
        Inicializa o solver Simplex com os dados do problema de Programação Linear.

        Args:
            c_from_parser (list): Vetor de custos para as variáveis já processadas pelo parser.
            A_from_parser (list): Matriz de coeficientes das restrições para as variáveis do parser.
            b (list): Vetor de termos independentes (lado direito) das restrições.
            signs (list): Lista com os sinais de cada restrição ('<=', '>=', '=').
            was_min (bool, optional): True se o problema original era de minimização. Default é False.
            interpretation_info (dict, optional): Dicionário com dados para mapear a solução de volta às variáveis originais.
        """

        self.c_parser_vars = np.array(c_from_parser, dtype=float)
        self.A_parser_vars = np.array(A_from_parser, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.signs = signs
        self.was_min = was_min
        self.interpretation_info = interpretation_info
        self.m, self.n_parser_vars = self.A_parser_vars.shape

          # Atributos que serão preenchidos durante a preparação do problema
        self.A = None
        self.c = None
        self.b = None
        self.num_vars = None

    def _prepare_problem(self):
        """
            Converte o problema de PL para a Forma Padrão (A'x = b', x >= 0, b' >= 0).

            Esta função realiza as seguintes etapas:
            1. Garante que todos os elementos de 'b' (lado direito) sejam não-negativos,
            multiplicando as restrições por -1 quando necessário.
            2. Adiciona variáveis de folga e excesso para converter as inequalidades
            em igualdades.
            3. Monta a matriz 'A' e o vetor de custos 'c' finais, que incluem
            as variáveis de decisão originais e as de folga/excesso.
        """

        self.b = np.copy(self.b_orig)
        temp_A = np.copy(self.A_parser_vars)
        
        for i in range(self.m):
            if self.b[i] < 0:
                self.b[i] *= -1
                temp_A[i, :] *= -1
                if self.signs[i] == '<=': self.signs[i] = '>='
                elif self.signs[i] == '>=': self.signs[i] = '<='

        # Calcula o número de variáveis de folga/excesso
        num_slack_vars = sum(1 for sign in self.signs if sign != '=')
        self.num_vars = self.n_parser_vars + num_slack_vars
        
        # Cria a matriz A e o vetor c na forma padrão
        self.A = np.zeros((self.m, self.num_vars))
        self.A[:, :self.n_parser_vars] = temp_A
        self.c = np.zeros(self.num_vars)
        self.c[:self.n_parser_vars] = self.c_parser_vars

        # Adiciona as variáveis de folga/excesso
        slack_ptr = self.n_parser_vars
        for i, sign in enumerate(self.signs):
            if sign == '<=':
                self.A[i, slack_ptr] = 1.0
                slack_ptr += 1
            elif sign == '>=':
                self.A[i, slack_ptr] = -1.0
                slack_ptr += 1

    # --------------------------------------------------------------------------
    # FLUXO DE SOLUÇÃO PRINCIPAL
    # --------------------------------------------------------------------------
    def solve(self, method='revised'):
        """
            Ponto de entrada principal para resolver o problema de Programação Linear.

            A função prepara o problema para a forma padrão e, em seguida, seleciona
            o método de solução (Tabular ou Revisado) para encontrar a solução ótima.

            Args:
                method (str, optional): O método a ser usado. Pode ser 'tabular' ou 'revised'.
                                        Default é 'revised'.

            Returns:
                dict: Um dicionário contendo o status final da otimização ('optimal', 
                    'infeasible', 'unbounded', etc.), a solução encontrada (se houver) e 
                    outras informações relevantes.
        """
        self._prepare_problem()
        
        if method == 'revised':
            result = self._solve_revised()
        elif method == 'tabular':
            result = self._solve_tabular()
        else:
            raise ValueError("Método inválido. Escolha 'tabular' ou 'revised'.")

        return self._format_final_solution(result)

    # --------------------------------------------------------------------------
    # LÓGICA PARA O MÉTODO REVISADO
    # --------------------------------------------------------------------------
    def _solve_revised(self):
        """
            Orquestra a solução usando o método Simplex Revisado de duas fases.

            Fase 1: Encontra uma Solução Básica Factível inicial para o problema.
            Fase 2: Usa a base factível da Fase 1 para encontrar a solução ótima.

            Returns:
                dict: Dicionário com o resultado do processo. Se a Fase 1 determinar
                    que o problema é infactível, o status 'infeasible' é retornado.
                    Caso contrário, retorna o resultado da Fase 2.
        """

        # Fase 1: Encontrar uma base factível
        phase1_result = self._run_phase1_revised()
        if phase1_result.get('status') != 'feasible':
            return phase1_result
        
        initial_base_indices = phase1_result['base']
        
         # Fase 2: Encontrar a solução ótima
        return self._revised_simplex_engine(self.A, self.b, self.c, initial_base_indices)

    def _run_phase1_revised(self):
        """
            Executa a Fase 1 do Simplex Revisado para encontrar uma base factível.

            Constrói e resolve um problema auxiliar cujo objetivo é minimizar a soma das
            variáveis artificiais. Se o valor ótimo deste problema for zero, uma
            base factível foi encontrada. Caso contrário, o problema original é infactível.

            Returns:
                dict: Um dicionário com o status.
                    - {'status': 'feasible', 'base': [...]} se uma base factível for encontrada.
                    - {'status': 'infeasible'} se o problema original for infactível.
        """

        # Identifica restrições que precisam de variáveis artificiais
        artificial_rows = {i for i, sign in enumerate(self.signs) if sign in ['>=', '=']}
        if not artificial_rows:
            # Base trivial com vars de folga (se houver)
            slack_indices = [self.n_parser_vars + i for i, s in enumerate(self.signs) if s != '=']
            return {'status': 'feasible', 'base': slack_indices}

        # Constrói o problema da Fase 1
        num_artificial = len(artificial_rows)
        A_phase1 = np.hstack([self.A, np.zeros((self.m, num_artificial))])
        c_phase1 = np.zeros(self.A.shape[1] + num_artificial)
        c_phase1[self.A.shape[1]:] = -1.0
        
        # Adiciona variáveis artificiais e define a base inicial da Fase 1
        initial_base_phase1 = [-1] * self.m
        art_ptr = self.A.shape[1]
        slack_ptr = self.n_parser_vars
        for i in range(self.m):
            if i in artificial_rows:
                A_phase1[i, art_ptr] = 1.0
                initial_base_phase1[i] = art_ptr
                art_ptr += 1
            else: # Restrição <=
                initial_base_phase1[i] = slack_ptr
                slack_ptr += 1
        
        result_phase1 = self._revised_simplex_engine(A_phase1, self.b, c_phase1, initial_base_phase1)

        # Verifica o resultado da Fase 1
        if result_phase1.get('status') != 'optimal' or abs(result_phase1.get('value', 0)) > 1e-9:
            return {'status': 'infeasible'}

        final_base_phase1 = result_phase1['final_basis_indices']
        
        # Verifica se alguma variável artificial permaneceu na base
        if any(b >= self.A.shape[1] for b in final_base_phase1):
             return {'status': 'error_redundant_constraint', 'message': 'Não foi possível expulsar as variáveis artificiais da base. O modelo pode ter restrições redundantes.'}

        return {'status': 'feasible', 'base': final_base_phase1}

    def _revised_simplex_engine(self, A, b, c, initial_basic_indices):
        """
            Motor principal que executa as iterações do algoritmo Simplex Revisado.

            Args:
                A (np.array): A matriz de coeficientes das restrições (forma padrão).
                b (np.array): O vetor do lado direito (forma padrão).
                c (np.array): O vetor de custos (forma padrão).
                initial_basic_indices (list): Lista de índices das variáveis na base inicial.

            Returns:
                dict: Um dicionário descrevendo o resultado da otimização ('optimal', 'unbounded', etc.).
        """
        basic_indices = np.array(initial_basic_indices, dtype=int)
        num_vars = A.shape[1]
        
        for _ in range(self.m * num_vars * 2): # Limite de iterações
            B = A[:, basic_indices]
            try:
                B_inv = inv(B)
            except np.linalg.LinAlgError:
                return {'status': 'error_singular_matrix'}
            
            non_basic_indices = np.setdiff1d(np.arange(num_vars), basic_indices)
            c_b = c[basic_indices]
            x_b = B_inv @ b
            y = c_b @ B_inv
            cj_zj = c[non_basic_indices] - y @ A[:, non_basic_indices]

            # Condição de otimalidade
            if np.all(cj_zj <= 1e-9):
                sol = np.zeros(num_vars)
                sol[basic_indices] = x_b
                return {'status': 'optimal', 'solution': sol, 'value': c_b @ x_b,
                        'is_degenerate': np.any(np.isclose(x_b, 0)),
                        'has_multiple_solutions': np.any(np.isclose(cj_zj, 0)),
                        'final_basis_indices': basic_indices, 'final_B_inv': B_inv}
            
            # Escolha da variável que entra na base
            entering_idx = non_basic_indices[np.argmax(cj_zj)]
            d = B_inv @ A[:, entering_idx]
            
            # Condição de solução ilimitada
            if np.all(d <= 1e-9): return {'status': 'unbounded'}
            
            # Teste da razão para escolher a variável que sai da base
            ratios = np.array([x_b[i] / d[i] if d[i] > 1e-9 else np.inf for i in range(self.m)])
            leaving_row = np.argmin(ratios)
            basic_indices[leaving_row] = entering_idx
        
        return {'status': 'max_iterations_reached'}

    # --------------------------------------------------------------------------
    # LÓGICA PARA O MÉTODO TABULAR
    # --------------------------------------------------------------------------
    def _solve_tabular(self):
        """
            Orquestra a solução usando o método Simplex Tabular de duas fases.

            Fase 1: Constrói um tableau e encontra uma solução básica factível.
            Fase 2: Usa o tableau resultante da Fase 1 para encontrar a solução ótima.

            Returns:
                dict: Dicionário com o resultado do processo. Se a Fase 1 determinar
                    que o problema é infactível, o status 'infeasible' é retornado.
                    Caso contrário, retorna o resultado da Fase 2.
        """

        # Fase 1: Construir e resolver o tableau
        tableau, basic_indices, artificial_indices, status = self._build_and_run_phase1_tabular()
        if status != 'optimal':
            return {'status': status} # Retorna 'infeasible' ou outro erro

        # Fase 2: Preparar e resolver o tableau para a otimização
        tableau, basic_indices = self._prepare_phase2_tableau(tableau, basic_indices, artificial_indices)
        return self._tabular_simplex_engine(tableau, basic_indices, self.c)
        
    def _build_and_run_phase1_tabular(self):
        """
            Cria e resolve o tableau da Fase 1 para encontrar uma base factível.

            Constrói um tableau inicial adicionando variáveis de folga, excesso e artificiais
            conforme necessário. Em seguida, resolve este tableau usando um objetivo auxiliar
            para minimizar as variáveis artificiais.

            Returns:
                tuple: Uma tupla contendo:
                    - tableau (np.array): O tableau final da Fase 1.
                    - basic_indices (list): Os índices da base factível encontrada.
                    - artificial_indices (list): Os índices das variáveis artificiais.
                    - status (str): 'optimal' se factível, 'infeasible' caso contrário.
        """


        num_artificial = sum(1 for sign in self.signs if sign in ['>=', '='])
        num_slack = self.num_vars - self.n_parser_vars
        
        # Monta o tableau da Fase 1
        tableau_width = self.n_parser_vars + num_slack + num_artificial + 1
        tableau = np.zeros((self.m, tableau_width))
        tableau[:, :self.n_parser_vars] = self.A[:, :self.n_parser_vars]
        tableau[:, -1] = self.b
        
        basic_indices = [-1] * self.m
        artificial_indices = []
        
        c_phase1 = np.zeros(tableau_width - 1)
        
        # Preenche o tableau com as variáveis de folga, excesso e artificiais
        slack_ptr = self.n_parser_vars
        art_ptr = self.n_parser_vars + num_slack
        
        for i in range(self.m):
            sign = self.signs[i]
            if sign == '<=':
                tableau[i, slack_ptr] = 1.0
                basic_indices[i] = slack_ptr
                slack_ptr += 1
            else: # >= ou =
                if sign == '>=':
                    tableau[i, slack_ptr] = -1.0
                    slack_ptr += 1
                tableau[i, art_ptr] = 1.0
                basic_indices[i] = art_ptr
                artificial_indices.append(art_ptr)
                c_phase1[art_ptr] = -1.0 # max -w
                art_ptr += 1

        # Resolve o problema da Fase 1
        result = self._tabular_simplex_engine(tableau, basic_indices, c_phase1)
        
        # Verifica se a Fase 1 encontrou uma solução factível
        if result.get('status') != 'optimal' or abs(result.get('value', 0)) > 1e-9:
            return None, None, None, 'infeasible'
        
        return result['tableau'], result['final_basis_indices'], artificial_indices, 'optimal'

    def _prepare_phase2_tableau(self, tableau, basic_indices, artificial_indices):
        """
            Modifica o tableau final da Fase 1 para iniciar a Fase 2.

            A principal tarefa é remover as colunas correspondentes às variáveis
            artificiais e ajustar os índices da base de acordo.

            Args:
                tableau (np.array): O tableau resultante da Fase 1.
                basic_indices (list): A lista de índices da base da Fase 1.
                artificial_indices (list): A lista de índices das variáveis artificiais.

            Returns:
                tuple: Uma tupla contendo o novo tableau e a nova lista de índices da base.
        """
        # Remove colunas das variáveis artificiais
        cols_to_keep = [i for i in range(tableau.shape[1] - 1) if i not in artificial_indices]
        tableau = tableau[:, cols_to_keep + [tableau.shape[1] - 1]]
        
        # Mapeia os índices da base antigos para os novos
        map_old_to_new = {old: new for new, old in enumerate(cols_to_keep)}
        new_basic_indices = [map_old_to_new[b] for b in basic_indices]
        
        return tableau, new_basic_indices

    def _tabular_simplex_engine(self, tableau_init, basic_indices_init, c_original):
        """
            Motor principal que executa as iterações do algoritmo Simplex Tabular.

            Args:
                tableau_init (np.array): O tableau inicial para a fase atual.
                basic_indices_init (list): A lista inicial de índices da base.
                c_original (np.array): O vetor de custos original para esta fase.

            Returns:
                dict: Um dicionário descrevendo o resultado da otimização ('optimal', 'unbounded', etc.).
        """
        tableau = np.copy(tableau_init)
        basic_indices = list(basic_indices_init)
        num_vars = tableau.shape[1] - 1
        
        # Limite de iterações para evitar loops infinitos
        for _ in range(self.m * num_vars * 2):
            # Calcula os custos reduzidos (linha cj - zj)
            cb = c_original[basic_indices]
            zj = cb @ tableau[:, :-1]
            cj_zj = c_original - zj

            # Zera os custos reduzidos das variáveis básicas por precisão numérica
            cj_zj[basic_indices] = 0

           # Condição de otimalidade
            if np.all(cj_zj <= 1e-9):
                solution = np.zeros(num_vars)
                for i, idx in enumerate(basic_indices):
                    solution[idx] = tableau[i, -1]
                
                # Checa por múltiplas soluções nas variáveis NÃO básicas
                non_basic_indices = np.setdiff1d(np.arange(num_vars), basic_indices)
                has_multiple_solutions = np.any(np.isclose(cj_zj[non_basic_indices], 0))

                return {
                    'status': 'optimal',
                    'solution': solution,
                    'value': cb @ solution[basic_indices],
                    'is_degenerate': np.any(np.isclose(tableau[:, -1], 0)),
                    'has_multiple_solutions': has_multiple_solutions,
                    'tableau': tableau,
                    'final_basis_indices': basic_indices
                }

            # Escolhe a variável para entrar na base
            entering_col = np.argmax(cj_zj)
            
            # Condição de solução ilimitada
            if np.all(tableau[:, entering_col] <= 1e-9):
                return {'status': 'unbounded'}

            # Teste da razão para escolher a variável que sai da base
            ratios = np.array([tableau[i, -1] / tableau[i, entering_col] if tableau[i, entering_col] > 1e-9 else np.inf for i in range(self.m)])
            leaving_row = np.argmin(ratios)
            
            # Realiza o pivoteamento para atualizar o tableau
            pivot_element = tableau[leaving_row, entering_col]
            if abs(pivot_element) < 1e-9:
                return {'status': 'error_numerical_instability'}
            
            tableau[leaving_row, :] /= pivot_element
            for i in range(self.m):
                if i != leaving_row:
                    tableau[i, :] -= tableau[i, entering_col] * tableau[leaving_row, :]
            
            # Atualiza a base
            basic_indices[leaving_row] = entering_col

        return {'status': 'max_iterations_reached'}

    # --------------------------------------------------------------------------
    # PÓS-PROCESSAMENTO
    # --------------------------------------------------------------------------
    def _format_final_solution(self, result):
        """
        Formata o dicionário de resultado bruto em uma saída final para o usuário.

        Esta função traduz a solução, que está em termos de variáveis internas do 
        Simplex (incluindo folga, etc.), de volta para as variáveis originais do problema.
        Também ajusta o valor final da função objetivo se o problema era de minimização.

        Args:
            result (dict): O dicionário de resultado bruto dos motores Simplex.

        Returns:
            dict: O dicionário final formatado, contendo a solução em termos das
                  variáveis originais, o valor ótimo e o status.
        """
        # Se o resultado não for ótimo, retorna diretamente
        if result.get('status') != 'optimal':
            return result
        
        if self.interpretation_info == None:
            return result

        sol_vector = result['solution']
        final_sol = {}

        # Usa o 'interpretation_info' para reverter a solução para as variáveis originais
        for var, info in self.interpretation_info['simplex_vars_map'].items():
            cols = info['cols_parser']
            if info['type'] == 'free':
                final_sol[var] = sol_vector[cols[0]] - sol_vector[cols[1]]
            else:
                final_sol[var] = sol_vector[cols[0]] * info['mult']
        
        ordered_sol = [final_sol[v] for v in self.interpretation_info['sorted_original_vars']]
        
        # Recalcula o valor da função objetivo com base nos coeficientes do problema original
        final_value = self.c_parser_vars @ ordered_sol
        
        return {'status': 'optimal', 'solution': ordered_sol,
                'value': -final_value if self.was_min else final_value,
                'is_degenerate': result.get('is_degenerate', False),
                'has_multiple_solutions': result.get('has_multiple_solutions', False)}
