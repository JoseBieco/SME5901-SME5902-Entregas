# 📈 Métodos de Otimização

Este repositório contém implementações de algoritmos clássicos de otimização, desenvolvidos como parte das disciplinas de Otimização Linear I e II do Instituto de Ciências Matemáticas e de Computação (ICMC) da USP.

> O objetivo inicial era criar *solvers* robustos para problemas de programação linear, e futuramente o escopo será expandido para incluir também métodos de otimização inteira.

---

## 🚀 Algoritmos Implementados

Atualmente, o repositório conta com as seguintes implementações em Jupyter Notebook:

### 1. Método Simplex (`simplex.ipynb`)
Uma implementação detalhada do Simplex, oferecendo duas abordagens de solução:
- **Simplex Tabular**: O método tradicional com pivoteamento em um *tableau* completo.
- **Simplex Revisado**: Uma versão mais eficiente em memória, que calcula os componentes necessários a cada iteração.
- **Método de Duas Fases**: Integrado para lidar automaticamente com problemas que não possuem uma solução básica factível trivial.

### 2. Método de Pontos Interiores (`PontosInteriores.ipynb`)
Um *solver* que utiliza o método **Primal-Dual de Pontos Interiores**, convergindo para a solução ótima ao seguir um "caminho central".

### 3. Decomposição de Dantzig-Wolfe (`Dantzig-Wolfe.ipynb`)
Implementação do algoritmo para resolver problemas de programação linear com **estrutura de blocos angular**, decompondo um problema grande em um Problema Mestre e múltiplos Subproblemas.

---

## ✨ Funcionalidades e Características

- **Parser de Modelos (`parser.py`)**: Um parser flexível que lê modelos a partir de arquivos de texto (`.txt`) e os prepara para o *solver*. Suporta:
  - Objetivos de maximização ou minimização.
  - Restrições com sinais de `≤`, `≥`, ou `=`.
  - Tratamento avançado de variáveis, incluindo livres e negativas, que são automaticamente convertidas para o formato padrão.

- **Detecção de Casos Especiais**: O *solver* Simplex é capaz de identificar e relatar diversos status da solução, como:
  - ✅ Solução ótima (única ou múltiplas)
  - ⛔ Problema infactível
  - ⚠️ Solução ilimitada
  - 🔄 Degeneração

- **Modelos de Teste**: O diretório `modelos/` contém diversos exemplos para testar os algoritmos, incluindo casos de degeneração, problemas infactíveis e ilimitados.

---
### ⚙️ Como Utilizar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    ```

2.  **Abra um dos Jupyter Notebooks** (ex: `simplex.ipynb`).
    > As implementações estão organizadas em classes. Você pode instanciar um *solver* passando os dados do problema ou utilizar o `parser.py` para carregar um modelo a partir de um arquivo `.txt`.

3.  **Execute as células para ver a solução.** Os notebooks também incluem exemplos e testes com os modelos fornecidos.

---

### 📁 Estrutura do Repositório

```
├── 📄 simplex.ipynb
├── 📄 PontosInteriores.ipynb
├── 📄 Dantzig-Wolfe.ipynb
├── 🐍 parser.py
└── 📂 modelos/
    ├── degenerada_1.txt
    ├── ilimitada_1.txt
    ├── infactivel_1.txt
    └── ... (outros modelos)
```
---

### 🎯 Próximos Passos

- [ ] Adicionar implementações de métodos de Otimização Inteira (ex: *Branch and Bound*).
- [ ] Refatorar os *solvers* para uma biblioteca Python mais modular.
- [ ] Expandir a documentação com mais exemplos teóricos e práticos.