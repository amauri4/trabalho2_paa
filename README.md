# Análise de Algoritmos MCS 

Este projeto realiza a **busca do Máximo Subgrafo Comum (MCS) Induzido** entre dois grafos, permitindo comparar diferentes **algoritmos exatos** de forma prática e reprodutível.

---

## Modos de Operação

### 1️⃣ Execução Única
Compara **dois grafos específicos** (`.edgelist`) usando um algoritmo à sua escolha.
Ao fazer a execução única, um relatório .txt com o nome do algoritmo escolhido e uma imagem da solução encontrada serão gerados na raiz do repositório. **Deve-se salvar caso não queira que esses sejam sobrescritos na próxima execução do algoritmo.**

**Uso:**
```bash
python main.py grafoA.edgelist grafoB.edgelist --algo=<backtracking|vf2|clique>
# Deve-se informar a pasta com base no diretório raiz deste repositório (olhar os exemplos)
```

### 2️⃣ Modo Benchmark 
Executa todos os algoritmos em pares de grafos encontrados em um diretório e gera um relatório de desempenho (benchmark_results.csv).
```bash
python main.py ./casos_de_teste --benchmark [--repeat=N]
#--repeat: número de repetições por algoritmo (padrão: 5)
```

### 💡 Exemplos de Uso

Execução única com visualização (padrão)

```bash 
python main.py casos_de_teste/par_001_A.edgelist casos_de_teste/par_001_B.edgelist --algo=backtracking
```

Execução única sem visualização 
```bash
python main.py casos_de_teste/par_001_A.edgelist casos_de_teste/par_001_B.edgelist --algo=vf2 --no-visual
```

Benchmark Padrão 
```bash
python main.py ./casos_de_teste --benchmark
```

Benchmark com 10 repetições por algoritmo
```bash
python main.py ./casos_de_teste --benchmark --repeat=10
```
