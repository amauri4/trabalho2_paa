import time
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import json
import os
import csv
import statistics

# ==============================================================================
# 1. Estrutura de Grafo 
# ==============================================================================

class SimpleGraph:
    def __init__(self):
        self.nodes = set()
        self.adj = defaultdict(set)

    def add_edge(self, u, v):
        self.nodes.add(u)
        self.nodes.add(v)
        self.adj[u].add(v)
        self.adj[v].add(u)

    def has_edge(self, u, v):
        return v in self.adj.get(u, ())

    def __len__(self):
        return len(self.nodes)

    @classmethod
    def from_edgelist(cls, filepath):
        """Lê um grafo de um arquivo .edgelist. Ignora linhas inválidas."""
        g = cls()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        u, v = parts[0], parts[1]
                        g.add_edge(u, v)
            return g
        except FileNotFoundError:
            print(f"Erro: Arquivo {filepath} não encontrado.")
            exit(1)

# ==============================================================================
# 2. Algoritmo 1: Backtracking Simples (INDUCED)
# ==============================================================================

class SimpleBacktrackingMatcher:
    def __init__(self, g1: 'SimpleGraph', g2: 'SimpleGraph'):
        # Garantir que self.g1 seja o menor (menos nós) para reduzir busca
        if len(g1) > len(g2):
            self.g1, self.g2 = g2, g1
            self.swapped = True
        else:
            self.g1, self.g2 = g1, g2
            self.swapped = False

        self.mapping = {}
        self.best_mapping = {}

    def _is_consistent_induced(self, u1, v2):
        """
        Para todo w já mapeado:
           g1.has_edge(u1, w) == g2.has_edge(v2, mapping[w])
        """
        for w in self.mapping.keys():
            g1_has = w in self.g1.adj[u1] 
            mapped = self.mapping[w]
            g2_has = mapped in self.g2.adj[v2]
            if g1_has != g2_has:
                return False
        return True

    def _solve(self, g1_unmapped, g2_unmapped):
        # atualiza melhor solução
        if len(self.mapping) > len(self.best_mapping):
            self.best_mapping = self.mapping.copy()

        if not g1_unmapped:
            return

        u1 = g1_unmapped[0]
        remaining_g1 = g1_unmapped[1:]

        for i, v2 in enumerate(g2_unmapped):
            if self._is_consistent_induced(u1, v2):
                self.mapping[u1] = v2
                remaining_g2 = g2_unmapped[:i] + g2_unmapped[i+1:]
                self._solve(remaining_g1, remaining_g2)
                del self.mapping[u1]

        # opção de pular u1 (permite subgrafo próprio)
        self._solve(remaining_g1, g2_unmapped)

    def find_mcs_mapping(self):
        self.mapping = {}
        self.best_mapping = {}
        g1_nodes_list = sorted(list(self.g1.nodes))
        g2_nodes_list = sorted(list(self.g2.nodes))
        self._solve(g1_nodes_list, g2_nodes_list)

        if self.swapped:
            # invert mapping para retornar G1->G2 na ordem original
            return {v: k for k, v in self.best_mapping.items()}
        return self.best_mapping

# ==============================================================================
# 3. Algoritmo 2: Branch & Bound 
# ==============================================================================

class VF2Matcher:
    def __init__(self, g1: 'SimpleGraph', g2: 'SimpleGraph'):
        if len(g1) <= len(g2):
            self.g1, self.g2 = g1, g2
            self.swapped = False
        else:
            self.g1, self.g2 = g2, g1
            self.swapped = True

        self.mapping = {}
        self.best_mapping = {}
        self.g1_nodes_list = sorted(self.g1.nodes)
        self.g2_all_nodes = set(self.g2.nodes)

    def _is_consistent_induced(self, u1, v2):
        """
        g1.has_edge(u,w) == g2.has_edge(v,mapping[w])
        """
        for w, mapped in self.mapping.items():
            if self.g1.has_edge(u1, w) != self.g2.has_edge(v2, mapped):
                return False

        # verifica consistência global (entre pares já mapeados)
        for (a, b) in itertools.combinations(self.mapping.keys(), 2):
            a_m, b_m = self.mapping[a], self.mapping[b]
            if self.g1.has_edge(a, b) != self.g2.has_edge(a_m, b_m):
                return False

        return True

    def _solve(self, g1_idx, g2_unmapped):
        # atualiza melhor mapeamento
        if len(self.mapping) > len(self.best_mapping):
            self.best_mapping = self.mapping.copy()

        # poda por limite de tamanho possível
        remaining = len(self.g1_nodes_list) - g1_idx
        if len(self.mapping) + remaining <= len(self.best_mapping):
            return

        if g1_idx >= len(self.g1_nodes_list):
            return

        u1 = self.g1_nodes_list[g1_idx]

        for v2 in sorted(g2_unmapped):
            if self._is_consistent_induced(u1, v2):
                self.mapping[u1] = v2
                g2_unmapped.remove(v2)
                self._solve(g1_idx + 1, g2_unmapped)
                g2_unmapped.add(v2)
                del self.mapping[u1]

        # possibilidade de ignorar este nó
        self._solve(g1_idx + 1, g2_unmapped)

    def find_mcs_mapping(self):
        self.mapping = {}
        self.best_mapping = {}
        self._solve(0, set(self.g2.nodes))
        if self.swapped:
            return {v: k for k, v in self.best_mapping.items()}
        return self.best_mapping

# ==============================================================================
# 4. Clique-based MCS (INDUCED)
# ==============================================================================

def mcs_clique_benchmark(g1_nx: nx.Graph, g2_nx: nx.Graph):
    M = nx.Graph()
    node_pairs = list(itertools.product(g1_nx.nodes(), g2_nx.nodes()))
    M.add_nodes_from(node_pairs)

    for (u1, v1), (u2, v2) in itertools.combinations(node_pairs, 2):
        if u1 == u2 or v1 == v2:
            continue
        g1_has_edge = g1_nx.has_edge(u1, u2)
        g2_has_edge = g2_nx.has_edge(v1, v2)
        if g1_has_edge == g2_has_edge:
            M.add_edge((u1, v1), (u2, v2))

    best_mapping = {}
    best_len = 0

    try:
        # iterar cliques e validar cada uma
        for clique in nx.find_cliques(M):
            # clique é lista de pares (u,v)
            mapping = {u: v for (u, v) in clique}
            # cheque injetividade (deve ser injetiva por construção, mas chequei mesmo assim por uns bugs)
            if len(set(mapping.keys())) != len(mapping) or len(set(mapping.values())) != len(mapping):
                continue

            # valida consistência induzida global
            ok = True
            for (a, b), (c, d) in itertools.combinations(mapping.items(), 2):
                if g1_nx.has_edge(a, c) != g2_nx.has_edge(b, d):
                    ok = False
                    break
            if not ok:
                continue

            if len(mapping) > best_len:
                best_len = len(mapping)
                best_mapping = mapping

        return best_mapping

    except Exception as e:
        print("Erro em clique:", e)
        return {}


# ==============================================================================
# 5. Relatórios, Visualização e Validação com ground-truth
# ==============================================================================

def generate_report(algorithm_name, mapping, duration_sec):
    report_lines = []
    report_lines.append(f"======== RELATÓRIO DO EXPERIMENTO ========")
    report_lines.append(f"Algoritmo: {algorithm_name}")
    report_lines.append(f"Tempo de Execução: {duration_sec:.6f} segundos")
    report_lines.append(f"Tamanho do MCS encontrado (nós): {len(mapping)}")
    report_lines.append("\nMapeamento (G1 -> G2):")

    if not mapping:
        report_lines.append("  (Nenhum mapeamento encontrado)")
    else:
        sorted_mapping = sorted(mapping.items())
        for u, v in sorted_mapping:
            report_lines.append(f"  {u} -> {v}")
    report_lines.append("==========================================")

    report_str = "\n".join(report_lines)
    filename = f"relatorio_{algorithm_name.lower().replace(' ', '_')}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"\nRelatório salvo em: {filename}")
    except IOError as e:
        print(f"Erro ao salvar relatório: {e}")

    return report_str

def visualize_mcs(g1_nx, g2_nx, mapping, title):
    if not mapping:
        print("\nVisualização pulada: Mapeamento vazio.")
        return

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    pos1 = nx.spring_layout(g1_nx, seed=42)
    mapped_nodes_g1 = list(mapping.keys())
    colors_g1 = ['#FF6B6B' if node in mapped_nodes_g1 else '#A0CBE2' for node in g1_nx.nodes()]
    nx.draw(g1_nx, pos1, with_labels=True, node_color=colors_g1, node_size=600, font_size=10)
    plt.title("Grafo 1 (Nós do MCS em vermelho)")

    plt.subplot(1, 2, 2)
    pos2 = nx.spring_layout(g2_nx, seed=42)
    mapped_nodes_g2 = list(mapping.values())
    colors_g2 = ['#FF6B6B' if node in mapped_nodes_g2 else '#A0CBE2' for node in g2_nx.nodes()]
    nx.draw(g2_nx, pos2, with_labels=True, node_color=colors_g2, node_size=600, font_size=10)
    plt.title("Grafo 2 (Nós do MCS em vermelho)")

    plt.suptitle(title, fontsize=14)
    filename = f"visualizacao_{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Visualização salva em: {filename}")
    plt.show()

def load_ground_truth_info(g1_path):
    """
    Procura um arquivo _info.json com o mesmo prefixo de g1_path
    """
    base = os.path.basename(g1_path)
    dirn = os.path.dirname(g1_path) or "."
    prefix = base.split("_A.edgelist")[0]
    info_path = os.path.join(dirn, f"{prefix}_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def validar_mcs_isomorfico(G1, G2, mapping_gt, mapping_encontrado, prefix="mcs_validacao"):
    """
    Compara a solução encontrada com o ground-truth em termos estruturais (isomorfismo).
    - Verifica se têm o mesmo tamanho.
    - Testa se os subgrafos induzidos são isomórficos (ignorando rótulos).
    - Gera visualizações lado a lado e salva em arquivos.
    """

    # subgrafos correspondentes
    sub_gt_G1 = G1.subgraph(mapping_gt.keys()).copy()
    sub_gt_G2 = G2.subgraph(mapping_gt.values()).copy()
    sub_alg_G1 = G1.subgraph(mapping_encontrado.keys()).copy()
    sub_alg_G2 = G2.subgraph(mapping_encontrado.values()).copy()

    size_gt = sub_gt_G1.number_of_nodes()
    size_alg = sub_alg_G1.number_of_nodes()

    # verificação estrutural
    iso_same = nx.is_isomorphic(sub_gt_G1, sub_alg_G1) and nx.is_isomorphic(sub_gt_G2, sub_alg_G2)
    iso_cross = nx.is_isomorphic(sub_gt_G1, sub_alg_G2)  # às vezes o mapeamento é invertido
    iso = iso_same or iso_cross

    # nós faltantes/extras
    missing = {u: v for u, v in mapping_gt.items() if u not in mapping_encontrado}
    extras = {u: v for u, v in mapping_encontrado.items() if u not in mapping_gt}

    print("\n=== Validação Estrutural via Isomorfismo ===")
    print(f"Tamanho GT: {size_gt}")
    print(f"Tamanho Algoritmo: {size_alg}")
    print(f"Mesmo tamanho: {size_gt == size_alg}")
    print(f"Isomórfico: {'✅ SIM' if iso else '❌ NÃO'}")
    print(f"Faltando (GT não encontrado): {missing}")
    print(f"Extras (encontrados mas não no GT): {extras}")
    print("============================================")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    pos1 = nx.spring_layout(sub_gt_G1, seed=42)
    pos2 = nx.spring_layout(sub_alg_G1, seed=42)

    nx.draw(sub_gt_G1, pos1, with_labels=True, node_color="skyblue", edge_color="gray", ax=axes[0])
    axes[0].set_title("Ground Truth (G1)")

    nx.draw(sub_alg_G1, pos2, with_labels=True, node_color="lightgreen", edge_color="gray", ax=axes[1])
    axes[1].set_title("Solução Encontrada (G1)")

    plt.tight_layout()
    fig.savefig(f"{prefix}_comparacao.png", dpi=200)
    plt.close(fig)

    print(f"Visualizações salvas em: {prefix}_comparacao.png")
    print(f"Subgrafos exportados como .edgelist para análise detalhada.\n")

    return {
        "mcs_size_gt": size_gt,
        "mcs_size_alg": size_alg,
        "same_size": size_gt == size_alg,
        "is_isomorphic": iso,
        "missing_in_alg": missing,
        "extra_in_alg": extras,
        "files": {
            "gt_g1": f"{prefix}_subgrafo_GT_G1.edgelist",
            "alg_g1": f"{prefix}_subgrafo_ALG_G1.edgelist",
            "comparacao_img": f"{prefix}_comparacao.png",
        }
    }
# ==============================================================================
# 6. Função Principal (CLI)
# ==============================================================================

def run_single_case(g1_file, g2_file, algo, repetitions=1, no_visual=False):
    """Executa um par de grafos, calcula MCS e valida com ground-truth """
    g1_custom = SimpleGraph.from_edgelist(g1_file)
    g2_custom = SimpleGraph.from_edgelist(g2_file)
    g1_nx = nx.read_edgelist(g1_file)
    g2_nx = nx.read_edgelist(g2_file)

    tempos = []
    mapping = {}

    n_execucoes = repetitions if repetitions and repetitions > 1 else 1

    for _ in range(n_execucoes):
        start_time = time.perf_counter()

        if algo == "backtracking":
            matcher = SimpleBacktrackingMatcher(g1_custom, g2_custom)
            mapping = matcher.find_mcs_mapping()
        elif algo == "vf2":
            matcher = VF2Matcher(g1_custom, g2_custom)
            mapping = matcher.find_mcs_mapping()
        elif algo == "clique":
            mapping = mcs_clique_benchmark(g1_nx, g2_nx)

        tempos.append(time.perf_counter() - start_time)

    duration = statistics.mean(tempos)
    report = generate_report(algo, mapping, duration)
    print(report)

    validation_result = None
    info = load_ground_truth_info(g1_file)
    if info is not None:
        gt_map = info.get("mapping_g1_to_g2")
        if gt_map:
            print("Ground-truth carregado de info.json. Validando (modo isomórfico)...")

            validation_result = validar_mcs_isomorfico(
                g1_nx, g2_nx, gt_map, mapping, prefix="validacao_mcs"
            )

            if not validation_result['is_isomorphic']:
                print(f"Faltando (GT não encontrado): {validation_result['missing_in_alg']}")
                print(f"Extras (encontrados mas não no GT): {validation_result['extra_in_alg']}")
            print("============================================\n")

        else:
            print("Info.json encontrado mas não contém 'mapping_g1_to_g2'.")

    if not no_visual:
        print("Gerando visualização...")
        visualize_mcs(g1_nx, g2_nx, mapping, f"MCS_via_{algo}")

    return {
        "g1_nodes": len(g1_nx.nodes),
        "g2_nodes": len(g2_nx.nodes),
        "g1_edges": len(g1_nx.edges),
        "g2_edges": len(g2_nx.edges),
        "mean_time": duration,
        "mcs_size": len(mapping),
        "validation": validation_result,
    }

def run_benchmark(benchmark_dir, repetitions=5):
    print(f"🔍 Procurando pares de grafos em: {benchmark_dir}")
    files = os.listdir(benchmark_dir)

    g1_files = sorted([f for f in files if f.endswith("_A.edgelist")])
    g2_files = sorted([f for f in files if f.endswith("_B.edgelist")])

    paired = []
    for g1 in g1_files:
        base_prefix = g1[:-len("_A.edgelist")]
        g2_expected = f"{base_prefix}_B.edgelist"
        if g2_expected in g2_files:
            paired.append((
                os.path.join(benchmark_dir, g1),
                os.path.join(benchmark_dir, g2_expected),
                base_prefix
            ))

    if not paired:
        print("⚠️ Nenhum par de grafos encontrado no diretório de benchmark.")
        return

    print(f"✅ {len(paired)} pares encontrados para benchmark.\n")

    algoritmos = ["backtracking", "vf2", "clique"]

    csv_file = "benchmark_results.csv"
    fieldnames = [
        "pair", "algorithm", "g1_nodes", "g2_nodes",
        "g1_edges", "g2_edges", "repetitions", "mean_time", "mcs_size"
    ]
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for g1_path, g2_path, base_prefix in paired:
            pair_name = base_prefix
            print(f"\n▶️ Rodando par: {pair_name}")

            for algo in algoritmos:
                print(f"   ⚙️ Algoritmo: {algo}")
                result = run_single_case(
                    g1_path, g2_path, algo,
                    repetitions=repetitions, no_visual=True
                )

                writer.writerow({
                    "pair": pair_name,
                    "algorithm": algo,
                    "g1_nodes": result["g1_nodes"],
                    "g2_nodes": result["g2_nodes"],
                    "g1_edges": result["g1_edges"],
                    "g2_edges": result["g2_edges"],
                    "repetitions": repetitions,
                    "mean_time": f"{result['mean_time']:.6f}",
                    "mcs_size": result["mcs_size"],
                })

                print(f"      → Média: {result['mean_time']:.6f}s | MCS: {result['mcs_size']} nós")

    print(f"\n✅ Benchmark finalizado! Resultados salvos em '{csv_file}'.")


def main():
    parser = argparse.ArgumentParser(description="Encontra o Máximo Subgrafo Comum (MCS) usando diferentes algoritmos (INDUCED).")
    parser.add_argument("g1_file", nargs="?", help="Caminho para o arquivo .edgelist do Grafo 1 (ou diretório para benchmark).")
    parser.add_argument("g2_file", nargs="?", help="Caminho para o arquivo .edgelist do Grafo 2 (ignorado se benchmark).")
    parser.add_argument(
        "--algo",
        choices=["backtracking", "vf2", "clique"],
        help="O algoritmo a ser usado (backtracking, vf2, clique)."
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Não gerar a visualização gráfica (útil para testes em lote)."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Ativa o modo benchmark (usa o diretório passado em g1_file)."
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Número de repetições no modo benchmark (média)."
    )

    args = parser.parse_args()

    if args.benchmark:
        if not args.g1_file or not os.path.isdir(args.g1_file):
            raise ValueError("Para modo benchmark, informe um diretório válido em g1_file.")
        run_benchmark(args.g1_file, repetitions=args.repeat)
    else:
        if not args.algo:
            parser.error("--algo é obrigatório quando não se está no modo benchmark.")
        if not args.g1_file or not args.g2_file:
            parser.print_help()
            return
        run_single_case(args.g1_file, args.g2_file, args.algo, repetitions=args.benchmark, no_visual=args.no_visual)


if __name__ == "__main__":
    main()
