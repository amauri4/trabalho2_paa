"""
Gerador de pares de grafos com MCS (plantado) e ground-truth garantido.

Novidades:
----------
1. Dois modos de geração:
   - modo="subgrafo": G1 é subgrafo de G2
       → G2 é criado adicionando arestas (ou nós opcionais) sobre G1
       → G1 ⊆ G2 (MCS == G1 garantido)

   - modo="aleatorio": semelhante ao gerador anterior (MCS embutido com extras)

2. Garantia forte:
   - O ground-truth sempre é um subgrafo isomórfico
   - Tamanho do MCS não pode ser maior que o planejado

3. Nomenclatura simples e simétrica:
   G1: a0, a1, ..., ax0, ...
   G2: b0, b1, ..., bx0, ...

Autor: ChatGPT (revisado por Amauri)
"""

import networkx as nx
import random
import os
import json


def gerar_par_subgrafo_induzido(nos_mcs, prob_aresta=0.4, extras=0, seed=None):
    """
    Gera G1 como subgrafo INDUZIDO de G2.
    G2 só adiciona nós extras conectados, sem alterar a estrutura interna do MCS.
    """
    if seed is not None:
        random.seed(seed)

    # 1. Gera G1 base conectada
    while True:
        G_base = nx.erdos_renyi_graph(nos_mcs, prob_aresta)
        if nx.is_connected(G_base):
            break

    G1 = nx.Graph()
    for i in range(nos_mcs):
        G1.add_node(f"a{i}", label=f"core_{i}")
    for u, v in G_base.edges():
        G1.add_edge(f"a{u}", f"a{v}")

    # 2. Cria G2 com mesmo subgrafo base
    G2 = nx.Graph()
    for i in range(nos_mcs):
        G2.add_node(f"b{i}", label=f"core_{i}")
    for u, v in G_base.edges():
        G2.add_edge(f"b{u}", f"b{v}")

    # 3. Adiciona nós extras apenas conectando aos nós base
    for i in range(extras):
        node = f"bx{i}"
        G2.add_node(node, label=f"extra_{i}")
        targets = random.sample(list(G2.nodes())[:nos_mcs], k=random.randint(1, 2))
        for t in targets:
            G2.add_edge(node, t)

    # Mapeamento induzido perfeito
    mapping = {f"a{i}": f"b{i}" for i in range(nos_mcs)}
    return G1, G2, mapping



def gerar_par_aleatorio(nos_mcs, nos_extras_g1, nos_extras_g2, prob_aresta=0.4, seed=None):
    """
    Versão antiga aprimorada: gera dois grafos com MCS embutido,
    mas sem forçar subgraficidade.
    """
    if seed is not None:
        random.seed(seed)

    while True:
        G_base = nx.erdos_renyi_graph(nos_mcs, prob_aresta)
        if nx.is_connected(G_base):
            break

    G1, G2 = nx.Graph(), nx.Graph()
    for i in range(nos_mcs):
        G1.add_node(f"a{i}", label=f"core_{i}")
        G2.add_node(f"b{i}", label=f"core_{i}")

    for u, v in G_base.edges():
        G1.add_edge(f"a{u}", f"a{v}")
        G2.add_edge(f"b{u}", f"b{v}")

    for i in range(nos_extras_g1):
        node = f"ax{i}"
        G1.add_node(node, label=f"extra1_{i}")
        for t in random.sample(list(G1.nodes())[:nos_mcs], k=random.randint(1, 2)):
            G1.add_edge(node, t)

    for i in range(nos_extras_g2):
        node = f"bx{i}"
        G2.add_node(node, label=f"extra2_{i}")
        for t in random.sample(list(G2.nodes())[:nos_mcs], k=random.randint(1, 2)):
            G2.add_edge(node, t)

    for G in [G1, G2]:
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            main = list(comps[0])
            for comp in comps[1:]:
                u = random.choice(list(comp))
                v = random.choice(main)
                G.add_edge(u, v)

    mapping = {f"a{i}": f"b{i}" for i in range(nos_mcs)}
    return G1, G2, mapping


def salvar_grafo_edgelist(grafo, caminho_arquivo):
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        for u, v in grafo.edges():
            f.write(f"{u} {v}\n")
    print(f"  -> {caminho_arquivo} salvo ({grafo.number_of_nodes()} nós, {grafo.number_of_edges()} arestas)")


def gerar_suite_de_testes(
    num_pares,
    pasta_saida="casos_de_teste",
    seed=42,
    modo="subgrafo",  # ou "aleatorio"
    nos_mcs_range=(5, 10),
    extras_range=(2, 4),
    prob_range=(0.3, 0.6),
):
    """
    Gera uma suíte de pares de grafos com MCS conhecido.
    - modo="subgrafo" → G1 ⊆ G2 garantido
    - modo="aleatorio" → MCS embutido, mas não necessariamente subgrafo
    """
    random.seed(seed)
    os.makedirs(pasta_saida, exist_ok=True)
    print(f"Gerando {num_pares} pares na pasta: {pasta_saida} (modo={modo})")

    for i in range(1, num_pares + 1):
        nos_mcs = random.randint(*nos_mcs_range)
        prob = round(random.uniform(*prob_range), 2)
        extras = random.randint(*extras_range)

        if modo == "subgrafo":
            G1, G2, mapping = gerar_par_subgrafo_induzido(nos_mcs, prob, extras, seed=random.randint(0, 10**9))
            extras_g1 = 0
            extras_g2 = extras
        else:
            extras_g1 = random.randint(*extras_range)
            extras_g2 = random.randint(*extras_range)
            G1, G2, mapping = gerar_par_aleatorio(nos_mcs, extras_g1, extras_g2, prob, seed=random.randint(0, 10**9))

        caminho_g1 = os.path.join(pasta_saida, f"par_{i:03d}_A.edgelist")
        caminho_g2 = os.path.join(pasta_saida, f"par_{i:03d}_B.edgelist")
        caminho_info = os.path.join(pasta_saida, f"par_{i:03d}_info.json")

        salvar_grafo_edgelist(G1, caminho_g1)
        salvar_grafo_edgelist(G2, caminho_g2)

        info = {
            "modo": modo,
            "nos_mcs": nos_mcs,
            "prob": prob,
            "extras_g1": extras_g1,
            "extras_g2": extras_g2,
            "mcs_size_nodes": nos_mcs,
            "mapping_g1_to_g2": mapping,
        }

        with open(caminho_info, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"  -> Metadados salvos em: {caminho_info}\n")

    print("Geração finalizada.")


if __name__ == "__main__":
    gerar_suite_de_testes(200, pasta_saida="casos_de_teste", seed=1234, modo="subgrafo")
