#!/usr/bin/env python3
"""Compara duas capturas do Barenn; tolerâncias devem ser informadas pelo usuário."""
import argparse
import math
import sys
from pathlib import Path


def chaves_esperadas(n_teste):
    keys = set()
    camadas = {"l1": (32, 64), "l2": (16, 32), "l3": (10, 16)}

    def vetor(prefixo, n):
        keys.update(f"{prefixo}/{i}" for i in range(n))

    def parametros(prefixo, nome, saidas, entradas):
        for i in range(saidas):
            vetor(f"{prefixo}/{nome}/peso/{i}", entradas)
        vetor(f"{prefixo}/{nome}/vies", saidas)

    for etapa in ("inicial", "final"):
        for nome, (saidas, entradas) in camadas.items():
            parametros(etapa, nome, saidas, entradas)
    for epoca in (1, 2, 51):
        for amostra in (0, 1):
            prefixo = f"treino/{epoca}/{amostra}"
            keys.add(f"{prefixo}/perda")
            vetor(f"{prefixo}/grad_perda", 10)
            for nome, (saidas, entradas) in camadas.items():
                frente = f"{prefixo}/{nome}/forward"
                vetor(f"{frente}/entrada", entradas)
                vetor(f"{frente}/z", saidas)
                vetor(f"{frente}/saida", saidas)
                tras = f"{prefixo}/{nome}/backward"
                vetor(f"{tras}/grad_saida", saidas)
                vetor(f"{tras}/delta", saidas)
                vetor(f"{tras}/grad_entrada", entradas)
                vetor(f"{tras}/grad_vies", saidas)
                for i in range(saidas):
                    vetor(f"{tras}/grad_peso/{i}", entradas)
                for etapa in ("antes", "depois"):
                    parametros(f"{tras}/{etapa}", nome, saidas, entradas)
    keys.update(f"epoca/{i}/perda" for i in range(1, 52))
    for i in range(n_teste):
        prefixo = f"teste/0/{i}"
        vetor(f"{prefixo}/escore", 10)
        keys.update(f"{prefixo}/{k}" for k in ("perda", "rotulo", "previsto", "margem"))
    keys.update(f"confusao/{i}/{j}" for i in range(10) for j in range(10))
    keys.add("teste/acertos")
    return keys


def ler(caminho, esperadas):
    valores = {}
    with Path(caminho).open(encoding="utf-8") as f:
        if f.readline().rstrip("\n") != "BARENN_ORACLE\t1":
            raise ValueError(f"{caminho}: cabeçalho inválido")
        for linha, texto in enumerate(f, 2):
            campos = texto.rstrip("\n").split("\t")
            if len(campos) != 2 or not campos[0] or campos[0] in valores:
                raise ValueError(f"{caminho}:{linha}: registro inválido ou duplicado")
            valor = float(campos[1])
            if not math.isfinite(valor):
                raise ValueError(f"{caminho}:{linha}: valor não finito")
            valores[campos[0]] = valor
    faltam, sobram = esperadas - valores.keys(), valores.keys() - esperadas
    if faltam or sobram:
        raise ValueError(f"{caminho}: {len(faltam)} chaves ausentes, {len(sobram)} inesperadas; exemplo: {next(iter(faltam or sobram))}")
    return valores


def comparar(ref, atual, atol, rtol):
    erros = []
    max_abs = (0.0, "")
    max_rel = (0.0, "")
    for chave, esperado in ref.items():
        observado = atual[chave]
        absoluto = abs(observado - esperado)
        relativo = absoluto / abs(esperado) if esperado else (math.inf if absoluto else 0.0)
        if absoluto > max_abs[0]:
            max_abs = (absoluto, chave)
        if relativo > max_rel[0]:
            max_rel = (relativo, chave)
        inteiro = chave.startswith("confusao/") or chave == "teste/acertos" or chave.endswith(("/previsto", "/rotulo"))
        limite = 0.0 if inteiro else atol + rtol * abs(esperado)
        if absoluto > limite:
            erros.append((chave, esperado, observado, absoluto, limite))
    return erros, max_abs, max_rel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("referencia")
    parser.add_argument("candidato")
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--amostras-teste", type=int, default=1797)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)) or args.amostras_teste < 1:
        parser.error("tolerâncias devem ser finitas e não negativas; amostras deve ser positivo")
    try:
        esperadas = chaves_esperadas(args.amostras_teste)
        ref = ler(args.referencia, esperadas)
        candidato = ler(args.candidato, esperadas)
        erros, absoluto, relativo = comparar(ref, candidato, args.atol, args.rtol)
    except (ValueError, OSError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2
    print(f"Valores comparados: {len(ref)}; divergências: {len(erros)}")
    print(f"Maior erro absoluto: {absoluto[0]:.9g} ({absoluto[1] or 'nenhum'})")
    print(f"Maior erro relativo: {relativo[0]:.9g} ({relativo[1] or 'nenhum'})")
    mudancas_classe = sum(chave.endswith("/previsto") for chave, *_ in erros)
    print(f"Previsões diferentes: {mudancas_classe}")
    for chave, esperado, observado, erro, limite in erros[:10]:
        print(f"{chave}: referência={esperado:.9g} candidato={observado:.9g} erro={erro:.9g} limite={limite:.9g}")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
