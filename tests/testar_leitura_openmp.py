#!/usr/bin/env python3
"""Compara ordem, saída e erros da leitura OpenMP com a CPU de referência."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "70a7a53c716859151ef16a67442f3ae4a54fa9bf"


def sem_tempo(texto):
    return "\n".join(linha for linha in texto.splitlines()
                     if not linha.startswith("Tempo de treino "))


def main():
    normal = ",".join(["0"] * 64) + ",3"
    outra = ",".join(["1"] * 64) + ",7"
    casos = {
        "vazio": "",
        "atributos insuficientes": "0,1,2",
        "sem rótulo": ",".join(["0"] * 64) + ",",
        "negativo": ",".join(["-1"] + ["0"] * 63) + ",2",
        "acima do limite": ",".join(["17"] + ["0"] * 63) + ",2",
        "nan": ",".join(["nan"] + ["0"] * 63) + ",2",
        "infinito": ",".join(["inf"] + ["0"] * 63) + ",2",
        "campo extra": normal + ",1",
        "rótulo inválido": ",".join(["0"] * 64) + ",10",
        "múltiplos erros": normal + "\nruim\n" + outra + "\nruim2\n",
        "válido": normal + "\n" + outra + "\n",
    }
    with tempfile.TemporaryDirectory(prefix="barenn-leitura-") as tmp:
        pasta = Path(tmp)
        referencia = pasta / "referencia.cpp"
        referencia.write_text(subprocess.run(
            ["git", "show", f"{BASE}:barenn.cpp"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout)
        flags = ["g++", "-std=c++17", "-O3", "-ffp-contract=off"]
        subprocess.run(flags + [str(referencia), "-o", str(pasta / "ref")], check=True)
        subprocess.run(flags + ["-fopenmp", str(ROOT / "barenn.cpp"),
                                "-o", str(pasta / "omp")], check=True)
        teste = pasta / "teste"
        teste.write_text(outra + "\n")
        for nome, conteudo in casos.items():
            treino = pasta / "treino"
            treino.write_text(conteudo)
            ref = subprocess.run([str(pasta / "ref"), str(treino), str(teste)],
                                 capture_output=True, text=True)
            for threads in (1, 2, 4, 8):
                env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE")
                atual = subprocess.run([str(pasta / "omp"), str(treino), str(teste)],
                                       capture_output=True, text=True, env=env)
                assert (ref.returncode, sem_tempo(ref.stdout), ref.stderr) == (
                    atual.returncode, sem_tempo(atual.stdout), atual.stderr), (nome, threads)
    print(f"{len(casos)} casos e quatro configurações: saídas e erros iguais.")


if __name__ == "__main__":
    main()
