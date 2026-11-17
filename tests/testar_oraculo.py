#!/usr/bin/env python3
"""Testa a captura e o comparador contra o commit do baseline."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "27f88c44a9d2cdcf3494af4d0ce87f8f285f08da"


def rodar(cmd, **kwargs):
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True, **kwargs)


def saida_sem_tempo(s):
    return "\n".join(x for x in s.splitlines() if not x.startswith("Tempo de treino "))


def hash_dados():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("optdigits.tra", "optdigits.tes")}


def main():
    hashes_iniciais = hash_dados()
    with tempfile.TemporaryDirectory(prefix="barenn-oraculo-") as td:
        temp = Path(td)
        fonte = temp / "base.cpp"
        fonte.write_text(rodar(["git", "show", f"{BASE}:barenn.cpp"]).stdout)
        flags = ["g++", "-std=c++17", "-O3", "-g"]
        for origem, nome, extra in ((fonte, "base", []), (ROOT / "barenn.cpp", "normal", []),
                                    (ROOT / "barenn.cpp", "oraculo", ["-DBARENN_ORACLE"]),
                                    (ROOT / "tests/oraculo.cpp", "unitario", [])):
            rodar(flags + extra + [str(origem), "-o", str(temp / nome)])
        print(rodar([str(temp / "unitario")]).stdout.strip())
        pre_base = rodar(["g++", "-std=c++17", "-E", "-P", str(fonte)]).stdout
        pre_novo = rodar(["g++", "-std=c++17", "-E", "-P", "barenn.cpp"]).stdout
        assert pre_base == pre_novo, "build normal alterado pelo pré-processamento"
        print("Pré-processamento sem macro idêntico ao legado.")
        env = os.environ.copy()
        env.pop("BARENN_ORACLE_OUTPUT", None)
        saidas = []
        for nome in ("base", "normal", "oraculo"):
            for _ in range(5):
                proc = rodar([str(temp / nome)], env=env)
                assert not proc.stderr
                saidas.append(saida_sem_tempo(proc.stdout))
        assert all(x == saidas[0] for x in saidas)
        print("Cinco execuções de cada build sem captura: saídas iguais.")
        rastros = []
        for i in range(2):
            rastro = temp / f"rastro{i}.tsv"
            proc = rodar([str(temp / "oraculo")], env={**env, "BARENN_ORACLE_OUTPUT": str(rastro)})
            assert not proc.stderr and saida_sem_tempo(proc.stdout) == saidas[0]
            rastros.append(rastro)
        assert rastros[0].read_bytes() == rastros[1].read_bytes()
        print("Duas capturas completas idênticas e saída igual ao legado.")
        comp = [sys.executable, "scripts/validar_oraculo.py", str(rastros[0]), str(rastros[1]), "--atol", "0", "--rtol", "0"]
        print(rodar(comp).stdout.strip())
        original = rastros[0].read_text()
        dados = dict(line.split("\t") for line in original.splitlines()[1:])
        for chave, texto in dados.items():
            # Perdas por época são double; demais valores numéricos de rede são float.
            if ("/peso/" in chave or "/vies/" in chave or "/grad_" in chave
                    or "/forward/" in chave or "/delta/" in chave or "/escore/" in chave):
                numero = float(texto)
                bits = struct.pack("f", numero)
                reconstruido = struct.unpack("f", bits)[0]
                assert struct.pack("f", float(format(reconstruido, ".9g"))) == bits, chave
        def f32(numero):
            return struct.unpack("f", struct.pack("f", numero))[0]

        for epoca in (1, 2, 51):
            for camada, (saidas, entradas) in {"l1": (32, 64), "l2": (16, 32), "l3": (10, 16)}.items():
                for amostra in (0, 1):
                    base = f"treino/{epoca}/{amostra}/{camada}/backward"
                    for i in range(saidas):
                        delta = f32(float(dados[f"{base}/delta/{i}"]))
                        antes = f32(float(dados[f"{base}/antes/{camada}/vies/{i}"]))
                        depois = float(dados[f"{base}/depois/{camada}/vies/{i}"])
                        esperado = f32(antes - f32(f32(0.02) * delta))
                        assert f32(depois) == esperado, (base, "vies", i)
                        for j in range(entradas):
                            antes = f32(float(dados[f"{base}/antes/{camada}/peso/{i}/{j}"]))
                            depois = float(dados[f"{base}/depois/{camada}/peso/{i}/{j}"])
                            grad = f32(float(dados[f"{base}/grad_peso/{i}/{j}"]))
                            esperado = f32(antes - f32(f32(0.02) * grad))
                            assert f32(depois) == esperado, (base, i, j)
                for chave, valor in dados.items():
                    prefixo = f"treino/{epoca}/0/{camada}/backward/depois/{camada}/"
                    if chave.startswith(prefixo):
                        seguinte = f"treino/{epoca}/1/{camada}/backward/antes/{camada}/" + chave[len(prefixo):]
                        assert valor == dados[seguinte], seguinte
        print("Round-trip float, atualizações e continuidade entre duas amostras verificados.")
        for nome, conteudo in (
            ("truncado", original.rsplit("\n", 2)[0] + "\n"),
            ("duplicado", original + original.splitlines()[1] + "\n"),
            ("nan", original.replace(next(x for x in original.splitlines() if x.startswith("epoca/1/perda\t")), "epoca/1/perda\tnan", 1)),
            ("previsto", original.replace("teste/0/0/previsto\t0\n", "teste/0/0/previsto\t9\n", 1)),
        ):
            assert conteudo != original, f"mutação não aplicada: {nome}"
            arquivo = temp / f"{nome}.tsv"
            arquivo.write_text(conteudo)
            r = subprocess.run([sys.executable, "scripts/validar_oraculo.py", str(rastros[0]), str(arquivo), "--atol", "1", "--rtol", "1"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == (1 if nome == "previsto" else 2), (nome, r.stdout, r.stderr)
        print("Comparador rejeitou captura incompleta, duplicada, inválida e previsão alterada.")
        existente = subprocess.run([str(temp / "oraculo")], cwd=ROOT, env={**env, "BARENN_ORACLE_OUTPUT": str(rastros[0])}, capture_output=True, text=True)
        assert existente.returncode != 0 and "já existe" in existente.stderr
        assert hash_dados() == hashes_iniciais
        print("Arquivo existente protegido; hashes dos datasets preservados.")


if __name__ == "__main__":
    main()
