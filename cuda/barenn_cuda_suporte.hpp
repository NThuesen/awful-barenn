// Componentes de leitura e diagnóstico equivalentes aos da versão CPU.
// Se a semântica da CPU mudar, atualizar e comparar com o oráculo.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef BARENN_ORACLE
#include <cstdlib>
#include <limits>
#include <locale>
#endif

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

#ifdef BARENN_ORACLE
// A captura só existe no binário de diagnóstico e é ligada por variável de ambiente.
namespace Oracle {
ofstream arquivo;
int epoca = 0;
size_t amostra = 0;
bool treino = false;
bool selecionada = false;

void iniciar() {
    const char* caminho = getenv("BARENN_ORACLE_OUTPUT");
    if (!caminho) return;
    if (!*caminho) throw runtime_error("BARENN_ORACLE_OUTPUT vazio");
    if (filesystem::exists(caminho))
        throw runtime_error("Arquivo do oráculo já existe");
    arquivo.open(caminho);
    if (!arquivo) throw runtime_error("Não foi possível criar arquivo do oráculo");
    arquivo.imbue(locale::classic());
    arquivo << "BARENN_ORACLE\t1\n";
}

bool ativa() { return arquivo.is_open(); }
string prefixo() {
    return string(treino ? "treino/" : "teste/") + to_string(epoca) + "/" + to_string(amostra);
}
void valor(const string& chave, double valor, int precisao = numeric_limits<double>::max_digits10) {
    if (!ativa()) return;
    if (!isfinite(valor)) throw runtime_error("Valor não finito no oráculo: " + chave);
    arquivo << chave << '\t' << setprecision(precisao) << valor << '\n';
    if (!arquivo) throw runtime_error("Falha ao escrever arquivo do oráculo");
}
void vetor(const string& chave, const vector<float>& valores) {
    if (!ativa()) return;
    for (size_t i = 0; i < valores.size(); ++i)
        valor(chave + "/" + to_string(i), valores[i], numeric_limits<float>::max_digits10);
}
void terminar() {
    if (!ativa()) return;
    arquivo.flush();
    if (!arquivo) throw runtime_error("Falha ao gravar arquivo do oráculo");
    arquivo.close();
    if (arquivo.fail()) throw runtime_error("Falha ao fechar arquivo do oráculo");
}
} // namespace Oracle
#endif

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
#ifdef BARENN_ORACLE
    if (Oracle::ativa() && (!Oracle::treino || Oracle::selecionada)) {
        Oracle::valor(Oracle::prefixo() + "/perda", total, numeric_limits<float>::max_digits10);
        if (gradiente) Oracle::vetor(Oracle::prefixo() + "/grad_perda", *gradiente);
    }
#endif
    return total;
}


} // namespace Barenn
