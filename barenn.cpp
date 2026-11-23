#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
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

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // weights[i][j] liga a entrada j ao neurônio de saída i.
    vector<vector<float>> weights;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes.
    vector<float> last_input;
    vector<float> last_z;

#ifdef BARENN_ORACLE
    string nome_oraculo;
public:
    void oracle_nome(const string& nome) { nome_oraculo = nome; }
    void oracle_parametros(const string& contexto) const {
        if (!Oracle::ativa()) return;
        for (size_t i = 0; i < weights.size(); ++i)
            Oracle::vetor(contexto + "/" + nome_oraculo + "/peso/" + to_string(i), weights[i]);
        Oracle::vetor(contexto + "/" + nome_oraculo + "/vies", bias);
    }
    // Usado apenas pelos testes de camadas pequenas.
    void oracle_definir(const vector<vector<float>>& w, const vector<float>& b) {
        if (w.size() != weights.size() || b.size() != bias.size())
            throw runtime_error("Dimensões inválidas");
        for (const auto& linha : w)
            if (linha.size() != static_cast<size_t>(in_features))
                throw runtime_error("Dimensões inválidas");
        weights = w;
        bias = b;
    }
#endif
public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(output_size, vector<float>(input_size)), bias(output_size, 0) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (auto& row : weights) {
            for (float& weight : row) {
                weight = distribution(gen);
            }
        }
    }

    vector<float> Forward(vector<float> input) {
        last_input = input;
        last_z.resize(out_features);
        vector<float> out(out_features);
        // Cada neurônio soma o viés aos produtos entre entradas e pesos.
        for (int i = 0; i < out_features; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * weights[i][j];
            }
            last_z[i] = sum;
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            out[i] = final ? sum : max(0.0f, sum);
        }
#ifdef BARENN_ORACLE
        if (Oracle::ativa() && Oracle::treino && Oracle::selecionada) {
            const string chave = Oracle::prefixo() + "/" + nome_oraculo + "/forward";
            Oracle::vetor(chave + "/entrada", input);
            Oracle::vetor(chave + "/z", last_z);
            Oracle::vetor(chave + "/saida", out);
        }
#endif
        return out;
    }

#ifdef BARENN_PARALLEL_EVAL
    vector<float> Inferir(const vector<float>& input) const {
        vector<float> out(out_features);
        for (int i = 0; i < out_features; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) sum += input[j] * weights[i][j];
            out[i] = final ? sum : max(0.0f, sum);
        }
        return out;
    }
#endif

    vector<float> Backward(const vector<float>& grad_out, float learning_rate) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        vector<float> delta(out_features);
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // O gradiente da entrada usa os pesos anteriores à atualização.
        vector<float> grad_in(in_features);
        for (int j = 0; j < out_features; ++j) {
            for (int i = 0; i < in_features; ++i) {
                grad_in[i] += delta[j] * weights[j][i];
            }
        }

#ifdef BARENN_ORACLE
        if (Oracle::ativa() && Oracle::selecionada) {
            const string chave = Oracle::prefixo() + "/" + nome_oraculo + "/backward";
            Oracle::vetor(chave + "/grad_saida", grad_out);
            Oracle::vetor(chave + "/delta", delta);
            Oracle::vetor(chave + "/grad_entrada", grad_in);
            oracle_parametros(chave + "/antes");
            for (size_t i = 0; i < delta.size(); ++i) {
                Oracle::valor(chave + "/grad_vies/" + to_string(i), delta[i], numeric_limits<float>::max_digits10);
                for (size_t j = 0; j < last_input.size(); ++j)
                    Oracle::valor(chave + "/grad_peso/" + to_string(i) + "/" + to_string(j),
                                  delta[i] * last_input[j], numeric_limits<float>::max_digits10);
            }
        }
#endif
        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta.
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights[i][j] -= learning_rate * (delta[i] * last_input[j]);
            }
            bias[i] -= learning_rate * delta[i];
        }
#ifdef BARENN_ORACLE
        if (Oracle::ativa() && Oracle::selecionada)
            oracle_parametros(Oracle::prefixo() + "/" + nome_oraculo + "/backward/depois");
#endif
        return grad_in;
    }
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<string> linhas;
    string linha;
    while (getline(arquivo, linha)) {
        linhas.push_back(move(linha));
    }

    vector<Amostra> dados(linhas.size());
    vector<exception_ptr> erros(linhas.size());
    #pragma omp parallel for schedule(static)
    for (size_t indice = 0; indice < linhas.size(); ++indice) {
        try {
            istringstream campos(linhas[indice]);
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
                throw runtime_error(caminho + ":" + to_string(indice + 1) +
                    " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
            }
            dados[indice] = move(amostra);
        } catch (...) {
            erros[indice] = current_exception();
        }
    }
    // Reporta a primeira falha na ordem original, após a barreira implícita.
    for (const auto& erro : erros) {
        if (erro) rethrow_exception(erro);
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

void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    const auto inicio_avaliacao = chrono::steady_clock::now();
#ifdef BARENN_PARALLEL_EVAL
    vector<vector<float>> resultados(teste.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < teste.size(); ++i)
        resultados[i] = l3.Inferir(l2.Inferir(l1.Inferir(teste[i].imagem)));
#endif
    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
#ifdef BARENN_ORACLE
        Oracle::treino = false;
        Oracle::epoca = 0;
        Oracle::amostra = i;
#endif
#ifdef BARENN_PARALLEL_EVAL
        auto& escores = resultados[i];
#else
        auto escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
#endif
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);
#ifdef BARENN_ORACLE
        if (Oracle::ativa()) {
            const string chave = Oracle::prefixo();
            Oracle::vetor(chave + "/escore", escores);
            Oracle::valor(chave + "/rotulo", amostra.rotulo);
            Oracle::valor(chave + "/previsto", previsto);
            float segundo = -numeric_limits<float>::infinity();
            for (int j = 0; j < classes; ++j)
                if (j != previsto) segundo = max(segundo, escores[j]);
            Oracle::valor(chave + "/margem", escores[previsto] - segundo, numeric_limits<float>::max_digits10);
        }
#endif

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    const chrono::duration<double> tempo_avaliacao = chrono::steady_clock::now() - inicio_avaliacao;
    cerr << "Tempo avaliacao " << tempo_avaliacao.count() << " s\n";
#ifdef BARENN_ORACLE
    if (Oracle::ativa()) {
        for (int i = 0; i < classes; ++i)
            for (int j = 0; j < classes; ++j)
                Oracle::valor("confusao/" + to_string(i) + "/" + to_string(j), confusoes[i][j]);
        Oracle::valor("teste/acertos", acertos);
    }
#endif
    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
#ifdef BARENN_ORACLE
        Barenn::Oracle::iniciar();
#endif
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
#ifdef BARENN_ORACLE
        l1.oracle_nome("l1"); l2.oracle_nome("l2"); l3.oracle_nome("l3");
        l1.oracle_parametros("inicial"); l2.oracle_parametros("inicial"); l3.oracle_parametros("inicial");
#endif
        vector<float> gradiente(Barenn::classes);
        array<double, epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
#ifdef BARENN_ORACLE
            size_t indice = 0;
#endif
            for (const auto& amostra : treino) {
#ifdef BARENN_ORACLE
                Barenn::Oracle::treino = true;
                Barenn::Oracle::epoca = epoca + 1;
                Barenn::Oracle::amostra = indice++;
                Barenn::Oracle::selecionada = (epoca == 0 || epoca == 1 || epoca == 50) && Barenn::Oracle::amostra < 2;
#endif
                auto escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
                perda_total += Barenn::perda(escores, amostra.rotulo, &gradiente);

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                auto grad3 = l3.Backward(gradiente, taxa);
                auto grad2 = l2.Backward(grad3, taxa);
                l1.Backward(grad2, taxa);
            }
            perdas[epoca] = perda_total / treino.size();
#ifdef BARENN_ORACLE
            Barenn::Oracle::valor("epoca/" + to_string(epoca + 1) + "/perda", perdas[epoca]);
#endif
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

#ifdef BARENN_ORACLE
        l1.oracle_parametros("final"); l2.oracle_parametros("final"); l3.oracle_parametros("final");
#endif
        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
#ifdef BARENN_ORACLE
        Barenn::Oracle::terminar();
#endif
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
