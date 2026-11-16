#define BARENN_ORACLE
#define main barenn_programa_main
#include "../barenn.cpp"
#undef main
#include <cassert>
using Barenn::DenseLayer;
void perto(float valor, float esperado, float erro = 1e-6f) {
    if (std::abs(valor - esperado) > erro) {
        std::cerr << "Esperado " << esperado << ", recebido " << valor << '\n';
        std::exit(1);
    }
}
int main() {
    DenseLayer l(2, 1, true);
    l.oracle_definir({{0.5f, -0.25f}}, {0.1f});
    perto(l.Forward({2.0f, 4.0f})[0], 0.1f);
    auto entrada = l.Backward({0.4f}, 0.02f);
    perto(entrada[0], 0.2f); perto(entrada[1], -0.1f);
    perto(l.Forward({1.0f, 2.0f})[0], 0.012f);
    auto segunda = l.Backward({0.5f}, 0.02f);
    perto(segunda[0], 0.242f); perto(segunda[1], -0.141f);
    DenseLayer relu(1, 1);
    relu.oracle_definir({{1.0f}}, {0.0f});
    for (float x : {-1.0f, 0.0f, 1.0f}) {
        perto(relu.Forward({x})[0], std::max(0.0f, x));
        perto(relu.Backward({1.0f}, 0.0f)[0], x > 0 ? 1.0f : 0.0f);
    }
    const float h = 0.001f;
    auto perda_local = [](float peso) {
        DenseLayer camada(1, 1, true);
        camada.oracle_definir({{peso}}, {0.2f});
        const float erro = camada.Forward({2.0f})[0] - 0.7f;
        return 0.5f * erro * erro;
    };
    float grad_numerico = (perda_local(0.4f + h) - perda_local(0.4f - h)) / (2 * h);
    perto(grad_numerico, 0.6f, 1e-4f);
    DenseLayer grad(1, 1, true);
    grad.oracle_definir({{0.4f}}, {0.2f});
    const float erro = grad.Forward({2.0f})[0] - 0.7f;
    perto(grad.Backward({erro}, 0.1f)[0], 0.12f);
    perto(grad.Forward({2.0f})[0], 0.85f);
    std::cout << "Testes de camada, ReLU, gradiente e SGD passaram.\n";
}
