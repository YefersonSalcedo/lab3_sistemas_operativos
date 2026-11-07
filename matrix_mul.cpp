// =====================================================================
// Práctica #3 - Multiplicación de matrices con procesos (C++)
// Sistemas Operativos - Universidad de Antioquia
//
// Integrantes:
//   - Yeferson Alexis Salcedo Preciado - ID
//   - Jhoan Esteban Echeverri Villa - ID
//
// IPC elegido: MEMORIA COMPARTIDA (shmget/shmat). A, B y C viven en
// segmentos compartidos; cada hijo escribe sus filas de C directamente,
// así no hay copias ni mensajes que serializar.
//
// Compilar:  g++ -O2 -o matrix_mul matrix_mul.cpp
// Usos:
//   ./matrix_mul gen N M P          -> genera A.txt (NxM) y B.txt (MxP)
//   ./matrix_mul A.txt B.txt K      -> secuencial vs K procesos, escribe C.txt
//   ./matrix_mul bench A.txt B.txt  -> prueba K=1,2,4,8,16 y guarda results_cpp.csv
// Formato de archivo: primera línea "filas columnas", luego la matriz.
// =====================================================================
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std;
typedef long long ll;

// Tiempo de pared (no CPU): clock() sumaría solo el padre y daría resultados falsos.
static double now_sec() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Reserva un segmento de memoria compartida de 'count' enteros de 64 bits.
static ll *shared_alloc(size_t count) {
    size_t bytes = max(count, (size_t)1) * sizeof(ll);
    int id = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0600);
    if (id < 0) { perror("shmget"); exit(1); }
    void *p = shmat(id, nullptr, 0);
    if (p == (void *)-1) { perror("shmat"); exit(1); }
    // Se marca para borrado: el kernel lo libera cuando todos los procesos se desacoplan.
    shmctl(id, IPC_RMID, nullptr);
    return (ll *)p;
}

static bool read_matrix(const string &path, int &r, int &c, vector<ll> &v) {
    ifstream in(path);
    if (!in) return false;
    if (!(in >> r >> c) || r <= 0 || c <= 0) return false;
    v.resize((size_t)r * c);
    for (auto &x : v)
        if (!(in >> x)) return false;
    return true;
}

static void write_matrix(const string &path, const ll *X, int r, int c) {
    FILE *f = fopen(path.c_str(), "w");
    if (!f) { perror("fopen"); exit(1); }
    fprintf(f, "%d %d\n", r, c);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++)
            fprintf(f, j ? " %lld" : "%lld", X[(size_t)i * c + j]);
        fputc('\n', f);
    }
    fclose(f);
}

// Calcula las filas [start, end) de C = A(NxM) * B(MxP).
static void multiply_rows(const ll *A, const ll *B, ll *C, int start, int end, int M, int P) {
    for (int i = start; i < end; i++) {
        ll *c = C + (size_t)i * P;
        for (int j = 0; j < P; j++) c[j] = 0;
        for (int k = 0; k < M; k++) {
            ll a = A[(size_t)i * M + k];
            const ll *b = B + (size_t)k * P;
            for (int j = 0; j < P; j++) c[j] += a * b[j];
        }
    }
}

// Versión secuencial (línea base). Devuelve segundos.
static double multiply_sequential(const ll *A, const ll *B, ll *C, int N, int M, int P) {
    double t0 = now_sec();
    multiply_rows(A, B, C, 0, N, M, P);
    return now_sec() - t0;
}

// Versión paralela: K hijos con fork(), cada uno con un bloque de filas.
// Si un hijo muere (señal o código != 0), el padre recalcula sus filas.
static double multiply_parallel(const ll *A, const ll *B, ll *C, int N, int M, int P, int K) {
    if (K > N) K = N;
    if (K < 1) K = 1;
    vector<pid_t> pids(K);
    vector<int> st(K), en(K);
    int base = N / K, extra = N % K, row = 0;

    double t0 = now_sec();
    for (int p = 0; p < K; p++) {
        st[p] = row;
        row += base + (p < extra ? 1 : 0);  // reparte el residuo: sirve para cualquier N
        en[p] = row;
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            pids[p] = -1;
        } else if (pid == 0) {
            const char *cr = getenv("CRASH_CHILD");  // para probar el manejo de errores
            if (cr && atoi(cr) == p) abort();
            multiply_rows(A, B, C, st[p], en[p], M, P);
            _exit(0);
        } else {
            pids[p] = pid;
        }
    }
    for (int p = 0; p < K; p++) {
        if (pids[p] < 0) {
            multiply_rows(A, B, C, st[p], en[p], M, P);
            continue;
        }
        int status = 0;
        waitpid(pids[p], &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            fprintf(stderr, "[aviso] el proceso hijo %d falló; el padre recalcula las filas %d-%d\n",
                    p, st[p], en[p] - 1);
            multiply_rows(A, B, C, st[p], en[p], M, P);
        }
    }
    return now_sec() - t0;
}

static bool load_shared(const string &fa, const string &fb, int &N, int &M, int &P,
                        ll *&A, ll *&B, ll *&C) {
    int r2, c2;
    vector<ll> va, vb;
    if (!read_matrix(fa, N, M, va)) { cerr << "No se pudo leer " << fa << "\n"; return false; }
    if (!read_matrix(fb, r2, P, vb)) { cerr << "No se pudo leer " << fb << "\n"; return false; }
    if (r2 != M) { cerr << "Dimensiones incompatibles: A es " << N << "x" << M
                        << " y B es " << r2 << "x" << P << "\n"; return false; }
    (void)c2;
    A = shared_alloc(va.size());
    B = shared_alloc(vb.size());
    C = shared_alloc((size_t)N * P);
    memcpy(A, va.data(), va.size() * sizeof(ll));
    memcpy(B, vb.data(), vb.size() * sizeof(ll));
    return true;
}

static void cmd_gen(int N, int M, int P) {
    srand(12345);
    vector<ll> a((size_t)N * M), b((size_t)M * P);
    for (auto &x : a) x = rand() % 10;
    for (auto &x : b) x = rand() % 10;
    write_matrix("A.txt", a.data(), N, M);
    write_matrix("B.txt", b.data(), M, P);
    printf("Generadas A.txt (%dx%d) y B.txt (%dx%d)\n", N, M, M, P);
}

static void cmd_bench(const string &fa, const string &fb) {
    int N, M, P;
    ll *A, *B, *C;
    if (!load_shared(fa, fb, N, M, P, A, B, C)) exit(1);
    vector<ll> Cseq((size_t)N * P);
    const int REPS = 3;
    double tseq = 1e18;
    for (int r = 0; r < REPS; r++) tseq = min(tseq, multiply_sequential(A, B, Cseq.data(), N, M, P));
    printf("Secuencial: %.4f s\n\n%-10s %-12s %-10s\n", tseq, "Procesos", "Tiempo (s)", "Speedup");
    FILE *csv = fopen("results_cpp.csv", "w");
    fprintf(csv, "procesos,tiempo_s,speedup\n1,%.6f,1.0\n", tseq);
    int ks[] = {1, 2, 4, 8, 16};
    for (int K : ks) {
        double t = 1e18;
        for (int r = 0; r < REPS; r++) t = min(t, multiply_parallel(A, B, C, N, M, P, K));
        bool ok = memcmp(C, Cseq.data(), Cseq.size() * sizeof(ll)) == 0;
        printf("%-10d %-12.4f %-10.2f %s\n", K, t, tseq / t, ok ? "" : "(¡RESULTADO DISTINTO!)");
        if (K > 1) fprintf(csv, "%d,%.6f,%.4f\n", K, t, tseq / t);
    }
    fclose(csv);
    printf("\nGuardado results_cpp.csv\n");
}

int main(int argc, char **argv) {
    if (argc == 5 && string(argv[1]) == "gen") {
        cmd_gen(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
        return 0;
    }
    if (argc == 4 && string(argv[1]) == "bench") {
        cmd_bench(argv[2], argv[3]);
        return 0;
    }
    if (argc != 4) {
        cerr << "Uso:\n  " << argv[0] << " gen N M P\n  " << argv[0] << " A.txt B.txt K\n  "
             << argv[0] << " bench A.txt B.txt\n";
        return 1;
    }
    int K = atoi(argv[3]);
    if (K < 1) { cerr << "K debe ser >= 1\n"; return 1; }

    int N, M, P;
    ll *A, *B, *C;
    if (!load_shared(argv[1], argv[2], N, M, P, A, B, C)) return 1;

    vector<ll> Cseq((size_t)N * P);
    double tseq = multiply_sequential(A, B, Cseq.data(), N, M, P);
    double tpar = multiply_parallel(A, B, C, N, M, P, K);

    bool ok = memcmp(C, Cseq.data(), Cseq.size() * sizeof(ll)) == 0;
    write_matrix("C.txt", C, N, P);

    printf("Sequential time: %.4f seconds\n", tseq);
    printf("Parallel time (%d processes): %.4f seconds\n", K, tpar);
    printf("Speedup: %.2fx\n", tseq / tpar);
    printf("Verificación paralelo == secuencial: %s\n", ok ? "OK" : "FALLÓ");
    return ok ? 0 : 2;
}