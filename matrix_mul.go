// =====================================================================
// Práctica #3 - Multiplicación de matrices con procesos (Go)
// Sistemas Operativos - Universidad de Antioquia
//
// Integrantes:
//   - Yeferson Alexis Salcedo Preciado - ID
//   - Jhoan Esteban Echeverri Villa - ID
//
// IPC elegido: MEMORIA COMPARTIDA. Go no tiene fork(), así que el padre
// lanza K procesos hijos re-ejecutando este mismo binario (os/exec) en
// modo "--worker". A, B y C viven en un archivo en /dev/shm proyectado
// con mmap(MAP_SHARED): todos los procesos ven la misma memoria física
// y cada hijo escribe directamente sus filas de C.
//
// Compilar:  go build -o matrix_mul_go matrix_mul.go
// Usos:
//   ./matrix_mul_go gen N M P          -> genera A.txt (NxM) y B.txt (MxP)
//   ./matrix_mul_go A.txt B.txt K      -> secuencial vs K procesos, escribe C.txt
//   ./matrix_mul_go bench A.txt B.txt  -> prueba K=1,2,4,8,16 y guarda results_go.csv
// Formato de archivo: primera línea "filas columnas", luego la matriz.
// =====================================================================
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"syscall"
	"time"
	"unsafe"
)

// ---------- E/S de matrices ----------

func readMatrix(path string) (rows, cols int, data []int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	if _, err = fmt.Fscan(r, &rows, &cols); err != nil || rows <= 0 || cols <= 0 {
		return 0, 0, nil, fmt.Errorf("encabezado inválido en %s", path)
	}
	data = make([]int64, rows*cols)
	for i := range data {
		if _, err = fmt.Fscan(r, &data[i]); err != nil {
			return 0, 0, nil, fmt.Errorf("faltan datos en %s: %v", path, err)
		}
	}
	return rows, cols, data, nil
}

func writeMatrix(path string, x []int64, rows, cols int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	fmt.Fprintf(w, "%d %d\n", rows, cols)
	buf := make([]byte, 0, 24)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if j > 0 {
				w.WriteByte(' ')
			}
			buf = strconv.AppendInt(buf[:0], x[i*cols+j], 10)
			w.Write(buf)
		}
		w.WriteByte('\n')
	}
	return w.Flush()
}

// ---------- Cálculo ----------

// Calcula las filas [start, end) de C = A(NxM) * B(MxP).
func multiplyRows(A, B, C []int64, start, end, M, P int) {
	for i := start; i < end; i++ {
		c := C[i*P : (i+1)*P]
		for j := range c {
			c[j] = 0
		}
		for k := 0; k < M; k++ {
			a := A[i*M+k]
			b := B[k*P : (k+1)*P]
			for j := range c {
				c[j] += a * b[j]
			}
		}
	}
}

// ---------- Memoria compartida (mmap sobre /dev/shm) ----------

type shared struct {
	path    string
	mem     []byte
	A, B, C []int64
	N, M, P int
}

func mapShared(path string, N, M, P int, create bool) (*shared, error) {
	total := (N*M + M*P + N*P) * 8
	var f *os.File
	var err error
	if create {
		dir := "/dev/shm"
		if st, e := os.Stat(dir); e != nil || !st.IsDir() {
			dir = os.TempDir()
		}
		f, err = os.CreateTemp(dir, "matmul-*.shm")
		if err != nil {
			return nil, err
		}
		path = f.Name()
		if err = f.Truncate(int64(total)); err != nil {
			return nil, err
		}
	} else {
		f, err = os.OpenFile(path, os.O_RDWR, 0)
		if err != nil {
			return nil, err
		}
	}
	defer f.Close()
	mem, err := syscall.Mmap(int(f.Fd()), 0, total, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	ints := unsafe.Slice((*int64)(unsafe.Pointer(&mem[0])), total/8)
	return &shared{
		path: path, mem: mem, N: N, M: M, P: P,
		A: ints[:N*M],
		B: ints[N*M : N*M+M*P],
		C: ints[N*M+M*P:],
	}, nil
}

func (s *shared) close(remove bool) {
	syscall.Munmap(s.mem)
	if remove {
		os.Remove(s.path)
	}
}

// ---------- Modo hijo ----------
// args: path N M P start end idx

func worker(args []string) int {
	if len(args) != 7 {
		return 1
	}
	n := make([]int, 6)
	for i := range n {
		v, err := strconv.Atoi(args[i+1])
		if err != nil {
			return 1
		}
		n[i] = v
	}
	N, M, P, start, end, idx := n[0], n[1], n[2], n[3], n[4], n[5]
	if c := os.Getenv("CRASH_CHILD"); c != "" { // para probar el manejo de errores
		if v, err := strconv.Atoi(c); err == nil && v == idx {
			os.Exit(3)
		}
	}
	s, err := mapShared(args[0], N, M, P, false)
	if err != nil {
		fmt.Fprintln(os.Stderr, "worker:", err)
		return 1
	}
	multiplyRows(s.A, s.B, s.C, start, end, M, P)
	s.close(false)
	return 0
}

// ---------- Secuencial y paralelo ----------

func multiplySequential(s *shared, C []int64) time.Duration {
	t0 := time.Now()
	multiplyRows(s.A, s.B, C, 0, s.N, s.M, s.P)
	return time.Since(t0)
}

func multiplyParallel(s *shared, K int) time.Duration {
	if K > s.N {
		K = s.N
	}
	if K < 1 {
		K = 1
	}
	self, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	cmds := make([]*exec.Cmd, K)
	st := make([]int, K)
	en := make([]int, K)
	base, extra, row := s.N/K, s.N%K, 0

	t0 := time.Now()
	for p := 0; p < K; p++ {
		st[p] = row
		row += base
		if p < extra { // reparte el residuo: sirve para cualquier N
			row++
		}
		en[p] = row
		cmd := exec.Command(self, "--worker", s.path,
			strconv.Itoa(s.N), strconv.Itoa(s.M), strconv.Itoa(s.P),
			strconv.Itoa(st[p]), strconv.Itoa(en[p]), strconv.Itoa(p))
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			fmt.Fprintf(os.Stderr, "[aviso] no se pudo lanzar el hijo %d: %v\n", p, err)
			cmd = nil
		}
		cmds[p] = cmd
	}
	for p := 0; p < K; p++ {
		if cmds[p] == nil {
			multiplyRows(s.A, s.B, s.C, st[p], en[p], s.M, s.P)
			continue
		}
		if err := cmds[p].Wait(); err != nil {
			fmt.Fprintf(os.Stderr, "[aviso] el proceso hijo %d falló (%v); el padre recalcula las filas %d-%d\n",
				p, err, st[p], en[p]-1)
			multiplyRows(s.A, s.B, s.C, st[p], en[p], s.M, s.P)
		}
	}
	return time.Since(t0)
}

// ---------- Comandos ----------

func loadShared(fa, fb string) (*shared, error) {
	N, M, a, err := readMatrix(fa)
	if err != nil {
		return nil, err
	}
	M2, P, b, err := readMatrix(fb)
	if err != nil {
		return nil, err
	}
	if M2 != M {
		return nil, fmt.Errorf("dimensiones incompatibles: A es %dx%d y B es %dx%d", N, M, M2, P)
	}
	s, err := mapShared("", N, M, P, true)
	if err != nil {
		return nil, err
	}
	copy(s.A, a)
	copy(s.B, b)
	return s, nil
}

func equal(a, b []int64) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func cmdGen(N, M, P int) {
	rng := rand.New(rand.NewSource(12345))
	a := make([]int64, N*M)
	b := make([]int64, M*P)
	for i := range a {
		a[i] = int64(rng.Intn(10))
	}
	for i := range b {
		b[i] = int64(rng.Intn(10))
	}
	writeMatrix("A.txt", a, N, M)
	writeMatrix("B.txt", b, M, P)
	fmt.Printf("Generadas A.txt (%dx%d) y B.txt (%dx%d)\n", N, M, M, P)
}

func cmdBench(fa, fb string) {
	s, err := loadShared(fa, fb)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer s.close(true)
	cseq := make([]int64, s.N*s.P)
	const reps = 3
	tseq := time.Duration(1 << 62)
	for r := 0; r < reps; r++ {
		if d := multiplySequential(s, cseq); d < tseq {
			tseq = d
		}
	}
	fmt.Printf("Secuencial: %.4f s\n\n%-10s %-12s %-10s\n", tseq.Seconds(), "Procesos", "Tiempo (s)", "Speedup")
	csv, _ := os.Create("results_go.csv")
	defer csv.Close()
	fmt.Fprintf(csv, "procesos,tiempo_s,speedup\n1,%.6f,1.0\n", tseq.Seconds())
	for _, K := range []int{1, 2, 4, 8, 16} {
		t := time.Duration(1 << 62)
		for r := 0; r < reps; r++ {
			if d := multiplyParallel(s, K); d < t {
				t = d
			}
		}
		note := ""
		if !equal(s.C, cseq) {
			note = "(¡RESULTADO DISTINTO!)"
		}
		sp := tseq.Seconds() / t.Seconds()
		fmt.Printf("%-10d %-12.4f %-10.2f %s\n", K, t.Seconds(), sp, note)
		if K > 1 {
			fmt.Fprintf(csv, "%d,%.6f,%.4f\n", K, t.Seconds(), sp)
		}
	}
	fmt.Println("\nGuardado results_go.csv")
}

func main() {
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "--worker" {
		os.Exit(worker(args[1:]))
	}
	usage := func() {
		fmt.Fprintf(os.Stderr, "Uso:\n  %s gen N M P\n  %s A.txt B.txt K\n  %s bench A.txt B.txt\n",
			os.Args[0], os.Args[0], os.Args[0])
		os.Exit(1)
	}
	switch {
	case len(args) == 4 && args[0] == "gen":
		N, _ := strconv.Atoi(args[1])
		M, _ := strconv.Atoi(args[2])
		P, _ := strconv.Atoi(args[3])
		if N < 1 || M < 1 || P < 1 {
			usage()
		}
		cmdGen(N, M, P)
	case len(args) == 3 && args[0] == "bench":
		cmdBench(args[1], args[2])
	case len(args) == 3:
		K, err := strconv.Atoi(args[2])
		if err != nil || K < 1 {
			fmt.Fprintln(os.Stderr, "K debe ser >= 1")
			os.Exit(1)
		}
		s, err := loadShared(args[0], args[1])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer s.close(true)
		cseq := make([]int64, s.N*s.P)
		tseq := multiplySequential(s, cseq)
		tpar := multiplyParallel(s, K)
		ok := equal(s.C, cseq)
		if err := writeMatrix("C.txt", s.C, s.N, s.P); err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
		fmt.Printf("Sequential time: %.4f seconds\n", tseq.Seconds())
		fmt.Printf("Parallel time (%d processes): %.4f seconds\n", K, tpar.Seconds())
		fmt.Printf("Speedup: %.2fx\n", tseq.Seconds()/tpar.Seconds())
		if ok {
			fmt.Println("Verificación paralelo == secuencial: OK")
		} else {
			fmt.Println("Verificación paralelo == secuencial: FALLÓ")
			s.close(true)
			os.Exit(2)
		}
	default:
		usage()
	}
}