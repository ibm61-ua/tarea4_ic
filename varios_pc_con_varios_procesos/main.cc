#include "sample.h"
#include "knn.h"
#include "mnist.h"
#include "crossValidation.h"
#include "timer.h"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace knn;
using namespace mnist;
using namespace crossValidation;

// =========================================================================
// Etiquetas MPI para la comunicación entre ajuste (tuning) y test especulativo
// =========================================================================
enum MPITags {
    TAG_NEW_K = 1,        // Proceso de ajuste notifica a Rank 0 un nuevo mejor K encontrado
    TAG_FINISHED = 2,     // Proceso de ajuste notifica a Rank 0 la finalización del tuning con el K óptimo definitivo
    TAG_TUNER_UPDATE = 3  // Actualización asíncrona entre procesos de tuning del mejor accuracy conocido
};

// =========================================================================
// Configuración configurable del número de muestras para depuración rápida o prueba completa.
// - Puede definirse en tiempo de compilación con -DFAST_DEBUG (1000 muestras) o -DMAX_TRAIN_SAMPLES=N.
// - También se puede modificar por argumentos CLI: mpirun -np X ./knn_mpi [num_train] [num_test]
// =========================================================================
#ifndef MAX_TRAIN_SAMPLES
#if defined(FAST_DEBUG)
#define MAX_TRAIN_SAMPLES 1000
#else
#define MAX_TRAIN_SAMPLES 20000 // Tamaño por defecto para el fine tuning según el guion
#endif
#endif

#ifndef MAX_TEST_SAMPLES
#if defined(FAST_DEBUG)
#define MAX_TEST_SAMPLES 200
#else
#define MAX_TEST_SAMPLES 0 // 0 = usar todas las muestras disponibles del conjunto de test
#endif
#endif

const std::size_t DEFAULT_TRAIN_LIMIT = MAX_TRAIN_SAMPLES;
const std::size_t DEFAULT_TEST_LIMIT = MAX_TEST_SAMPLES;

namespace {
bool fileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}
}

/// <summary>
/// Loads the MNIST dataset.
/// </summary>
bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    std::string prefix = "";
    if (fileExists("train/train-images-idx3-ubyte")) {
        prefix = "";
    } else if (fileExists("build/train/train-images-idx3-ubyte")) {
        prefix = "build/";
    } else if (fileExists("../train/train-images-idx3-ubyte")) {
        prefix = "../";
    }

    // Cargar datos de entrenamiento
    std::string img_path = prefix + "train/train-images-idx3-ubyte";
    std::string label_path = prefix + "train/train-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), train))
    {
        std::cerr << "The training data could not be read from " << img_path << std::endl;
        return false;
    }
    
    // Cargar datos de evaluación
    img_path = prefix + "test/t10k-images-idx3-ubyte";
    label_path = prefix + "test/t10k-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), test))
    {
        std::cerr << "The test data could not be read from " << img_path << std::endl;
        return false;
    }
    
    return true;
}

/// <summary>
/// Fine tuning secuencial utilizado en el caso fallback de 1 solo proceso.
/// </summary>
int fineTuningSequential(unsigned int minK, unsigned int maxK, const std::vector<TSample> &dataset, unsigned int folds)
{
    assert(minK >= 1 && maxK > minK);
    CrossValidation crossValidation = CrossValidation::NFold(dataset, folds);
    std::cout << "[Secuencial] Búsqueda de mejor K en rango [" << minK << ", " << maxK << "]..." << std::endl;
    
    unsigned int bestK = minK;
    double bestAccuracy = 0.0;
    for(unsigned int k = minK; k <= maxK; k++)
    {
        double start = elapsedTime();
        double accuracy = crossValidation.validate(k);
        std::cout << "K: " << k << " Accuracy: " << accuracy * 100 << "% en " << elapsedTime() - start << "s" << std::endl;
        if(accuracy > bestAccuracy)
        {
            bestAccuracy = accuracy;
            bestK = k;
        }
    }
    return bestK;
}

/// <summary>
/// Función ejecutada por el proceso de evaluación especulativa (Rank 0).
/// Comienza inmediatamente a evaluar test con un K inicial y escucha con MPI_Iprobe
/// actualizaciones de K o la señal final de terminación del tuning.
/// </summary>
void runSpeculativeTest(const std::vector<TSample>& train, const std::vector<TSample>& test, int initialK)
{
    int currentK = initialK;
    KNN knn(train, currentK);
    std::vector<int> labels;
    int matches = 0;
    bool tuningFinished = false;
    int finalOptimalK = -1;

    std::cout << "\n>>> [Test Rank 0] Iniciando evaluación especulativa con K inicial = " << currentK
              << " (" << test.size() << " muestras de test) sin esperar al tuning...\n" << std::endl;

    double testStart = MPI_Wtime();
    std::size_t i = 0;

    while (i < test.size()) {
        // ---------------------------------------------------------------------
        // MPI_Iprobe: Comprobación periódica no bloqueante de mensajes entrantes.
        // Permite reaccionar a nuevos valores de K sin bloquear el cómputo de test.
        // ---------------------------------------------------------------------
        int flag = 0;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        bool needRestart = false;

        while (flag) {
            if (status.MPI_TAG == TAG_NEW_K) {
                int newK = 0;
                // MPI_Recv: Recibir el nuevo mejor K reportado por algún proceso de ajuste
                MPI_Recv(&newK, 1, MPI_INT, status.MPI_SOURCE, TAG_NEW_K, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (newK != currentK) {
                    std::cout << "\n[Test Rank 0] <<< Recibido NUEVO MEJOR K=" << newK
                              << " desde Rank " << status.MPI_SOURCE
                              << " (progreso actual: " << i << "/" << test.size() << " con K=" << currentK << ")."
                              << " Descartando resultados parciales y REINICIANDO test especulativo con K=" << newK << "...\n" << std::endl;

                    currentK = newK;
                    knn = KNN(train, currentK);
                    labels.clear();
                    matches = 0;
                    i = 0;
                    needRestart = true;
                    break; // Salir para reiniciar el bucle desde la primera muestra
                }
            } else if (status.MPI_TAG == TAG_FINISHED) {
                int finalK = 0;
                // MPI_Recv: Recibir la señal de finalización y el K óptimo definitivo
                MPI_Recv(&finalK, 1, MPI_INT, status.MPI_SOURCE, TAG_FINISHED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                tuningFinished = true;
                finalOptimalK = finalK;

                std::cout << "\n[Test Rank 0] <<< Recibida senal de FIN DE AJUSTE. K óptimo definitivo: " << finalOptimalK << std::endl;

                if (finalOptimalK != currentK) {
                    std::cout << "[Test Rank 0] El K óptimo (" << finalOptimalK << ") difiere del K actual (" << currentK
                              << "). Descartando progreso y ejecutando evaluación final con K definitivo...\n" << std::endl;
                    currentK = finalOptimalK;
                    knn = KNN(train, currentK);
                    labels.clear();
                    matches = 0;
                    i = 0;
                    needRestart = true;
                    break;
                } else {
                    std::cout << "[Test Rank 0] [ÉXITO ESPECULATIVO] El K actual (" << currentK
                              << ") coincide con el óptimo final. Se conserva el trabajo realizado ("
                              << i << "/" << test.size() << " muestras procesadas).\n" << std::endl;
                }
            } else {
                // Consumir cualquier otro tag residual
                MPI_Recv(nullptr, 0, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // Comprobar si hay más mensajes acumulados en la cola
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        }

        if (needRestart) {
            continue;
        }

        // Clasificar la muestra de test actual
        int pred = knn.classify(test[i]);
        labels.push_back(pred);
        if (pred == test[i].label) {
            matches++;
        }
        i++;

        // Progreso periódico
        if (i % 100 == 0 || i == test.size()) {
            std::cout << "[Test Rank 0] Muestra " << i << "/" << test.size()
                      << " (K=" << currentK << ", aciertos parciales: " << matches << ")\r" << std::flush;
        }
    }

    std::cout << std::endl;

    // -------------------------------------------------------------------------
    // Si la evaluación especulativa completó todas las muestras antes de recibir
    // TAG_FINISHED, espera la confirmación del tuning para validar el K óptimo.
    // -------------------------------------------------------------------------
    while (!tuningFinished) {
        int flag = 0;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        if (flag) {
            if (status.MPI_TAG == TAG_NEW_K) {
                int newK = 0;
                MPI_Recv(&newK, 1, MPI_INT, status.MPI_SOURCE, TAG_NEW_K, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (newK != currentK) {
                    std::cout << "\n[Test Rank 0] Test completo terminado previamente con K=" << currentK
                              << ", pero llegó un mejor K=" << newK << ". Reevaluando test con K=" << newK << "...\n" << std::endl;
                    currentK = newK;
                    knn = KNN(train, currentK);
                    labels.clear();
                    matches = 0;
                    for (std::size_t j = 0; j < test.size(); ++j) {
                        int f = 0; MPI_Status st;
                        if (MPI_Iprobe(MPI_ANY_SOURCE, TAG_FINISHED, MPI_COMM_WORLD, &f, &st) && f) break;
                        int pred = knn.classify(test[j]);
                        labels.push_back(pred);
                        if (pred == test[j].label) matches++;
                    }
                }
            } else if (status.MPI_TAG == TAG_FINISHED) {
                int finalK = 0;
                MPI_Recv(&finalK, 1, MPI_INT, status.MPI_SOURCE, TAG_FINISHED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                tuningFinished = true;
                finalOptimalK = finalK;

                std::cout << "\n[Test Rank 0] <<< Recibida señal final de FIN DE AJUSTE. K óptimo final = " << finalOptimalK << std::endl;
                if (finalOptimalK != currentK) {
                    std::cout << "[Test Rank 0] El K final óptimo (" << finalOptimalK << ") difiere de K=" << currentK
                              << ". Reevaluando con el K definitivo...\n" << std::endl;
                    currentK = finalOptimalK;
                    knn = KNN(train, currentK);
                    labels.clear();
                    matches = 0;
                    for (std::size_t j = 0; j < test.size(); ++j) {
                        int pred = knn.classify(test[j]);
                        labels.push_back(pred);
                        if (pred == test[j].label) matches++;
                    }
                } else {
                    std::cout << "\n[Test Rank 0] [ÉXITO ESPECULATIVO COMPLETO] El test finalizó antes del fin de tuning con el K óptimo ("
                              << currentK << ")! No se requiere recomputación alguna.\n" << std::endl;
                }
            }
        } else {
            // Pausa breve para evitar consumo de CPU mientras se espera el final del tuning
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    double testAccuracy = static_cast<double>(matches) / test.size();
    std::cout << "\n==========================================================" << std::endl;
    std::cout << "[Test Rank 0] Evaluación de test finalizada con éxito." << std::endl;
    std::cout << "[Test Rank 0] K definitivo utilizado: " << currentK << std::endl;
    std::cout << "[Test Rank 0] Accuracy final en test: " << testAccuracy * 100 << "% (" << matches << "/" << test.size() << ")" << std::endl;
    std::cout << "[Test Rank 0] Tiempo dedicado a test: " << MPI_Wtime() - testStart << " segundos." << std::endl;
    std::cout << "==========================================================" << std::endl;
}

/// <summary>
/// Función ejecutada por los procesos de ajuste (Rank > 0).
/// Reparten la evaluación de los distintos valores de K mediante validación cruzada.
/// Cuando encuentran un valor que supera el accuracy previo, lo notifican a Rank 0 con MPI_Isend.
/// </summary>
void runTuningWorker(int rank, int size, const std::vector<TSample>& train, unsigned int minK, unsigned int maxK, unsigned int folds)
{
    int numTuners = size - 1;
    int tunerId = rank - 1; // Índice entre los procesos de tuning (0 a numTuners-1)

    // Reparto cíclico de los valores de K entre los procesos de tuning
    std::vector<int> myCandidates;
    for (unsigned int k = minK; k <= maxK; ++k) {
        if ((k - minK) % numTuners == static_cast<unsigned int>(tunerId)) {
            myCandidates.push_back(k);
        }
    }

    std::cout << "[Tuning Rank " << rank << "] Asignados " << myCandidates.size() << " valores de K: [";
    for (std::size_t idx = 0; idx < myCandidates.size(); ++idx) {
        std::cout << myCandidates[idx] << (idx + 1 < myCandidates.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;

    CrossValidation crossValidation = CrossValidation::NFold(train, folds);
    double localBestAccuracy = -1.0;
    int localBestK = minK;
    double knownGlobalBestAccuracy = 0.0;

    for (int k : myCandidates) {
        // Comprobar si otros procesos de tuning han compartido un accuracy superior
        int flag = 0;
        MPI_Status status;
        while (MPI_Iprobe(MPI_ANY_SOURCE, TAG_TUNER_UPDATE, MPI_COMM_WORLD, &flag, &status) && flag) {
            double update[2] = {0.0, 0.0};
            MPI_Recv(update, 2, MPI_DOUBLE, status.MPI_SOURCE, TAG_TUNER_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (update[1] > knownGlobalBestAccuracy) {
                knownGlobalBestAccuracy = update[1];
            }
        }

        std::cout << "[Tuning Rank " << rank << "] Evaluando K=" << k << " (" << crossValidation.getName() << ")..." << std::endl;
        double tK = MPI_Wtime();
        double accuracy = crossValidation.validate(k);
        std::cout << "[Tuning Rank " << rank << "] K=" << k << " Accuracy: " << accuracy * 100
                  << "% completado en " << MPI_Wtime() - tK << "s" << std::endl;

        // Si este K supera el accuracy local y el global conocido:
        if (accuracy > localBestAccuracy && accuracy > knownGlobalBestAccuracy) {
            localBestAccuracy = accuracy;
            localBestK = k;
            knownGlobalBestAccuracy = accuracy;

            std::cout << "[Tuning Rank " << rank << "] >>> ¡NUEVO MEJOR K ENCONTRADO!: " << k
                      << " (Accuracy: " << accuracy * 100 << "%). Notificando a Rank 0 (test)..." << std::endl;

            // -----------------------------------------------------------------
            // MPI_Isend: Notificación asíncrona no bloqueante a Rank 0 (test)
            // -----------------------------------------------------------------
            MPI_Request req;
            MPI_Isend(&k, 1, MPI_INT, 0, TAG_NEW_K, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req); // Se completará de fondo sin bloquear al worker

            // Notificar asíncronamente a los demás procesos de tuning del nuevo accuracy alcanzado
            for (int otherRank = 1; otherRank < size; ++otherRank) {
                if (otherRank != rank) {
                    double update[2] = {static_cast<double>(k), accuracy};
                    MPI_Request updateReq;
                    MPI_Isend(update, 2, MPI_DOUBLE, otherRank, TAG_TUNER_UPDATE, MPI_COMM_WORLD, &updateReq);
                    MPI_Request_free(&updateReq);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Sincronización y consenso final entre todos los procesos de ajuste.
    // Creamos un sub-comunicador con MPI_Comm_split exclusivo para los tuners.
    // -------------------------------------------------------------------------
    MPI_Comm tuningComm;
    MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &tuningComm);

    struct {
        double accuracy;
        int rank;
    } localWinner = { localBestAccuracy, rank };

    struct {
        double accuracy;
        int rank;
    } globalWinner;

    // MPI_Allreduce con MPI_MAXLOC para determinar qué rank obtuvo el mayor accuracy
    MPI_Allreduce(&localWinner, &globalWinner, 1, MPI_DOUBLE_INT, MPI_MAXLOC, tuningComm);

    // El proceso ganador difunde su mejor K a todos los procesos de tuning
    int optimalK = localBestK;
    int rootWinnerInTuningComm = globalWinner.rank - 1; // Rango local en tuningComm
    MPI_Bcast(&optimalK, 1, MPI_INT, rootWinnerInTuningComm, tuningComm);

    // Rank 1 (líder de tuning) es el encargado de enviar la señal de fin (TAG_FINISHED) a Rank 0
    if (rank == 1) {
        std::cout << "\n[Tuning Lead Rank 1] Ajuste completado por todos los workers." << std::endl;
        std::cout << "[Tuning Lead Rank 1] K óptimo global seleccionado: " << optimalK
                  << " (Accuracy: " << globalWinner.accuracy * 100 << "% por Rank " << globalWinner.rank << ")." << std::endl;
        std::cout << "[Tuning Lead Rank 1] Enviando señal TAG_FINISHED a Rank 0 con K=" << optimalK << "...\n" << std::endl;

        // MPI_Send: Envío garantizado del K óptimo final a Rank 0
        MPI_Send(&optimalK, 1, MPI_INT, 0, TAG_FINISHED, MPI_COMM_WORLD);
    }

    MPI_Comm_free(&tuningComm);
}

/// <summary>
/// Punto de entrada del programa distribuido con MPI.
/// </summary>
int main(int argc, char** argv)
{
    // 1. Inicialización del entorno MPI
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::size_t trainLimit = DEFAULT_TRAIN_LIMIT;
    std::size_t testLimit = DEFAULT_TEST_LIMIT;

    if (argc > 1) {
        trainLimit = static_cast<std::size_t>(std::atol(argv[1]));
    }
    if (argc > 2) {
        testLimit = static_cast<std::size_t>(std::atol(argv[2]));
    }

    if (rank == 0) {
        std::cout << "==========================================================" << std::endl;
        std::cout << " KNN Distribuido con MPI - Paralelismo Funcional y Especulativo" << std::endl;
        std::cout << " Número de procesos (MPI Comm Size): " << size << std::endl;
        std::cout << " Configuración muestras: Train=" << trainLimit << ", Test=" << testLimit << std::endl;
        std::cout << "==========================================================" << std::endl;
    }

    // Cargar dataset MNIST en todos los procesos
    std::vector<TSample> train;
    std::vector<TSample> test;
    if (!loadDataset(train, test)) {
        std::cerr << "[Rank " << rank << "] Error al cargar el dataset MNIST." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (train.size() > trainLimit) {
        train.erase(train.begin() + trainLimit, train.end());
    }
    if (testLimit > 0 && test.size() > testLimit) {
        test.erase(test.begin() + testLimit, test.end());
    }

    // 2. Medición del tiempo global de ejecución con MPI_Wtime()
    MPI_Barrier(MPI_COMM_WORLD);
    double globalStart = MPI_Wtime();

    const unsigned int minK = 1;
    const unsigned int maxK = 10;
    const unsigned int folds = 5;

    if (size == 1) {
        // Fallback secuencial si solo hay 1 proceso
        std::cout << "[Rank 0] Modo secuencial (np=1). Ejecutando ajuste y evaluación..." << std::endl;
        int bestK = fineTuningSequential(minK, maxK, train, folds);
        std::cout << "Mejor K encontrado: " << bestK << std::endl;

        std::vector<int> labels;
        KNN knn(train, bestK);
        double accuracy = knn.classifyAndEvaluate(test, labels);
        std::cout << "Test accuracy: " << accuracy * 100 << "%" << std::endl;
    } else {
        // 3. Paralelismo funcional y ejecución especulativa:
        // Rank 0: Evaluación especulativa del conjunto de test
        // Ranks > 0: Procesos de ajuste de hiperparámetros (tuning)
        if (rank == 0) {
            // Rank 0 no participa en el Comm_split de tuning
            MPI_Comm dummyComm;
            MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, rank, &dummyComm);

            runSpeculativeTest(train, test, minK);
        } else {
            runTuningWorker(rank, size, train, minK, maxK, folds);
        }
    }

    // Sincronización final de todos los procesos para registrar el tiempo máximo global
    MPI_Barrier(MPI_COMM_WORLD);
    double localElapsed = MPI_Wtime() - globalStart;
    double maxElapsed = 0.0;

    // MPI_Reduce: Calcular el tiempo máximo entre todos los procesos (tiempo global real)
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n==========================================================" << std::endl;
        std::cout << " Tiempo global máximo de ejecución (todos los procesos): " << maxElapsed << " segundos" << std::endl;
        std::cout << "==========================================================" << std::endl;
    }

    // 4. Finalización limpia del entorno MPI
    MPI_Finalize();
    return 0;
}
