# HoloTrade

High-performance federated Deep Learning / Vector Symbolic (VSA / Hyperdimensional Computing) framework exploring alongside traditional Deep Learning for deep relational embedding for HFT Order Books combined with ultra-low-latency financial signal processing and execution at the microsecond edge via VSA, crucial to HFT systems.

> **Status:** Active Systems Research & Core Engine Engineering.

---

## 1. Architectural Overview

Traditional Deep Learning (DL) models excel at discovering complex, non-linear relationships and market regimes from historical or macroscopic data. However, running heavy matrix multiplication and backpropagation loops on the live execution edge introduces high inference latency—rendering them impractical in the microsecond realm of High-Frequency Trading (HFT).

`HoloTrade` resolves this bottleneck via an **asymmetrical, federated architecture** split into two distinct execution loops:

              +----------------------------------------+
              |         OFFLINE RESEARCH LOOP          |
              |  - Historical L3 Limit Order Book data |
              |  - PyTorch Deep Learning Brain         |
              |  - Identifies Regimes & Probabilities  |
              +-------------------+--------------------+
                                  |
                       Passes Weights / Patterns
                                  |
                                  v  [pybind11 Zero-Copy Bridge]
              +-------------------+--------------------+
              |         ONLINE EXECUTION ENGINE        |
              |  - Maps real-time L3 LOB into Hypervecs|
              |  - Compile-Time Dimension Optimization |
              |  - SIMD/AVX-512 Parallel Bitwise Math  |
              |  - Sub-microsecond Distance Matching   |
              +----------------------------------------+

1. **The Brain (Offline / Slow Loop):** A Python/PyTorch loop running in the research environment to process historical Level 3 Limit Order Book (LOB) data, generating high-dimensional policy embeddings or regime patterns.
2. **The Muscle (Online / Fast Loop):** A native C++17 execution core sitting at the live edge. It ingests real-time streaming market telemetry, encodes it instantly into the hyperdimensional space using bitwise primitives, and performs sub-microsecond distance matching against the pre-computed patterns to execute trades.

---

## 2. Low-Latency Systems Engineering

To meet the efficiency demands of modern trading infrastructure, the core execution engine is built around tight hardware-adjacent optimization principles:

* **Compile-Time Optimization:** Employs C++ templates (`template <size_t DIM>`) to fix hypervector sizing (e.g., 10,000 bits) at compile time. This eliminates runtime heap allocation overhead and allows the compiler to inline execution paths aggressively.
* **Cache-Line Alignment:** Forces explicit data alignment (`alignas(64)`) to map data structures cleanly to CPU cache lines, minimizing cache misses during high-throughput data streams.
* **SIMD Bit-Parallelism:** Replaces floating-point math with highly parallelizable, low-precision binary vector operations (bitwise XOR binding, cyclic shift permutations). Employs AVX-512 intrinsic structures to process hundreds of bits in a single instruction cycle.
* **Hardware-Native Bit Counting:** Utilizes compiler-native popcount instructions (`__builtin_popcountll`) to execute high-dimensional Hamming distance matching across arrays of `uint64_t` words instantly at the hardware layer.

---

## 3. Production Deployment Strategy

While this project is organized as a monorepo for presentation, open-source legibility, and unified automated testing, the components are engineered to be entirely decoupled for production deployment:

* **The Quantitative Research Silo:** The Python research framework operates independently within historical data environments, outputting serialized associative memory matrices.
* **The Exchange Colocation Silo:** The compiled C++ core engine strips out all training dependencies, Python runtimes, and garbage collection overhead. It runs standalone at the exchange colocation center, receiving serialized pattern updates via low-overhead binary protocols and executing live inference under strict sub-microsecond requirements.

---

## 4. Repository Structure

```text
holo-trade/
│
├── CMakeLists.txt          # Configures C++ compilation and pybind11 linking
├── requirements.txt        # Python research dependencies
│
├── src/                    # C++ Execution Engine Core
│   ├── main.cpp            # Standalone C++ engine entry point
│   ├── vsa_engine.hpp      # Compile-time templates & SIMD bitwise operations
│   ├── vsa_engine.cpp      # Binding, bundling, and distance matching implementations
│   └── bindings.cpp        # Pybind11 wrapper exposing the C++ engine to Python
│
├── research/               # Python Research & Alpha Generation
│   ├── train_brain.py      # PyTorch training pipelines for regime detection
│   └── test_bridge.py      # Performance profiling of the C++ execution engine
│
└── tests/                  # C++ and Python validation suites