# Reproducing "A Primer for Neural Arithmetic Logic Modules"

Repository này chứa mã nguồn tái lập (re-implementation) các thí nghiệm **Single Layer Task** từ bài báo:
> **A Primer for Neural Arithmetic Logic Modules** > *Bhumika Mistry, Katayoun Farrahi, Jonathon Hare (2022)*

Mục tiêu của dự án là so sánh khả năng **Ngoại suy (Extrapolation)** của các mạng nơ-ron số học khác nhau trên 4 phép tính cơ bản (+, -, ×, ÷).

## Cấu Trúc Thư Mục

Dự án được tổ chức theo hướng module hóa, tách biệt dữ liệu, mô hình và quy trình huấn luyện:

```text
my_nalm_repro/
├── data/
│   ├── __init__.py
│   └── generator.py         # Sinh dữ liệu nội suy (Train) và ngoại suy (Test) theo Table 3
├── models/
│   ├── __init__.py
│   ├── nac.py               # Neural Accumulator
│   ├── nalu.py              # Neural Arithmetic Logic Unit (Original)
│   ├── inalu.py             # Improved NALU (với clipping & stability)
│   ├── gnalu.py             # Golden Ratio NALU
│   ├── nau.py               # Neural Addition Unit (SOTA cho cộng/trừ)
│   ├── nmu.py               # Neural Multiplication Unit (SOTA cho nhân)
│   ├── npu.py               # Neural Power Unit (Complex)
│   └── realnpu.py           # Real NPU (SOTA cho chia & số âm)
├── training/
│   ├── __init__.py
│   └── regularization.py    # Scheduler & Loss phạt cho NAU, NMU, NPU (Table 6, 7)
├── results/                 # Chứa file kết quả CSV sau khi chạy
├── logs/                    # Chứa log chi tiết khi chạy script hàng loạt
├── main.py                  # Entry point chính để chạy huấn luyện
├── run_paper_benchmark.sh   # Script tự động chạy toàn bộ thí nghiệm
└── requirements.txt         # Các thư viện cần thiết
```

## Cài đặt
Cài đặt các thư viện trong requirements.txt:
```text
pip install -r requirements.txt
```

## Hướng dẫn chạy (Quick Start)
Bạn có thể chạy benchmark cho một mô hình cụ thể bằng lệnh python main.py

Các tham số chính:
    - --model: Tên mô hình (nac, nalu, inalu, gnalu, nau, nmu, npu, realnpu)
    - --op: Phép toán (add, sub, mul, div)
    - --range: Vùng dữ liệu (U1 đến U9). U1 là khó nhất (bao gồm số âm lớn)
    - --gpu: Sử dụng GPU để train (khuyên dùng)

Ví dụ:
```text
python main.py --model nau --op add --range U1 --gpu
```

## Chạy toàn bộ Benchmark
Để tái hiện lại bảng kết quả tổng hợp trong bài báo, hãy sử dụng script tự động hóa. Script này sẽ chạy quét qua các model và phép toán tương ứng, lặp lại 25 seeds cho mỗi cấu hình.
```text
// Cấp quyền thực thi
chmod +x run_paper_benchmark.sh
// Chạy script
./run_paper_benchmark.sh
```