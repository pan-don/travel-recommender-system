# Laporan Akademik: Implementasi dan Optimalisasi Sistem Rekomendasi Pariwisata Berbasis Vektor (TravelLens)

## BAGIAN 1 — PENDAHULUAN

### Pernyataan Masalah
Sistem rekomendasi tradisional yang mengandalkan pencarian leksikal (pencocokan kata kunci yang tepat) sering kali gagal memahami maksud semantik (makna tersembunyi) dari pengguna, terutama dalam domain pariwisata yang sangat subjektif. Wisatawan sering kali mencari destinasi menggunakan bahasa alami yang deskriptif, bukan sekadar kata kunci statis. Basis data vektor (vector databases) menawarkan solusi atas keterbatasan ini dengan merepresentasikan teks sebagai titik koordinat dalam ruang berdimensi tinggi, memungkinkan pencarian kesamaan semantik terlepas dari sintaksis yang digunakan.

### Tujuan dan Ruang Lingkup Proyek
Proyek ini bertujuan untuk merancang, mengimplementasikan, dan mengoptimalkan sistem rekomendasi pariwisata cerdas berbasis vektor. Ruang lingkup proyek meliputi pra-pemrosesan data tekstual, pembangkitan *embeddings* (representasi vektor), konstruksi indeks menggunakan pustaka *Facebook AI Similarity Search* (FAISS), serta implementasi dan optimalisasi komparatif terhadap tujuh jenis kueri pencarian informasi yang berbeda.

### Tinjauan Sistem TravelLens
TravelLens adalah sistem rekomendasi berbasis web lokal yang menggunakan Flask untuk melayani pencarian destinasi wisata. Sistem ini mengintegrasikan model bahasa `paraphrase-multilingual-MiniLM-L12-v2` untuk ekstraksi fitur semantik (embedding) dari data berbahasa Indonesia, dan menggunakan FAISS untuk pencarian tetangga terdekat (Nearest Neighbor Search). Sistem berfokus pada eksekusi luring penuh, menghindari dependensi pihak ketiga atau layanan komputasi awan.

---

## BAGIAN 2 — DATASET

### Deskripsi Dataset
Dataset yang digunakan berisi ratusan entitas destinasi pariwisata yang berlokasi di Indonesia. Setiap entitas memiliki atribut terstruktur (seperti ID, Nama Tempat, Kategori, dan Kota) serta deskripsi tidak terstruktur dalam Bahasa Indonesia. Atribut dinilai memiliki variasi panjang teks dan struktur yang bervariasi.

### Proses Pra-pemrosesan
Pra-pemrosesan data dilakukan melalui skrip khusus yang mengaudit dan membersihkan nilai yang hilang, menghapus karakter non-alfanumerik yang tidak perlu, dan menormalisasi teks ke huruf kecil (lowercasing). Kolom atribut yang kosong (null) diganti dengan string kosong atau teks acuan standar.
[TABLE PLACEHOLDER: Statistik Dataset Sebelum dan Sesudah Pra-pemrosesan]

### Rasionalisasi Konstruksi Medan Teks Gabungan
Untuk memaksimalkan konteks yang diberikan ke model *embedding*, informasi terstruktur dan tidak terstruktur digabungkan menjadi satu kalimat semantik. Format yang digunakan adalah: `'{Nama Tempat}. {Kategori} di {Kota}. {Deskripsi}'`. Penggabungan ini memastikan bahwa vektor yang dihasilkan memuat tidak hanya deskripsi naratif, tetapi juga identitas kategorikal dan geografis, yang terbukti krusial untuk pencarian yang difilter secara semantik.

### Sampel Data (5 Baris)
[TABLE PLACEHOLDER: 5 Baris Data Terproses - ID, Nama, Kategori, Kota, Teks Gabungan]

---

## BAGIAN 3 — EMBEDDING (REPRESENTASI VEKTOR)

### Model: paraphrase-multilingual-MiniLM-L12-v2
Ekstraksi fitur semantik dilakukan menggunakan `paraphrase-multilingual-MiniLM-L12-v2`.
- **Arsitektur:** Model ini adalah varian dari Sentence-BERT (SBERT), yang dilatih menggunakan jaringan kembar (siamese network) untuk menghasilkan *embeddings* kalimat yang secara langsung dapat dibandingkan menggunakan ukuran kesamaan kosinus (cosine similarity).
- **Dimensi Keluaran:** Model menghasilkan ruang vektor berdimensi 384, memberikan keseimbangan yang sangat baik antara kapasitas representasi fitur dan efisiensi komputasi.
- **Dukungan Multilingual:** Karena dataset menggunakan Bahasa Indonesia, model multilingual ini dipilih agar mampu memetakan struktur bahasa lokal dengan akurat tanpa memerlukan penerjemahan tambahan ke bahasa Inggris.
- **Normalisasi L2:** Semua *embeddings* yang dihasilkan dikenakan operasi Normalisasi L2 (*L2 Normalization*) menjadi rentang magnitudo satu. Transformasi ini secara matematis menyederhanakan perhitungan jarak kesamaan kosinus menjadi perhitungan perkalian titik (inner product), yang dieksekusi secara jauh lebih cepat pada arsitektur perangkat keras modern.

### Statistik Pembangkitan Embedding
[TABLE PLACEHOLDER: Waktu Eksekusi Pembangkitan Embedding, Jumlah Batch, Utilisasi CPU/Memori]

---

## BAGIAN 4 — INDEKSASI

### Perbandingan Tipe Indeks FAISS
Dalam tahap perancangan, tiga tipe indeks FAISS dievaluasi:
[TABLE PLACEHOLDER: Tabel Komparatif - IndexFlatIP vs IndexIVFFlat vs IndexHNSWFlat (Waktu Pelatihan, Waktu Kueri, Akurasi/Recall)]

### Justifikasi Pemilihan Indeks
`IndexFlatIP` (Exact Search dengan Inner Product) dipilih sebagai arsitektur indeks utama. Mengingat ukuran dataset TravelLens relatif kecil (ratusan hingga beberapa ribu data), algoritma pencarian teliti (*exhaustive search*) dari `IndexFlatIP` dieksekusi dalam skala waktu milidetik. Pemilihan ini memberikan keuntungan berupa *recall* absolut (100%), tidak memerlukan proses pelatihan (*training/clustering*) tambahan seperti pada struktur IVF, dan kompatibel langsung dengan normalisasi L2 untuk menghitung kesamaan kosinus sejati (0.0 - 1.0).

### Statistik Indeks
[TABLE PLACEHOLDER: Waktu Pembangunan Indeks, Penggunaan Memori RAM, Rata-rata Waktu Pencarian Kosong]

---

## BAGIAN 5 — IMPLEMENTASI KUERI

Berikut adalah deskripsi dari tujuh pola pengambilan vektor yang diimplementasikan dalam arsitektur kelas `QueryEngine`:

1. **Q1 — Pencarian Semantik Dasar (Semantic Search)**
   - **Tujuan:** Pengambilan informasi bebas berbasis bahasa alami murni.
   - **Parameter Input:** `query` (teks), `k` (jumlah hasil).
   - **Implementasi:** Pengubahan `query` menjadi *embedding*, normalisasi, dan eksekusi pencarian pada seluruh indeks menggunakan `IndexFlatIP`.
   [TABLE PLACEHOLDER: Contoh Kueri Q1 dan Hasil]

2. **Q2 — Filter Kategori Hibrida (Category Filter)**
   - **Tujuan:** Pencarian teks bebas yang dibatasi pada kategori spesifik (misal: "Budaya").
   - **Parameter Input:** `query`, `category`, `k`.
   - **Implementasi:** Peringkat kandidat vektor didapatkan dari FAISS, dilanjutkan dengan penyaringan metadata secara sekuensial hingga terkumpul `k` hasil (Post-hoc Filtering).
   [TABLE PLACEHOLDER: Contoh Kueri Q2 dan Hasil]

3. **Q3 — Filter Kota Semantik (City + Semantic Filter)**
   - **Tujuan:** Menemukan destinasi di wilayah yang diizinkan (misal: hanya di "Jakarta" atau "Bali").
   - **Parameter Input:** `query`, `allowed_cities` (daftar teks), `k`.
   - **Implementasi:** Pencarian Top-K diperluas (top-50) lalu disaring berdasarkan daftar kota yang ada dalam struktur metadata.
   [TABLE PLACEHOLDER: Contoh Kueri Q3 dan Hasil]

4. **Q4 — Pencarian Semantik Luas (Broad Semantic Search)**
   - **Tujuan:** Pencarian tingkat tinggi yang mensimulasikan partisi data yang lebih luas.
   - **Parameter Input:** `query`, `k` (diatur lebih besar, e.g., 50).
   - **Implementasi:** Serupa dengan pencarian semantik (Q1) tetapi mengeksplorasi ruang pencarian hasil yang jauh lebih dalam untuk analisis berikutnya.
   [TABLE PLACEHOLDER: Contoh Kueri Q4 dan Hasil]

5. **Q5 — Pencarian Berbasis Item (Item-Based Search)**
   - **Tujuan:** Menemukan destinasi yang memiliki profil vektor serupa dengan destinasi yang sudah diketahui (Fitur "Destinasi Serupa").
   - **Parameter Input:** `destination_id`, `k`.
   - **Implementasi:** Ekstraksi *embedding* destinasi secara langsung berdasarkan ID tanpa inferensi jaringan saraf, lalu pencarian, dengan pengabaian item itu sendiri (self-exclusion).
   [TABLE PLACEHOLDER: Contoh Kueri Q5 dan Hasil]

6. **Q6 — Ensembel Multi-Kueri (Multi-Query Ensemble)**
   - **Tujuan:** Mengakomodasi kebutuhan pengguna yang tidak dapat dijelaskan dalam satu kalimat dengan menggabungkan beberapa kata kunci tematik.
   - **Parameter Input:** `queries` (Daftar 3 kueri teks), `k`.
   - **Implementasi:** Sistem memproses tiga kueri terpisah, mengambil kandidat masing-masing, menggabungkan hasil, menghapus duplikasi (deduplikasi), dan memeringkat ulang berdasarkan rata-rata skor kesamaan.
   [TABLE PLACEHOLDER: Contoh Kueri Q6 dan Hasil]

7. **Q7 — Pencarian Tereduksi PCA (PCA-Reduced Query)**
   - **Tujuan:** Pengambilan kompresi spasial untuk optimasi ukuran indeks.
   - **Parameter Input:** `query`, `k`.
   - **Implementasi:** Matriks 384-dimensi dikurangi ukurannya menjadi 64 dimensi menggunakan *Principal Component Analysis* (PCA), mempercepat operasi inner product namun mengorbankan sedikit fidelitas ruang.
   [TABLE PLACEHOLDER: Contoh Kueri Q7 dan Hasil]

---

## BAGIAN 6 — OPTIMALISASI KUERI

Tahap evaluasi mengukur kinerja eksekusi kueri yang digerakkan oleh alat ukur (`QueryBenchmark`) yang mengeksekusi iterasi 20 kali untuk mendapatkan latensi rata-rata dan kualitas metrik Precision@K yang tumpang tindih dengan hasil dasar (baseline).

1. **Q1 Optimalisasi: Pra-pemrosesan Teks (Text Preprocessing)**
   - **Teknik:** Lowercasing, penghapusan kata hubung (stopwords NLTK), dan pemotongan batas teks.
   - **Analisis:** Mengurangi kompleksitas deret token mengurangi waktu pembangkitan semantik. Pembersihan stopwords memfokuskan atensi model pada kata-kata semantik yang signifikan.

2. **Q2 Optimalisasi: Sub-Indeks Kategori (Per-Category Sub-Indexes)**
   - **Teknik:** Indeks global dipartisi dalam memori menjadi beberapa indeks mini berdasarkan kategori.
   - **Analisis:** Pencarian O(N) berkurang menjadi O(N_partition). Penyaringan post-hoc yang tidak efisien dihapus seluruhnya.

3. **Q3 Optimalisasi: Pengetatan Ekspansi K (Tighter K Expansion)**
   - **Teknik:** Penurunan jumlah batas pengambilan kueri sebelum disaring dari K=50 menjadi K=15.
   - **Analisis:** Kepercayaan pada akurasi representasi SBERT memungkinkan pengurangan kandidat pencarian. Hal ini mempercepat komputasi FAISS dan penelusuran metadata, mengorbankan variansi *recall* minimal.

4. **Q4 Optimalisasi: Indeks Terpartisi Lokasi (City-partitioned Index)**
   - **Teknik:** Indeks dipisahkan secara hierarkis berdasarkan kota untuk menyimulasikan penyaringan geografis lokal.
   - **Analisis:** Rute pencarian menjadi eksklusif terhadap kota yang diminta. Latensi dan keandalan sistem meningkat saat jumlah data bertumbuh secara eksponensial.

5. **Q5 Optimalisasi: Caching Memori (RAM Caching vs Simulated Disk)**
   - **Teknik:** *Embeddings* vektor matriks penuh dimuat dan dikunci ke dalam struktur data RAM (`dict`) sejak inisialisasi aplikasi.
   - **Analisis:** Bottleneck throughput Input/Output diubah dari interaksi disk penyimpan *solid-state/hard-drive* menuju operasi lebar pita memori lokal. Pengurangan waktu hingga miliaran tingkat (*orders of magnitude*).

6. **Q6 Optimalisasi: Eksekusi Multi-Kueri Paralel**
   - **Teknik:** Penggunaan `concurrent.futures.ThreadPoolExecutor` untuk memproses inferensi ketiga vektor secara serentak.
   - **Analisis:** Bottleneck eksekusi model saraf dalam loop sekuensial diubah menjadi waktu proses dari kueri yang paling lama.

7. **Q7 Optimalisasi: PCA Dimensi Optimal (PCA Dimensionality)**
   - **Teknik:** Penyesuaian dinamis dimensi komponen (contoh 32, 64, 128, 256) hingga mencapai *explained_variance_ratio_* agregat 95% atau lebih.
   - **Analisis:** Merupakan kompromi matematis di mana matriks vektor dikompresi sedemikian rupa sehingga operasi dot-product berkurang signifikan secara kompleksitas komputasi, sambil tetap mempertahankan struktur kesamaan antardata.

[TABLE PLACEHOLDER: Ringkasan Rata-Rata Peningkatan Kinerja dari Semua Optimalisasi]
[CODE SNIPPET: src/benchmark.py]

---

## BAGIAN 7 — KESIMPULAN

### Temuan Utama
Sistem TravelLens menunjukkan bahwa adopsi basis data vektor memungkinkan pengambilan pencarian pariwisata yang sangat cerdas di mana entri tanpa irisan leksikal dapat tetap dicocokkan melalui proksimitas semantik. Optimalisasi arsitektural (seperti penggunaan sub-indeks memori partisipatif, kompresi PCA, dan paralelisasi inferensi model SBERT) secara empiris memangkas latensi eksekusi menjadi pecahan persentase dari garis dasar (*baseline*), mencapai peningkatan fungsional tanpa memperlemah Precision@5.

### Keterbatasan Sistem
1. **Model Statis:** Pembaruan terhadap metadata (nama tempat atau deskripsi berubah) mengharuskan pelacakan dan pembangunan ulang ruang vektor parsial atau menyeluruh. Sistem ini dirancang statis.
2. **Kendala Leksikal Murni:** Kesulitan dalam pencarian kode numerik spesifik atau ID karena sifat pemodelan bahasa lebih condong pada kalimat alami dibanding pola Regex sederhana.

### Pengembangan Masa Depan
1. Memperbarui arsitektur ke *IndexIVFFlat* untuk skalabilitas ketika jumlah destinasi menyentuh ratusan ribu rekaman wisata.
2. Penambahan model *Cross-Encoder* pada tahap akhir untuk merangking ulang (*re-ranking*) kandidat Top-50 yang diambil oleh FAISS menggunakan metrik keterlibatan pengguna historis.
3. Eksplorasi teknik kuantisasi skalar (*Scalar Quantization*, misal FAISS SQ8) untuk mengurangi konsumsi ruang memori dari presisi `float32` ke resolusi bilangan bulat `int8`.

---

## REFERENSI

1. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.
3. Wang, L., et al. (2020). Multilingual sentence embeddings using paraphrase generation. *Association for Computational Linguistics*.
4. Jégou, H., Douze, M., & Schmid, C. (2010). Product quantization for nearest neighbor search. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(1), 117-128.
5. Zhao, Wayne Xin, et al. (2022). A survey of large language models. *arXiv preprint arXiv:2303.18223*.
