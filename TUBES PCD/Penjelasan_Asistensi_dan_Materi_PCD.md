# Panduan Asistensi & Dokumentasi Teknis Tahapan Pengolahan Citra (Minggu 11–12 Menuju Minggu 13)

**Judul Proyek:** Otomatisasi Penggantian Latar Belakang Pas Foto Formal pada Mahasiswa ITERA Menggunakan Segmentasi Warna HSV/LAB dan Deteksi Tepi Sobel  
**Kelompok:** Cihuy (Kelas RA — Topik 10)  
**Dosen Pengampu:** Andika Setiawan, S.Kom., M.Cs.  

---

## 1. Ringkasan Status Progres (Hingga Akhir Minggu 12 / Awal Minggu 13)

Sesuai jadwal pada Panduan Tugas Besar PCD:
* **Minggu 11 (Selesai):** Penyusunan proposal, penyesuaian ruang lingkup dataset primer menjadi **50 citra mahasiswa ITERA** (25 *half-body* dan 25 *close-shot*), perancangan *ground truth binary mask*, dan perancangan *baseline* klasik.
* **Minggu 12 (Selesai — Fokus Asistensi Saat Ini):** Implementasi penuh seluruh tahapan *pipeline* pengolahan citra secara modular di dalam `tubes_pcd_pipeline.ipynb` (mulai dari standarisasi geometri, pra-pemrosesan *image enhancement*, segmentasi hibrida warna + konvolusi tepi Sobel, operasi logika & morfologi, hingga *feathering* & *alpha blending*), serta penyelesaian Notebook Pengayaan Mandiri (`PCD_YOLO_Bagian_A.ipynb` dan `PCD_YOLO_Bagian_B.ipynb`).
* **Minggu 13 (Tahap Selanjutnya):** Pelaksanaan *hyperparameter tuning* (*grid search*) pada 40 citra *Tuning Set*, evaluasi kuantitatif metrik *Intersection over Union* (IoU) dan waktu komputasi pada 10 citra *Testing Set* untuk ke-4 skenario (*Baseline*, *Ablasi 1*, *Ablasi 2*, dan *Metode Usulan*), serta analisis kasus gagal (*failure cases*) pada area rambut/bahu.

---

## 2. Pemetaan Arsitektur Pipeline terhadap Materi Perkuliahan (Materi 01–10)

Agar saat asistensi maupun presentasi akhir setiap anggota kelompok dapat menjawab pertanyaan dosen secara teoritis, tabel berikut memetakan setiap fungsi yang sudah diimplementasikan di `tubes_pcd_pipeline.ipynb` dengan konsep dan rumus di slide materi kuliah:

| Tahap Pipeline | Fungsi pada Kode | Konsep & Aras Komputasi (Materi Kuliah) | Referensi Slide Materi Dosen |
| :--- | :--- | :--- | :--- |
| **0. Akuisisi & Standarisasi** | `fit_3x4()`, `load_pair()` | Digitalisasi Citra (*Sampling* & Kuantisasi), Format Citra (*Truecolor* 24-bit & Citra Biner), serta **Operasi Geometri (Penskalaan / *Image Zooming* & *Cropping*)** | **Materi 01** (Slide 12–17), **Materi 03** (Slide 10–25), **Materi 04** (Slide 33–45), **Materi 05** (Slide 44, 54–57) |
| **1a. Perbaikan Kontras Lokal** | `preprocess()` $\rightarrow$ `cv2.createCLAHE()` | **Aras Global / Lokal (ROI):** Transformasi Ruang Warna (`BGR` $\rightarrow$ `LAB`) dan **Perbaikan Lokal dengan Perataan Histogram (*Local Histogram Equalization*)** pada kanal luminansi ($L$) | **Materi 06** (Slide 15–28), **Materi 08** (Slide 12–18), **Materi 09** (Slide 4–27 & **Slide 47–48**) |
| **1b. Reduksi Derau (*Smoothing*)** | `preprocess()` $\rightarrow$ `cv2.GaussianBlur()` | **Aras Lokal ($O_{\text{lokal}}$):** Konvolusi Spasial 2D dengan **Penapis Lolos-Rendah (*Low-Pass Filter* / *Gaussian Smoothing*)** berukuran $5 \times 5$ | **Materi 05** (Slide 14–15), **Materi 07** (Slide 14–37), **Materi 10** (Slide 6–32) |
| **2a. Segmentasi Warna Latar** | `background_mask()` | **Aras Titik ($O_{\text{titik}}$):** Pengambangan (*Color Thresholding*) rentang multi-kanal berbasis statistik median 3 *patch* sampel latar ($30 \times 30$ piksel) | **Materi 05** (Slide 3–5), **Materi 08** (Slide 10, 51–52) |
| **2b. Deteksi Tepi Konvolusi Sobel** | `sobel_edges()` | **Aras Titik & Aras Lokal:** Konversi *Grayscale Luminance* ($Y$) diikuti **Konvolusi Spasial Penapis Lolos-Tinggi (*High-Pass / 1st Derivative Gradient*)** Kernel Sobel $3 \times 3$ ($G_x, G_y$) | **Materi 05** (Slide 12–13), **Materi 07** (Slide 15–38), **Materi 10** (Slide 38–49) |
| **3a. Penggabungan & Pembersihan Masker** | `border_connected()`, `clean_foreground()` | **Aras Titik & Aras Objek ($O_{\text{objek}}$):** **Operasi Logika Boolean (`AND`, `OR`, `NOT`)**, Morfologi (*Closing* & *Opening*), Seleksi Komponen Terbesar, dan *Hole Filling* | **Materi 05** (Slide 17, 37–43), **Materi 09** (Slide 49–50) |
| **3b. Penghalusan Tepi & Ganti Latar** | `run_pipeline()`, `replace_bg()` | **Aras Lokal & Aras Titik:** *Gaussian Edge Feathering* ($9 \times 9$) untuk membentuk *Alpha Matte*, diikuti **Operasi Aritmetika Matriks (Perkalian Citra / *Masking* dan Penjumlahan Citra)** | **Materi 05** (Slide 19–31), **Materi 09** (Slide 49–60), **Materi 10** (Slide 25–30) |

---

## 3. Penjelasan Mendetail Setiap Tahapan Pengolahan Citra (Bahan Penjelasan Asistensi)

Berikut adalah penjelasan mendalam dari setiap tahap dari sisi **teori kuliah**, **alasan pemilihan metode**, dan **implementasi kodenya**:

---

### Tahap 0: Digitalisasi, Format Citra, dan Standarisasi Geometri (`fit_3x4`)

#### A. Kaitan dengan Materi Kuliah
1. **Representasi Matriks & Resolusi Spasial (Materi 01 & 03):**
   Citra digital direpresentasikan sebagai matriks 2D berukuran $M \times N$, di mana setiap elemen matriks merupakan piksel $f(x, y)$ hasil pencuplikan ruang (*spatial sampling*) dan kuantisasi intensitas (*gray-level quantization*). Foto dari kamera *smartphone* anggota kelompok memiliki resolusi awal yang sangat besar ($12\text{ MP} - 48\text{ MP}$, misalnya $3000 \times 4000$ piksel) dan rasio aspek yang beragam ($4:3$, $16:9$, atau $1:1$).
2. **Format Citra (Materi 04 Slide 33–45):**
   * Citra input merupakan **Citra Warna-Sejati (*Truecolor Image* 24-bit)** yang tersimpan sebagai 3 matriks berukuran $M \times N$ untuk kanal $R, G, B \in [0, 255]$.
   * Citra *ground truth* dan masker hasil segmentasi merupakan **Citra Biner (*Binary Image*)** yang hanya memiliki dua nilai logika: `0` (hitam / latar belakang) dan `255` atau `1` (putih / objek manusia).
3. **Operasi Geometri — Penskalaan Citra / *Image Zooming* (Materi 05 Slide 44 & 54–57):**
   Transformasi geometri memetakan koordinat $(x, y)$ ke koordinat baru $(x', y')$ dengan persamaan penskalaan:
   $$x' = s_x \cdot x, \quad y' = s_y \cdot y$$
   Di mana $s_x$ dan $s_y$ adalah faktor skala horizontal dan vertikal. Pada pengecilan ukuran citra (*zoom-in / downsampling*), nilai piksel baru diperoleh dengan merata-ratakan piksel-piksel tetangga yang bersesuaian (di OpenCV diimplementasikan dengan `cv2.INTER_AREA`).

#### B. Implementasi pada Kode Kita
```python
SIZE = (720, 960)  # (lebar N, tinggi M) dengan rasio pasfoto formal 3:4

def fit_3x4(a, interp):
    h, w = a.shape[:2]
    target_w = int(round(h * 3 / 4))
    if w > target_w:                       # jika terlalu lebar: center-crop kiri-kanan
        x0 = (w - target_w) // 2; a = a[:, x0:x0 + target_w]
    elif w < target_w:                     # jika terlalu tinggi: center-crop atas-bawah
        new_h = int(round(w * 4 / 3)); y0 = (h - new_h) // 2; a = a[y0:y0 + new_h]
    return cv2.resize(a, SIZE, interpolation=interp)
```
* **Mengapa `INTER_AREA` untuk foto dan `INTER_NEAREST` untuk mask?**
  * Foto asli diperkecil menggunakan `cv2.INTER_AREA` (perata-rataan area piksel, sesuai Materi 05 Slide 57) agar bebas dari efek *aliasing* (bergerigi).
  * Masker *ground truth* diperkecil menggunakan `cv2.INTER_NEAREST` (*nearest neighbor*) dan di-ambangkan ulang `(gt > 127) * 255` agar nilainya tetap murni biner (`0` atau `255`) dan tidak muncul nilai abu-abu di perbatasan.

---

### Tahap 1: Pra-pemrosesan / *Image Enhancement* (`preprocess`)

Tahap ini bertujuan memperbaiki kualitas citra sebelum disegmentasi karena foto mahasiswa yang diambil secara mandiri di dalam (*indoor*) maupun luar ruangan (*outdoor*) sering mengalami degradasi berupa **kontras rendah (*low-contrast*), pencahayaan tidak merata (*under/over-exposed*)**, serta **derau sensor (*noise*)** (Materi 01 Slide 18–20 & Materi 08 Slide 2–4).

#### A. Perbaikan Kontras Lokal dengan CLAHE (Materi 06, 08, dan 09)
1. **Teori Histogram & Kontras (Materi 06 Slide 15–28 & Materi 08 Slide 12–18):**
   Kontras menyatakan sebaran terang (*lightness*) di dalam citra. Citra berkontras rendah memiliki histogram yang sempit, sehingga nilai variansi ($\sigma^2 = \sum (k - \mu)^2 h(k)$) dan simpangan bakunya ($\sigma$) kecil. Akibatnya, batas antara warna pakaian/rambut dan latar belakang menjadi sulit dibedakan.
2. **Mengapa CLAHE dan Bukan *Global Histogram Equalization* Biasa? (Materi 09 Slide 4–27 & Slide 47–48):**
   * Pada **Materi 09 Slide 4–9**, *Perataan Histogram Global* (*Global Histogram Equalization*, operasi **Aras Global**) mengubah derajat keabuan $r$ menjadi $s = T(r) = \int_0^r P_r(w)\,dw$ agar distribusinya seragam (*uniform*).
   * Namun, sesuai **Materi 09 Slide 26–27**, perataan histogram global **tidak selalu memberikan hasil yang bagus**: pada citra dengan latar belakang luas yang seragam, perataan global justru membuat citra terlalu terang (*over-exposed*) dan memperkuat derau (*noise*) pada latar belakang!
   * Solusinya dijelaskan pada **Materi 09 Slide 47 (*"Perbaikan lokal dengan perataan histogram"*)**: perataan histogram dilakukan secara lokal pada area/blok kecil (*tile* / *Region of Interest*). Teknik **CLAHE (*Contrast Limited Adaptive Histogram Equalization*)** membagi citra menjadi *grid* lokal berukuran $8 \times 8$ dan membatasi penguatan kontras (*clip limit* $= 2,0$) agar derau pada tembok latar tidak ikut diperkuat.
3. **Mengapa Hanya Diterapkan pada Kanal $L$ (Ruang Warna CIE-LAB)?**
   Jika perataan histogram diterapkan langsung pada kanal $R, G, B$ secara terpisah, proporsi warna asli wajah dan latar akan berubah (*color shifting*). Dengan mengonversi citra ke ruang warna **LAB**, kanal $L$ (*Lightness/Luminance*) terpisah sepenuhnya dari kanal warna $a$ dan $b$. CLAHE hanya diterapkan pada kanal $L$, sehingga kontras meningkat tajam tanpa merusak warna asli!

#### B. Pelembutan Citra (*Image Smoothing*) dengan *Gaussian Low-Pass Filter* (Materi 07 & 10)
1. **Teori Konvolusi & Penapis Lolos-Rendah (Materi 07 Slide 14–37 & Materi 10 Slide 6–32):**
   * Pelembutan citra termasuk operasi **Aras Lokal ($O_{\text{lokal}}$)** yang memodifikasi setiap piksel berdasarkan ketetanggaannya $N(x,y)$ melalui operator **konvolusi dwimatra (2D)**:
     $$g(x, y) = f(x, y) * h(x, y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} h(s, t) \, f(x - s, y - t)$$
   * Sesuai **Materi 10 Slide 23–24**, penapis pelembutan merupakan **Penapis Lolos-Rendah (*Low-Pass Filter*)** yang menekan komponen berfrekuensi tinggi (derau/bintik sensor kamera) dan meloloskan komponen berfrekuensi rendah. Aturan wajibnya: seluruh bobot kernel bernilai positif dan **jumlah seluruh bobot sama dengan 1** ($\sum h(s,t) = 1$) agar kecerahan rata-rata citra tidak menjadi lebih gelap atau lebih terang.
   * Sesuai **Materi 10 Slide 25–32 (*Averaging vs Gaussian Smoothing*)**, kita memilih **Gaussian Filter ($5 \times 5$)** dibandingkan *Mean/Averaging Filter* karena bobot Gaussian mengikuti distribusi normal 2D (bobot terbesar berada di piksel pusat dan meluruh secara mulus ke tepi kernel). Hasilnya, derau frekuensi tinggi hilang namun tepi objek tidak mengalami *blocky blurring* separah *Mean Filter*.

#### C. Implementasi pada Kode Kita
```python
def preprocess(img, p):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=p['clahe_clip'], tileGridSize=(p['clahe_grid'],) * 2)
    lab[..., 0] = clahe.apply(lab[..., 0])          # CLAHE hanya di kanal L (Luminance)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    k = p['blur_k'] | 1                             # ukuran kernel ganjil 5x5
    return cv2.GaussianBlur(out, (k, k), 0)
```

---

### Tahap 2: Segmentasi Hibrida — Pengambangan Warna (*Color Thresholding*) & Konvolusi Tepi Sobel

Kelemahan utama metode klasik jika hanya memakai *Color Thresholding* tunggal adalah: ketika mahasiswa memakai baju atau hijab yang warnanya agak mirip dengan warna latar (atau terkena bayangan lampu), area baju/bahu ikut terdeteksi sebagai latar belakang (*bocor*). Sebaliknya, jika hanya memakai deteksi tepi, kita hanya mendapatkan garis kontur yang bolong-bolong, bukan area objek yang utuh. Oleh karena itu, proyek ini menggabungkan keduanya secara hibrida.

#### A. Cabang 1 — Pengambangan Warna Adaptif Berbasis Sampel Patch (`background_mask`)
1. **Kaitan dengan Materi Kuliah (Materi 05 Slide 3–5 & Materi 08 Slide 10):**
   * Pengambangan (*Thresholding*) adalah operasi **Aras Titik ($O_{\text{titik}}$)** yang memetakan citra ke dalam citra biner berdasarkan nilai ambang $T$:
     $$g(x, y) = \begin{cases} 255 & \text{jika } \|f(x, y) - \mathbf{c}_{\text{ref}}\| \le \mathbf{T} \\ 0 & \text{lainnya} \end{cases}$$
2. **Inovasi pada Kode Kita (Sampling 3 Patch Sisi Atas):**
   Daripada mematok angka warna latar secara kaku (*hardcoded* untuk satu warna tembok saja), fungsi `background_mask()` secara otomatis mengambil **3 kotak sampel (*patch* berukuran $30 \times 30$ piksel)** di bagian atas foto (pojok kiri atas, tengah atas, dan pojok kanan atas — area yang pada pasfoto formal pasti berisi latar belakang).
   * Nilai **median** dari ketiga *patch* tersebut pada ruang warna **LAB** (atau **HSV**) dijadikan warna referensi latar $\mathbf{c}_{\text{ref}} = (L_{\text{ref}}, a_{\text{ref}}, b_{\text{ref}})$.
   * Setiap piksel yang selisih warnanya berada di dalam rentang toleransi $[\mathbf{c}_{\text{ref}} - \text{tol}, \mathbf{c}_{\text{ref}} + \text{tol}]$ ditandai sebagai kandidat latar belakang (`255`).

#### B. Cabang 2 — Konvolusi Spasial Deteksi Tepi Sobel sebagai *Edge Barrier* (`sobel_edges`)
1. **Kaitan dengan Materi Kuliah (Materi 05 Slide 12–13, Materi 07 Slide 15–38, Materi 10 Slide 38–49):**
   * **Langkah 1 — Konversi ke Grayscale Luminance (Materi 05 Slide 12–13):**
     Sebelum dikonvolusi dengan kernel Sobel, citra berwarna BGR dikonversi menjadi satu kanal keabuan (*luminance* $Y$) menggunakan persamaan berbobot yang sesuai dengan sensitivitas mata manusia (Materi 05 Slide 13: *"Lebih bagus daripada rata-rata $(R+G+B)/3$"*):
     $$Y = 0,299R + 0,587G + 0,144B$$
   * **Langkah 2 — Konvolusi dengan Penapis Lolos-Tinggi / Gradien Turunan Pertama (Materi 07 & Materi 10 Slide 40–49):**
     Operator **Sobel** menggunakan dua buah kernel konvolusi $3 \times 3$ untuk menghitung pendekatan turunan pertama (gradien) intensitas dalam arah horizontal ($G_x$) dan vertikal ($G_y$):
     $$h_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad h_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
     Perhatikan bahwa jumlah seluruh elemen di dalam kernel $h_x$ maupun $h_y$ adalah **sama dengan 0** ($\sum h = 0$). Sesuai **Materi 07 Slide 20 & Materi 10 Slide 41 & 47**: *"Jika jumlah bobot = 0, maka area dengan intensitas konstan (frekuensi rendah / tembok polos) akan bernilai 0, sedangkan area dengan perubahan intensitas tajam (frekuensi tinggi / tepi bahu dan rambut) akan bernilai besar."*
   * **Langkah 3 — Penanganan Piksel Pinggir / *Border Effect* (Materi 07 Slide 29–32):**
     Pada saat mengonvolusi piksel di tepi citra, OpenCV menggunakan duplikasi/refleksi elemen pinggir (solusi ke-3 pada Materi 07 Slide 32) sehingga tidak muncul garis tepi palsu di sekeliling bingkai foto.
   * **Langkah 4 — Magnitudo Gradien & Pengambangan Tepi:**
     Magnitudo gradien dihitung dengan $M(x, y) = \sqrt{G_x^2 + G_y^2}$, lalu di-ambangkan dengan `sobel_thr` dan ditebalkan sedikit (*dilasi* $3 \times 3$) agar membentuk garis pembatas (*barrier*) yang rapat dan tidak mudah bocor.

#### C. Implementasi pada Kode Kita
```python
def sobel_edges(img, p):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # Y = 0.299R + 0.587G + 0.144B (Materi 05 Slide 13)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)      # Konvolusi arah horizontal
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)      # Konvolusi arah vertikal
    mag = cv2.magnitude(gx, gy) / 4.0                 # Magnitudo gradien
    e = (mag > p['sobel_thr']).astype(np.uint8) * 255 # Thresholding tepi
    k = np.ones((p['sobel_dilate'],) * 2, np.uint8)
    return cv2.dilate(e, k)                           # Dilasi tipis agar garis barrier menyambung
```

---

### Tahap 3: Pascapemrosesan — Operasi Logika Boolean, Morfologi, Feathering, dan Aritmetika Alpha Blending

Tahap ini menyatukan hasil *Color Thresholding* dan *Sobel Barrier*, membersihkan cacat masker, serta mengganti latar belakang dengan warna baru.

#### A. Operasi Logika Boolean (`AND`, `OR`, `NOT`) untuk Menggabungkan Warna & Sobel (Materi 05 Slide 37–43 & Materi 09 Slide 49–50)
1. **Kaitan dengan Materi Kuliah:**
   Pada **Materi 05 Slide 37–42**, dijelaskan tiga operator logika Boolean piksel-per-piksel pada citra biner:
   * `NOT`: Menghasilkan komplemen (kebalikan) dari citra biner ($C(x,y) = \sim A(x,y)$).
   * `AND`: Mengambil irisan dua citra biner ($C(x,y) = A(x,y) \land B(x,y)$).
   * `OR`: Mengambil gabungan dua citra biner ($C(x,y) = A(x,y) \lor B(x,y)$).
2. **Cara Kerja di Pipeline Kita (`run_pipeline` & `border_connected`):**
   * Pertama, masker kandidat latar (`bg`) dipotong dengan komplemen dari garis tepi Sobel (`barrier`) menggunakan operasi **`AND` dan `NOT`**:
     $$\text{BG\_Terbatas} = \text{BG} \land (\neg \text{Barrier})$$
     Artinya: piksel yang warnanya mirip latar **tidak boleh melewati dinding garis tepi Sobel**!
   * Kedua, fungsi `border_connected()` hanya mengambil area putih yang menyambung ke tepi luar bingkai foto (seperti *flood-fill* dari pinggir foto). Jika ada bagian kemeja di tengah badan yang warnanya kebetulan mirip tembok, bagian itu tidak akan ikut terhapus karena sudah terhalang oleh dinding *Sobel Barrier* di garis bahu/kerah!
   * Ketiga, masker latar belakang yang sudah valid dibalik menggunakan operasi komplemen **`NOT`** (`fg = cv2.bitwise_not(bg)`) untuk menghasilkan masker objek manusia (*Foreground*).

#### B. Pembersihan Morfologi, Aras Objek, dan Pengisian Lubang (`clean_foreground`)
1. **Operasi Morfologi (*Closing* & *Opening*):**
   Menggunakan elemen penstruktur (*structuring element*) berbentuk elips $7 \times 7$:
   * **Closing** (Dilasi $\rightarrow$ Erosi) menutup celah/lubang kecil di perbatasan rambut dan bahu.
   * **Opening** (Erosi $\rightarrow$ Dilasi) menghapus bintik-bintik putih terisolasi (*noise*) di area latar belakang.
2. **Operasi Aras Objek — Seleksi Komponen Terbesar (Materi 05 Slide 17):**
   Sesuai **Materi 05 Slide 17**, operasi **Aras Objek** bekerja pada karakteristik objek (seperti luas area). Dengan `connectedComponentsWithStats()`, sistem menghitung luas setiap objek putih dan hanya mempertahankan **1 komponen dengan luas terbesar** (yaitu tubuh mahasiswa), sehingga bercak gangguan di latar otomatis terbuang.
3. **Pengisian Lubang Internal (*Hole Filling*) dengan Operasi Boolean `NOT`, `AND`, `OR`:**
   Jika di tengah badan masih ada lubang hitam (misalnya pantulan kacamata atau motif baju), lubang tersebut dideteksi sebagai area hitam (`inv = NOT fg`) yang **tidak** menyentuh tepi luar citra (`holes = inv AND (NOT border_connected(inv))`), lalu ditambal ke masker badan menggunakan operasi **`OR`** (`fg = fg OR holes`).

#### C. Penghalusan Tepi (*Gaussian Edge Feathering*) & Operasi Aritmetika *Alpha Blending* (`replace_bg`)
1. **Penghalusan Tepi (*Feathering* — Materi 10 Slide 25–30):**
   Masker biner $0/255$ memiliki tepi yang tajam dan berundak (*aliasing / staircase effect*), yang terlihat tidak alami pada helaian rambut. Dengan menerapkan **Gaussian Smoothing ($9 \times 9$)** pada masker biner yang telah dinormalisasi ke $[0, 1]$ (Materi 05 Slide 35: *pembagian citra dengan skalar 255 untuk normalisasi*), diperoleh matriks transparansi $\alpha(x, y) \in [0,0 \dots 1,0]$ yang bernilai $1,0$ di bagian dalam tubuh, $0,0$ di bagian latar, dan bertransisi lembut ($0 < \alpha < 1$) tepat di perbatasan rambut dan bahu.
2. **Komposisi Latar dengan Operasi Aritmetika Perkalian & Penjumlahan Citra (Materi 05 Slide 19–31 & Materi 09 Slide 49–60):**
   * Pada **Materi 05 Slide 30–31**, dijelaskan bahwa **Perkalian Citra** ($C(x,y) = A(x,y) \cdot B(x,y)$) digunakan untuk *image masking* (memilih area objek).
   * Pada **Materi 05 Slide 20–24**, dijelaskan bahwa **Penjumlahan Citra** ($C(x,y) = A(x,y) + B(x,y)$) digunakan untuk menggabungkan komponen dari satu citra ke citra lain.
   * Fungsi `replace_bg()` mengimplementasikan kedua operasi aritmetika tersebut secara langsung dalam persamaan *Alpha Blending*:
     $$I_{\text{akhir}}(x, y) = \underbrace{\alpha(x, y) \cdot I_{\text{asli}}(x, y)}_{\text{Perkalian Citra (Objek)}} + \underbrace{(1 - \alpha(x, y)) \cdot I_{\text{latar\_baru}}(x, y)}_{\text{Perkalian Citra (Latar Merah/Biru)}}$$

---

## 4. Rancangan Evaluasi & Skenario Ablasi (Yang Akan Dijalankan di Minggu 13)

Sesuai Ketentuan Umum Poin 4 dan 5 pada Panduan Tugas Besar PCD, sistem dirancang dengan **4 bendera (*flags*) modular** sehingga pengujian **Baseline** dan **Studi Ablasi** dapat dijalankan pada data dan prosedur *tuning* yang sama persis:

| Skenario | `use_preprocess` (CLAHE + Gaussian) | `use_sobel` (Konvolusi Sobel) | `use_logic_morph` (Logika + Morfologi) | `use_feathering` (Gaussian Edge Blur) | Tujuan Analisis Kontribusi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline Klasik** | ❌ | ❌ | ❌ | ❌ | Menunjukkan kinerja dasar pengambangan warna (*color thresholding*) tunggal tanpa bantuan pra-pemrosesan, Sobel, maupun morfologi. |
| **2. Ablasi 1 (Tanpa Sobel)** | ✅ | ❌ | ✅ | ❌ | Mengukur seberapa besar penurunan IoU (kebocoran latar ke baju/bahu) ketika pembatas konvolusi deteksi tepi Sobel dihilangkan. |
| **3. Ablasi 2 (Tanpa Morfologi & Feathering)** | ✅ | ✅ | ❌ | ❌ | Mengukur dampak hilangnya pembersihan morfologi/*hole-filling* dan penghalusan tepi terhadap lubang internal dan *noise* latar. |
| **4. Metode Usulan (*Proposed*)** | ✅ | ✅ | ✅ | ✅ | Pipeline lengkap untuk mencapai target $\text{IoU} \ge 0,80$ (atau $\ge 15\%$ di atas *baseline*) dan waktu proses $< 2\text{ detik/citra}$. |

---

## 5. Daftar Pertanyaan Asistensi yang Sering Ditanyakan Dosen & Jawabannya

1. **"Mengapa jumlah dataset kalian diubah dari proposal awal menjadi 50 citra?"**
   * **Jawaban:** Karena *pipeline* kami murni menggunakan metode pengolahan citra klasik (tanpa melatih bobot model *Deep Learning* yang membutuhkan ratusan gambar). Pada metode klasik, dataset dibagi menjadi **40 citra *Tuning Set*** (untuk *grid search* parameter toleransi warna `tol_l`, `tol_ab`, dan ambang gradien `sobel_thr`) serta **10 citra *Testing Set***. Dengan 50 citra primer mahasiswa ITERA (5 subjek $\times$ 10 variasi kondisi *half-body/close-shot*, hijab, kacamata, *indoor/outdoor*), tim *Data Engineer & Annotator* dapat membuat *ground truth binary mask* manual yang jauh lebih presisi pada area rumit seperti rambut dan bahu. Perubahan ini juga sudah kami dokumentasikan untuk subbab **4.4 Perbedaan dengan Proposal** pada Laporan Akhir.
2. **"Apa hubungan antara deteksi tepi Sobel dan Color Thresholding di metode kalian? Kenapa tidak pakai salah satu saja?"**
   * **Jawaban:** Kalau hanya *Color Thresholding* (Aras Titik), bagian pakaian atau bahu yang warnanya mirip tembok akan ikut terhapus sebagai latar. Kalau hanya *Sobel* (Aras Lokal), hasilnya hanya berupa garis tepi yang sering tidak tertutup rapat sehingga tidak bisa langsung dijadikan masker area tubuh. Kami menggabungkannya dengan **Operasi Logika Boolean (`AND` dan `NOT`)**: garis tepi Sobel dijadikan dinding penahan (*barrier*), lalu kami hanya mengambil area hasil *color thresholding* yang terhubung ke pinggir foto dan tidak menembus garis Sobel (`bg AND NOT barrier`).
3. **"Mengapa saat pra-pemrosesan kalian memakai CLAHE di kanal L, bukan Histogram Equalization biasa di RGB?"**
   * **Jawaban:** Sesuai **Materi 09 Slide 26–27 dan Slide 47**, *Global Histogram Equalization* meratakan histogram seluruh citra sekaligus, sehingga pada foto pasfoto yang memiliki area tembok luas, kontras latar akan mengalami *over-amplification* (derau tembok makin parah dan wajah menjadi terlalu terang). Dengan **CLAHE** (perataan histogram lokal per blok $8 \times 8$ dengan batas *clip limit* $2,0$) yang hanya dikenakan pada kanal luminansi ($L$) di ruang warna LAB, kontras lokal antara batas tubuh dan latar meningkat tajam tanpa mengubah warna asli (*hue/chrominance*) wajah dan pakaian.
4. **"Operasi dasar apa saja dari materi kuliah Minggu 5–10 yang ada di dalam kode kalian?"**
   * **Jawaban:** Lengkap mencakup keempat aras komputasi:
     * **Operasi Geometri:** *Center-cropping* rasio $3:4$ dan penskalaan (*image zooming/resizing*) ke $720 \times 960$.
     * **Aras Titik:** Konversi *Grayscale Luminance* ($Y = 0,299R + 0,587G + 0,144B$), pengambangan (*thresholding*), normalisasi pembagian skalar ($/255$), operasi logika Boolean (`AND`, `OR`, `NOT`), serta operasi aritmetika perkalian dan penjumlahan matriks citra pada *Alpha Blending*.
     * **Aras Lokal:** Konvolusi spasial 2D untuk pelembutan citra (*Gaussian Low-Pass Filter* $5 \times 5$ dan *Feathering* $9 \times 9$), konvolusi penapis lolos-tinggi gradien turunan pertama (**Sobel** $3 \times 3$), perataan histogram lokal (**CLAHE**), serta operasi morfologi (*Closing*, *Opening*, *Dilasi*).
     * **Aras Objek:** Analisis komponen terhubung (*Connected Components*) untuk memilih objek dengan luas area terbesar (tubuh manusia).

