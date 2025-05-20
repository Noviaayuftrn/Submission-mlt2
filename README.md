# Laporan Proyek Machine Learning - Sistem Rekomendasi Film - Novia Ayu Fitriana

## Project Overview

Sistem rekomendasi telah menjadi bagian integral dari banyak platform digital, terutama dalam industri hiburan seperti film. Dengan jumlah film yang terus bertambah, pengguna seringkali kesulitan memilih tontonan yang sesuai dengan preferensi mereka. Oleh karena itu, proyek ini bertujuan untuk membangun sistem rekomendasi film berbasis machine learning yang dapat memberikan saran film yang relevan kepada pengguna.

Proyek ini menggunakan dataset **MovieLens 20M**, salah satu dataset film paling populer dari GroupLens, yang berisi lebih dari 20 juta rating dari pengguna terhadap berbagai film. Sistem ini dibangun menggunakan dua pendekatan utama: **Content-Based Filtering** dan **Collaborative Filtering**, yang masing-masing memiliki keunggulan tersendiri.

---

## Business Understanding

### Problem Statements
1. Bagaimana cara merekomendasikan film yang relevan berdasarkan preferensi pengguna?
2. Bagaimana sistem dapat memberikan saran film meskipun pengguna belum memberikan banyak penilaian?

### Goals
1. Membangun sistem rekomendasi film berbasis genre (Content-Based).
2. Membangun sistem rekomendasi berdasarkan pola rating pengguna lain (Collaborative Filtering).
3. Menyajikan top-N rekomendasi untuk masing-masing pendekatan.

### Solution Approach
Untuk mencapai tujuan di atas, dua pendekatan digunakan:
- **Content-Based Filtering**: Menggunakan informasi konten (genre) untuk menghitung kemiripan antar film dengan teknik TF-IDF dan cosine similarity.
- **Collaborative Filtering**: Menggunakan algoritma *Singular Value Decomposition (SVD)* untuk memprediksi rating berdasarkan interaksi pengguna dengan film.

---

## Data Understanding

Dataset yang digunakan adalah **MovieLens 20M** yang dapat diunduh di [Kaggle: MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).  
File yang digunakan:
- `movies.csv` — Informasi film (`movieId`, `title`, `genres`)
- `ratings.csv` — Interaksi pengguna terhadap film (`userId`, `movieId`, `rating`, `timestamp`)

### Jumlah Data:
- Film: 27,000+
- Interaksi rating yang diproses: 500,000 sampel (subset untuk efisiensi)

### Variabel:
- `movieId`: ID unik untuk film
- `title`: Judul film
- `genres`: Daftar genre film yang dipisahkan dengan "|"
- `userId`: ID unik pengguna
- `rating`: Nilai rating dari pengguna (0.5 - 5.0)
- `timestamp`: Waktu rating diberikan

### Exploratory Data Analysis (EDA)
- **Distribusi genre**
  
  Dari analisis kolom genre pada dataset, ditemukan bahwa genre yang paling dominan dalam koleksi film adalah Drama dan Comedy. Genre Drama menempati urutan teratas, diikuti oleh Comedy, Thriller, dan Action. Hal ini menunjukkan bahwa film dengan tema kehidupan, emosi, dan konflik personal (drama) serta hiburan ringan (komedi) cenderung diproduksi dan ditonton lebih banyak.

  Distribusi genre ini juga menjadi dasar yang penting bagi pendekatan Content-Based Filtering, di mana sistem akan merekomendasikan film berdasarkan kemiripan genre. Sebagai contoh, jika seorang pengguna menyukai film drama, maka sistem cenderung merekomendasikan film drama lainnya dengan rating tinggi atau yang memiliki genre gabungan dengan drama.

  Genre pada dataset bersifat multikategori, artinya satu film bisa memiliki lebih dari satu genre (misalnya Action|Adventure|Sci-Fi). Oleh karena itu, dilakukan teknik one-hot encoding atau tokenisasi genre untuk memungkinkan analisis frekuensi dan kemiripan antar film berdasarkan genre secara lebih akurat.
  
- **Distribusi rating**

  Distribusi rating menunjukkan bahwa mayoritas pengguna memberikan penilaian yang cukup tinggi terhadap film yang mereka tonton. Rating 4.0 merupakan skor yang paling sering diberikan, diikuti oleh rating 3.0 dan 5.0. Pola ini mencerminkan kecenderungan pengguna hanya memberikan rating terhadap film yang mereka sukai atau anggap layak ditonton, sehingga menyebabkan bias positif dalam distribusi penilaian.

  Distribusi ini juga memperlihatkan bahwa rating rendah (di bawah 2.0) jarang diberikan, yang mungkin mengindikasikan bahwa pengguna cenderung tidak meluangkan waktu untuk menilai film yang mereka anggap buruk. Selain itu, rating disimpan dalam skala 0.5 hingga 5.0 dengan interval 0.5, yang memungkinkan analisis granular terhadap persepsi kualitas film.

  Informasi ini penting dalam pendekatan Collaborative Filtering, karena sistem akan menggunakan pola penilaian tersebut untuk menghitung kesamaan antar pengguna (user similarity) atau antar item (item similarity). Dengan mengetahui sebaran rating, kita juga dapat mengantisipasi masalah seperti sparsity dan cold start, terutama untuk film atau pengguna dengan jumlah rating yang minim.

---

## Data Preparation

Tahapan persiapan data yang dilakukan:

1. **Penggabungan Data**: `movies.csv` dan `ratings.csv` digabungkan berdasarkan `movieId`.
2. **Pembersihan Data**:
   - Menghapus kolom `timestamp`.
   - Menghapus duplikat jika ada.
   - Menghapus "(no genres listed)" untuk genre kosong.
3. **Ekstraksi Fitur (Content-Based)**:
   - Genre diubah menjadi vektor dengan `TfidfVectorizer` menggunakan token `|`.
   - Dihitung kemiripan antar film menggunakan `cosine_similarity`.
   - Dibuat indeks pencarian berdasarkan `title` film.

Alasan: Untuk mempersiapkan input yang sesuai bagi masing-masing model (berbasis konten dan berbasis pengguna).

---

## Modeling and Result

### 1. Content-Based Filtering
- Film direpresentasikan berdasarkan genre menggunakan TF-IDF.
- Rekomendasi dihitung berdasarkan kemiripan cosine antar vektor genre.
- Fungsi rekomendasi dibuat berdasarkan film yang diberikan
- Output: Top-10 film paling mirip
- Contoh output untuk "Toy Story (1995)":
  
  | Index | Title                                                               |
  |-------|----------------------------------------------------------------------|
  | 2209  | Antz (1998)                                                          |
  | 3027  | Toy Story 2 (1999)                                                  |
  | 3663  | Adventures of Rocky and Bullwinkle, The (2000)                      |
  | 3922  | Emperor's New Groove, The (2000)                                    |
  | 4790  | Monsters, Inc. (2001)                                               |
  | 10114 | DuckTales: The Movie - Treasure of the Lost Lamp (1990)            |
  | 10987 | Wild, The (2006)                                                    |
  | 11871 | Shrek the Third (2007)                                              |
  | 13337 | Tale of Despereaux, The (2008)                                      |
  | 18274 | Asterix and the Vikings (Astérix et les Vikings) (2006)            |



### 2. Collaborative Filtering

- Teknik: SVD (Singular Value Decomposition) dari library `Surprise`
- Dataset dibagi menjadi 80% train dan 20% test.
- Model dilatih menggunakan data rating pengguna.
- Input: matriks userId, movieId, rating
- Output: Prediksi rating pengguna terhadap film, kemudian diurutkan untuk mendapatkan Top-N recommendation

### Kelebihan dan Kekurangan:

| Pendekatan            | Kelebihan                                                  | Kekurangan                                                   |
|-----------------------|------------------------------------------------------------|---------------------------------------------------------------|
| Content-Based         | Tidak butuh interaksi pengguna, cocok untuk pengguna baru | Tidak bisa menangkap selera tersembunyi dari pengguna        |
| Collaborative         | Menangkap pola preferensi pengguna                        | Butuh banyak data interaksi, cold start untuk pengguna baru  |

---

## Evaluation

### Content-Based Filtering

- **Metrik**: Precision, Recall, F1-Score
- **Threshold Similarity**: ≥ 0.3 untuk dianggap relevan

**Hasil Evaluasi:**
- Average Precision: **0.9775**
- Average Recall: **0.0018**
- Average F1-Score: **0.0037**

> Insight: Precision tinggi, namun recall sangat rendah karena pendekatan ini tidak mempertimbangkan rating dari pengguna lain.

### Collaborative Filtering

- **Metrik**: RMSE (Root Mean Squared Error)** dan **MAE (Mean Absolute Error)
- **Hasil Evaluasi**:
  - RMSE: **0.8851**
  - MAE: **0.6953**

> Insight: Model cukup baik dalam memprediksi rating pengguna dengan error <1, semakin akurat prediksi rating terhadap data test.

### Hubungan dengan Business Understanding dan Problem Statement
- Sistem rekomendasi yang dikembangkan sudah menjawab problem utama, yaitu menyediakan rekomendasi film yang relevan dan personal bagi pengguna. Content-Based Filtering sangat efektif dalam memberikan rekomendasi yang tepat dari segi kemiripan konten, sedangkan Collaborative Filtering mampu memprediksi rating dengan akurasi yang memadai, membantu personalisasi.
- Goals utama yaitu menghasilkan rekomendasi yang relevan dengan preferensi pengguna dan prediksi rating yang akurat telah tercapai secara parsial. Content-Based Filtering unggul dalam precision, tetapi kurang dalam recall. Collaborative Filtering unggul dalam prediksi rating, memberikan dasar yang kuat untuk rekomendasi personalisasi.
- Kedua solusi memberikan dampak positif yang berbeda: Content-Based Filtering membantu pengguna menemukan film yang serupa dengan apa yang mereka sukai, sedangkan Collaborative Filtering meningkatkan akurasi prediksi rating dan memungkinkan eksplorasi film yang mungkin belum diketahui pengguna. Penggabungan kedua metode (hybrid) bisa menjadi solusi yang lebih lengkap untuk menjawab tantangan rekomendasi film secara menyeluruh.


### Penjelasan Formula:

- **RMSE** = √(Σ(yᵢ - ŷᵢ)² / n)  
  Mengukur deviasi rata-rata antara nilai prediksi dan aktual.
- **MAE** = Σ|yᵢ - ŷᵢ| / n  
  Mengukur rata-rata selisih absolut antara nilai aktual dan prediksi.
- **Precision/Recall/F1**:
  - Precision = TP / (TP + FP)
  - Recall = TP / (TP + FN)
  - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

### Penjelasan Metrik:
- **Precision**: Proporsi rekomendasi yang benar dari seluruh rekomendasi yang diberikan.
- **Recall**: Proporsi rekomendasi yang benar dari seluruh item relevan yang tersedia.
- **F1-Score**: Harmonic mean dari precision dan recall.
- **RMSE**: Mengukur seberapa jauh prediksi model dari nilai aktual dengan penalti lebih besar untuk kesalahan besar.
- **MAE**: Rata-rata kesalahan absolut antara prediksi dan data aktual.

---

## Kesimpulan

- Pendekatan **Collaborative Filtering (SVD)** menunjukkan performa yang lebih stabil dalam memberikan rekomendasi.
- Pendekatan **Content-Based Filtering** efektif untuk item cold-start (film baru), tetapi terbatas pada informasi konten.

---
