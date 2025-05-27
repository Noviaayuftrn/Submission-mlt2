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

Proyek ini menggunakan MovieLens 20M Dataset, yang berisi lebih dari 20 juta rating film dan aktivitas tagging sejak tahun 1995. Dataset ini sering digunakan sebagai benchmark dalam penelitian sistem rekomendasi.

### URL Sumber Data:
- [Kaggle: MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)

### Jumlah Data:
- movies.csv: 27,278 baris × 3 kolom
- ratings.csv: 20,000,263 baris × 4 kolom

Dalam proyek ini, ratings.csv dibatasi hingga 500,000 baris untuk efisiensi pemrosesan.

### Kondisi Data:
- **Missing Values:** Tidak ditemukan nilai yang hilang pada kedua dataset.
- **Duplikat:** Duplikat pada data gabungan (ratings dan movies) telah dihapus.
- **Outlier:** Tidak dilakukan penanganan outlier karena rating berada dalam rentang 0.5 hingga 5.0.

### Uraian Fitur
**Movies.csv**
- `movieId`: ID unik untuk setiap film.
- `title`: Judul film beserta tahun rilis.
- `genres`: Daftar genre yang dipisahkan oleh tanda '|'.

**ratings.csv**
- `userId`: ID unik untuk setiap  pengguna.
- `movieId`: ID film yang dirating.
- `rating`: Rating dari pengguna (0.5 hingga 5.0).
- `timestamps`: Waktu rating diberikan (dalam format UNIX timestamp).

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

**Handling Missing Values**
- Tidak ada nilai yang hilang pada kedua dataset, sehingga tidak diperlukan penanganan khusus.

**Handling Duplicates**
- Duplikat pada data gabungan telah dihapus menggunakan `drop_duplicates()`.

**Handling Outliers**
- Distribusi rating diperiksa menggunakan histogram. Tidak ditemukan outlier ekstrem yang memerlukan penanganan khusus karena rating berada dalam rentang 0.5 hingga 5.0.

**Menggabungkan data `ratings` dan `movies`**
- Untuk mendapatkan informasi lengkap tentang film dan rating yang diberikan oleh pengguna dalam satu dataframe. Ini diperlukan untuk analisis lanjutan dan pembuatan model.

```python
  tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
  tfidf_matrix = tfidf.fit_transform(movies['genres'])
  ```

**Menghapus kolom `timestamp`**
- Kolom `timestamp` tidak relevan dalam sistem rekomendasi ini karena tidak digunakan untuk pemodelan. Penghapusannya membuat data lebih bersih dan efisien.

```python
 data.drop(['timestamp'], axis=1, inplace=True)
```

**Membersihkan data tanpa genre**
- Film tanpa genre dapat memengaruhi hasil sistem rekomendasi berbasis konten karena genre digunakan sebagai fitur utama. Dengan mengosongkan atau menyesuaikannya, model bisa bekerja lebih baik.

```python
  movies['genres'] = movies['genres'].replace('(no genres listed)', '')
```

**Content-Based Filtering Preparation**
- Ekstraksi Fitur TF-IDF: Menggunakan TfidfVectorizer untuk mengubah kolom genres menjadi representasi numerik yang dapat digunakan untuk menghitung kesamaan antar film. token_pattern=r'[^|]+' digunakan untuk memisahkan genre yang dipisahkan oleh '|'.
  
  ```python
  tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
  tfidf_matrix = tfidf.fit_transform(movies['genres'])
  ```
  
- Mapping Judul ke Indeks: Membuat mapping dari judul film ke indeks untuk mempermudah pencarian film berdasarkan judul.

    ```python
  indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
  ```

**Collaborative Filtering Preparation**
- Encode Label: Menggunakan library Surprise untuk membangun model collaborative filtering. Reader digunakan untuk menentukan skala rating, dan Dataset.load_from_df untuk memuat data ke dalam format yang sesuai.

  ```python
  reader = Reader(rating_scale=(0.5, 5.0))
  data_cf = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
  ```

- Split Data: Membagi data menjadi training dan testing set dengan rasio 80:20 menggunakan `train_test_split()`.

  ```python
  trainset, testset = train_test_split(data_cf, test_size=0.2, random_state=42)
  ```
    
---

## Modeling and Result

### 1. Content-Based Filtering
Pendekatan Content-Based Filtering digunakan dalam proyek ini karena sangat cocok diterapkan ketika hanya tersedia informasi preferensi dari satu pengguna tanpa memerlukan data dari pengguna lain. Metode ini berfokus pada kemiripan antar item (dalam hal ini film) berdasarkan fitur-fitur seperti genre, deskripsi, dan metadata lainnya. Dalam implementasinya, digunakan pendekatan Cosine Similarity untuk mengukur tingkat kemiripan antar film berdasarkan representasi vektor fitur yang diperoleh melalui teknik seperti TF-IDF atau Count Vectorizer. Fungsi recommend(title) dibuat untuk menerima input berupa judul film dan mengembalikan top-10 film yang paling mirip berdasarkan nilai cosine similarity. Pendekatan ini dipilih karena sifatnya yang sederhana namun efektif dalam memberikan rekomendasi yang relevan secara konten, terutama jika pengguna sudah memiliki film favorit sebagai referensi awal. Selain itu, metode ini tidak membutuhkan data rating dari pengguna lain, sehingga sangat sesuai jika menghadapi masalah cold-start pada pengguna baru.

- Cosine Similarity: Kode ini membentuk dasar dari sistem content-based recommendation, dengan menghitung sejauh mana kemiripan antar film berdasarkan deskripsi atau metadata mereka yang sudah diubah menjadi representasi numerik dengan TF-IDF.

  ```python
  cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
  ```

- Fungsi `recommend()` digunakan untuk menghasilkan 10 film teratas yang mirip dengan film yang diberikan, berdasarkan kemiripan konten (genres). Fungsi ini mencari indeks film dari judul yang dimasukkan, menghitung skor kemiripan cosine terhadap seluruh film lain, mengurutkannya secara menurun, dan mengambil 10 film teratas (dikecualikan film itu sendiri). Output dari fungsi ini berupa daftar judul film yang paling relevan secara konten dengan input yang diberikan.
  
  ```python
  def recommend(title, cosine_sim=cosine_sim):
      idx = indices[title]
      sim_scores = list(enumerate(cosine_sim[idx]))
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      sim_scores = sim_scores[1:11]  
      movie_indices = [i[0] for i in sim_scores]
      return movies['title'].iloc[movie_indices]
  ```

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
Untuk pendekatan Collaborative Filtering, digunakan algoritma SVD (Singular Value Decomposition) yang tersedia dalam library Surprise. Pemilihan SVD didasarkan pada kemampuannya sebagai salah satu teknik Matrix Factorization yang kuat dalam mempelajari preferensi pengguna secara laten dari data rating historis. SVD dapat menangkap hubungan kompleks antara pengguna dan item, meskipun hubungan tersebut tidak secara eksplisit tersedia dalam data. Dalam implementasinya, data rating terlebih dahulu diformat menggunakan objek Dataset dari Surprise, dengan Reader untuk menentukan skala rating yang digunakan. Dataset kemudian dibagi menjadi dua bagian, yaitu train set (80%) dan test set (20%), untuk keperluan pelatihan dan evaluasi model. Selanjutnya, model SVD dilatih (fit) menggunakan trainset untuk menghasilkan prediksi rating yang dapat digunakan dalam sistem rekomendasi.
```python
# Format data untuk Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data_cf = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split data menjadi train dan test (80% train, 20% test)
trainset, testset = train_test_split(data_cf, test_size=0.2, random_state=42)

# Gunakan model SVD
model = SVD()

# Training model dengan data train
model.fit(trainset)
```

- Output: Prediksi rating pengguna terhadap film, kemudian diurutkan untuk mendapatkan Top-N recommendation

```python
from collections import defaultdict

# Buat prediksi pada data test
predictions = model.test(testset)

# Fungsi untuk mendapatkan top-N rekomendasi per user
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Urutkan setiap user berdasarkan rating prediksi tertinggi
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Ambil top-10 rekomendasi untuk setiap user
top_n = get_top_n(predictions, n=10)

# Contoh: tampilkan rekomendasi untuk userId tertentu (misalnya 822)
for uid, user_ratings in top_n.items():
    if uid == 822:
        print(f"Rekomendasi untuk user {uid}:")
        for i, (movie_id, rating) in enumerate(user_ratings, 1):
            print(f"{i}. MovieID: {movie_id}, Predicted Rating: {rating:.2f}")
        break
```
- Setelah model SVD dilatih, dilakukan prediksi terhadap data uji menggunakan metode `.test()`. Selanjutnya, dilakukan proses pemeringkatan hasil prediksi dengan membuat fungsi `get_top_n()` untuk mengambil 10 film dengan prediksi rating tertinggi untuk setiap pengguna. Fungsi ini menyimpan hasil prediksi ke dalam struktur data dictionary berdasarkan `userId`, kemudian diurutkan berdasarkan nilai prediksi tertinggi.

  Sebagai contoh, ditampilkan 10 film dengan nilai prediksi tertinggi untuk pengguna dengan `userId = 822`. Hasil ini mencerminkan film-film yang kemungkinan besar akan disukai oleh pengguna tersebut berdasarkan pola preferensi yang telah dipelajari oleh model.

- Output rekomendasi untuk user 822:

  | No | MovieID | Predicted Rating |
  |----|---------|------------------|
  | 1  | 7153    | 4.46             |
  | 2  | 2959    | 4.30             |
  | 3  | 260     | 4.20             |
  | 4  | 1291    | 4.08             |
  | 5  | 2762    | 3.96             |
  | 6  | 1136    | 3.94             |
  | 7  | 40815   | 3.90             |
  | 8  | 4963    | 3.88             |
  | 9  | 4973    | 3.81             |
  | 10 | 8368    | 3.80             |

---
### Implementasi Top-N Recommendation
```python
from collections import defaultdict

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n
```
Fungsi ini akan:
- Mengelompokkan prediksi berdasarkan userId
- Menyortir prediksi berdasarkan nilai rating tertinggi
- Mengambil n film teratas (default: 10)

### Output Rekomendasi untuk dua user menampilkan MovieID, dan rating:
```python
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))

for uid, user_ratings in top_n.items():
    print(f"User {uid} Top-{len(user_ratings)} recommendations:")
    for movie_id, predicted_rating in user_ratings:
        title = movie_id_to_title.get(movie_id, "Unknown Title")
        print(f"  {title} (MovieID: {movie_id}) - Predicted Rating: {predicted_rating:.2f}")
    print("="*40)
    break  
```
Menampilkan 2 pengguna pertama dan menampilkan MovieID dan predicted rating saja.

#### User ID: 982

| No | MovieID | Predicted Rating |
|----|---------|------------------|
| 1  | 1136    | 4.28             |
| 2  | 3307    | 4.17             |
| 3  | 6016    | 4.14             |
| 4  | 1252    | 4.13             |
| 5  | 1089    | 4.13             |
| 6  | 1175    | 4.12             |
| 7  | 1230    | 4.09             |
| 8  | 2966    | 4.09             |
| 9  | 1212    | 4.07             |
| 10 | 6350    | 4.06             |

#### User ID: 515

| No | MovieID | Predicted Rating |
|----|---------|------------------|
| 1  | 47      | 4.66             |
| 2  | 150     | 3.93             |
| 3  | 292     | 3.84             |
| 4  | 288     | 3.78             |
| 5  | 588     | 3.69             |
| 6  | 316     | 3.65             |
| 7  | 587     | 3.60             |
| 8  | 597     | 3.57             |
| 9  | 500     | 3.38             |
| 10 | 586     | 3.36             |

---
### Output Rekomendasi untuk User 982 menampilkan Judul, MovieID, dan rating:
```python
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))

for uid, user_ratings in top_n.items():
    print(f"User {uid} Top-{len(user_ratings)} recommendations:")
    for movie_id, predicted_rating in user_ratings:
        title = movie_id_to_title.get(movie_id, "Unknown Title")
        print(f"  {title} (MovieID: {movie_id}) - Predicted Rating: {predicted_rating:.2f}")
    print("="*40)
    break  
```
Menampilkan 1 pengguna pertama saja (`break` langsung di luar loop) dan menampilkan judul film + MovieID + predicted rating

| No | Judul Film                                                                 | MovieID | Predicted Rating |
|----|----------------------------------------------------------------------------|---------|------------------|
| 1  | Monty Python and the Holy Grail (1975)                                     | 1136    | 4.28             |
| 2  | City Lights (1931)                                                         | 3307    | 4.17             |
| 3  | City of God (Cidade de Deus) (2002)                                        | 6016    | 4.14             |
| 4  | Chinatown (1974)                                                           | 1252    | 4.13             |
| 5  | Reservoir Dogs (1992)                                                      | 1089    | 4.13             |
| 6  | Delicatessen (1991)                                                        | 1175    | 4.12             |
| 7  | Annie Hall (1977)                                                          | 1230    | 4.09             |
| 8  | Straight Story, The (1999)                                                 | 2966    | 4.09             |
| 9  | Third Man, The (1949)                                                      | 1212    | 4.07             |
| 10 | Laputa: Castle in the Sky (Tenkû no shiro Rapyuta) (1986)                 | 6350    | 4.06             |


### Hasil
Hasil dari top-N recommendation membuktikan bahwa:
- Sistem berhasil memberikan 10 film rekomendasi terbaik untuk masing-masing pengguna.
- Rekomendasi bersifat personalisasi berdasarkan preferensi historis pengguna.
- Sistem siap digunakan untuk aplikasi nyata seperti layanan streaming film, dengan potensi meningkatkan pengalaman pengguna melalui rekomendasi yang relevan.

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
  - RMSE: **0.8553**
  - MAE: **0.6533**

> Insight: Model cukup baik dalam memprediksi rating pengguna dengan error <1, semakin akurat prediksi rating terhadap data test.
---
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
