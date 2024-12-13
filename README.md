# Recipe Recommendation System

Sistem ini adalah aplikasi sederhana untuk merekomendasikan resep masakan berdasarkan bahan-bahan yang dimasukkan pengguna. Aplikasi ini menggunakan **TF-IDF Vectorizer** dan **Cosine Similarity** untuk menghitung kemiripan antara bahan yang dimasukkan dan dataset resep yang tersedia.

## Fitur
- Membersihkan dataset bahan masakan dari teks yang tidak relevan.
- Melatih model **TF-IDF Vectorizer** untuk merepresentasikan teks bahan masakan.
- Menyimpan model dan vektor hasil pelatihan dalam format file.
- Memberikan rekomendasi resep berdasarkan bahan yang dimasukkan pengguna.

---

## Persyaratan
Pastikan Anda telah menginstal dependensi berikut:
- Python 3.x
- Pandas
- Scikit-learn
- Joblib

Untuk menginstal semua dependensi, jalankan perintah berikut:
```bash
pip install pandas scikit-learn joblib
```

- Masukkan bahan-bahan dalam variabel input_ingredients dan dapatkan rekomendasi resep.
- Pastikan jalur file (E:/NLP NAZWA/NLP/) sesuai dengan lokasi file di komputer Anda.
- Anda dapat menyesuaikan parameter top_n untuk mengatur jumlah rekomendasi yang diinginkan.
