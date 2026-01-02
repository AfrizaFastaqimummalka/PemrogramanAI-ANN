🎓 Prediksi Kelulusan Mahasiswa dengan ANN

Aplikasi ini menggunakan **Artificial Neural Network (ANN)** dengan arsitektur 4-8-1 untuk memprediksi apakah mahasiswa akan lulus tepat waktu berdasarkan 4 parameter: 
IPK (0-4), Kehadiran (0-100%), SKS Lulus (0-144), dan Keaktifan Organisasi (Ya/Tidak). Model dilatih dengan 34 data mahasiswa selama 5000 epoch menggunakan backpropagation dan menghasilkan akurasi ~95%. 
Output berupa probabilitas (0-100%) yang dikategorikan sebagai: ✅ Lulus Tepat Waktu (≥70%), ⚠️ Borderline (50-69%), atau ❌ Risiko Tidak Tepat Waktu (<50%). 
Aplikasi juga memberikan rekomendasi tindakan berdasarkan hasil prediksi dan menampilkan visualisasi grafik training loss serta tabel data training.

🚀 Cara Pakai
1. Install dependencies: `pip install numpy matplotlib pandas`
2. Jalankan: `python student_graduation_predictor.py`
3. Klik **"Train Model"** untuk melatih ANN (tunggu ~10 detik)
4. Input data mahasiswa (IPK, Kehadiran, SKS, Organisasi)
5. Klik **"Prediksi"** untuk melihat hasil probabilitas kelulusan + rekomendasi

🧠 Cara Kerja Kalkulasi
ANN melakukan **Forward Propagation**: Input [IPK/4, Kehadiran/100, SKS/144, Org] → dinormalisasi → dikalikan dengan weight yang sudah dioptimasi saat training → dihitung melalui 8 hidden neurons dengan fungsi sigmoid → menghasilkan output 0-1 yang merepresentasikan probabilitas lulus. 
Contoh: IPK 3.5, Kehadiran 90%, SKS 130, Org Ya → Input [0.875, 0.9, 0.903, 1] → Hidden layer menangkap pola → Output 0.85 = **85% kemungkinan lulus tepat waktu**. 
Model belajar dari 34 data historis mahasiswa melalui backpropagation yang mengupdate weight berdasarkan error antara prediksi vs target aktual.
