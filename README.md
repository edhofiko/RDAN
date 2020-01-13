#Klasifikasi Dokumen Berita menggunakan Jaringan Syaraf Tiruan CBOW-GRU
##Latar Belakang
Klasifikasi dokumen merupakan salah satu tugas yang dilakukan dalam ***Text Mining***.
Dokumen berita diklasifikasikan pada kanal berita tertentu seperti politik, olahraga, dll.
Banuel, dkk (2018) menunjukkan bahwa arsitektur jaringan syaraf tiruan ***HA-GRU*** dapat secara baik digunakan untuk melakukan klasifikasi dokumen.
Iyyer, dkk (2015) menunjukkan arsitektur jaringan syaraf tiruan ***DAN*** dapat digunakan untuk melakukan tugas klasifikasi dokumen dengan waktu yang cepat dengan hasil yang cukup baik. 
Penelitian ini berusaha menggabungkan dua arsitektur jaringan syarat tiruan yang telah disebut sebelumnya.

##Rumusan Masalah
1. Bagaimana mengimplementasikan JST CBOW-GRU dalam tugas klasifikasi dokumen berita?
2. Bagaimana hasil akurasi, presisi dan recall dari JST CBOW-GRU dalam tugas klasifikasi dokumen berita?

##Metode yang digunakan
fitur CBOW yang digunakan pada metode ini adalah sama dengan yang digunakan pada Iyyer, dkk (2015) untuk mengambil fitur setiap kalimat pada dokumen.
fitur dari beberapa kalimat akan membentuk dokumen yang kemudian akan menjadi fitur dari lapisan _Gated Recurrent unit_.
hasil dari lapisan _Gated Recurrent Unit_ akan menjadi masukan dari lapisan _Multi Layer Perceptron_ yang akan mengklasifikasikan dokumen.

#Daftar Pustaka
Banual, T., Nassour-Kassis, J., Cohen, R., Elhadad, M., & Elhadad, N. (2018). Multi-Label Classification of Patient Notes: Case Study on ICD Code Assignment. AAI Conference on Artificial Intelligence (pp. 409-416). Association for the Advencement of Artificial intelligence.
Iyyer, M., Manjunatha, V., Boyd-Graber, J., & Daume, H. (2015). Deep Unordered Composition Rivals Syntatic Methods for Text Classification. The 53rd Annual Meeting of the Association for Computational Linguistic (pp. 1681-1691). Beijing: Association for Computational Linguistic.
	
#Tugas Selesai
Data
Implementasi JST
Implementasi Word Embedding
Implementasi Model

#Tugas
Pelatihan Model
Pembuatan GUI