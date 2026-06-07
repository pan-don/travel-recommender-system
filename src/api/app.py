from flask import Flask, render_template, request
from src.core.query_engine import recommendation_system

app = Flask(__name__, static_folder='../web/static', template_folder='../web/templates')


@app.route('/')
def index():
    """
    Fungsi untuk menampilkan halaman utama (index) kepada pengguna.

    Returns
    -------
    render_template : str
        Mengembalikan file HTML 'index.html' sebagai tampilan awal.
    """
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    """
    Fungsi untuk memproses data input dari form HTML dan menghasilkan hasil rekomendasi.

    Alur Kerja:
    -----------
    - Mengambil input dari form: kota, kategori, dan deskripsi destinasi.
    - Memanggil fungsi recommendation_system() untuk mendapatkan hasil rekomendasi.
    - Mengonversi hasil DataFrame menjadi list of dictionaries.
    - Menampilkan hasil dalam template 'result.html'.

    Input (POST Form):
    ------------------
    city : str
        Kota yang dipilih oleh pengguna.
    category : str
        Kategori wisata yang dipilih pengguna.
    description : str
        Deskripsi destinasi yang diinginkan pengguna.

    Returns
    -------
    render_template : str
        Mengembalikan file HTML 'result.html' beserta hasil rekomendasi.
    """
    if request.method == 'GET':
        # Redirect to the home page or show an error message for GET requests
        return render_template('index.html', error="Please submit the form to get recommendations.")

    # Ambil data dari form HTML
    city = str(request.form['city'])          # Nama kota dari input user
    catg = str(request.form['category'])      # Kategori destinasi dari input user
    desc = request.form['description']        # Deskripsi destinasi dari input user

    # Proses rekomendasi berdasarkan input pengguna
    results = recommendation_system(
        description=desc, 
        category=catg, 
        city=city)
    
    # Cek apakah hasil rekomendasi kosong
    if results is None:
        return render_template('result.html', results=False)
    
    # Konversi hasil DataFrame menjadi list of dict agar mudah digunakan di HTML
    results_dict = results.to_dict(orient='records')
    # Tampilkan halaman hasil dengan data rekomendasi
    return render_template('result.html', results=results_dict)
