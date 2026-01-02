import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

class NeuralNetwork:
    def __init__(self):
        """Neural Network untuk prediksi kelulusan"""
        # Arsitektur: 4 input -> 8 hidden -> 1 output
        np.random.seed(42)
        self.weights1 = np.random.randn(4, 8) * 0.5
        self.bias1 = np.zeros((1, 8))
        self.weights2 = np.random.randn(8, 1) * 0.5
        self.bias2 = np.zeros((1, 1))
        self.learning_rate = 0.01
        
        # History untuk visualisasi
        self.loss_history = []
        
    def sigmoid(self, x):
        """Fungsi aktivasi sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Turunan sigmoid"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation"""
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output
    
    def backward(self, X, y, output):
        """Backpropagation"""
        # Hitung error
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)
        
        # Update weights
        self.weights2 += self.hidden.T.dot(output_delta) * self.learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights1 += X.T.dot(hidden_delta) * self.learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X, y, epochs=5000, callback=None):
        """Training neural network"""
        self.loss_history = []
        
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Hitung loss (Mean Squared Error)
            loss = np.mean((y - output) ** 2)
            self.loss_history.append(loss)
            
            # Backward propagation
            self.backward(X, y, output)
            
            # Callback untuk update UI
            if callback and epoch % 100 == 0:
                callback(epoch, loss)
        
        return self.loss_history
    
    def predict(self, X):
        """Prediksi untuk data baru"""
        return self.forward(X)


class StudentGraduationPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Prediksi Kelulusan Mahasiswa - ANN")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        self.nn = NeuralNetwork()
        self.is_trained = False
        
        # Data training (contoh data mahasiswa)
        self.X_train, self.y_train = self.generate_training_data()
        
        self.create_widgets()
        
    def generate_training_data(self):
        """Generate data training mahasiswa"""
        # Format: [IPK (0-4), Kehadiran (0-100), SKS (0-144), Organisasi (0/1)]
        # Target: 1 = Lulus Tepat Waktu, 0 = Tidak Tepat Waktu
        
        data = [
            # IPK, Kehadiran, SKS, Org -> Lulus
            [3.8, 95, 140, 1, 1],
            [3.5, 90, 135, 1, 1],
            [3.7, 92, 138, 0, 1],
            [3.9, 97, 142, 1, 1],
            [3.6, 88, 130, 1, 1],
            [3.4, 85, 128, 0, 1],
            [3.3, 87, 125, 1, 1],
            [3.8, 94, 137, 1, 1],
            [3.5, 91, 133, 0, 1],
            [3.6, 89, 131, 1, 1],
            
            # IPK rendah tapi rajin
            [2.8, 95, 120, 1, 1],
            [2.9, 93, 122, 1, 1],
            [3.0, 90, 125, 0, 1],
            
            # IPK tinggi tapi jarang masuk
            [3.5, 70, 110, 0, 0],
            [3.4, 65, 105, 1, 0],
            [3.6, 68, 108, 0, 0],
            
            # Tidak lulus tepat waktu
            [2.3, 60, 90, 0, 0],
            [2.5, 65, 95, 0, 0],
            [2.1, 55, 85, 0, 0],
            [2.4, 62, 92, 0, 0],
            [2.6, 68, 98, 0, 0],
            [2.2, 58, 88, 0, 0],
            [2.7, 70, 100, 0, 0],
            [2.0, 50, 80, 0, 0],
            
            # Kasus sedang
            [3.0, 75, 110, 0, 0],
            [2.9, 72, 108, 1, 0],
            [3.1, 78, 115, 0, 1],
            [3.2, 80, 120, 1, 1],
            
            # Tambahan data
            [3.7, 93, 136, 1, 1],
            [2.8, 88, 118, 1, 1],
            [3.4, 86, 127, 0, 1],
            [2.5, 67, 96, 0, 0],
            [3.9, 96, 141, 1, 1],
            [2.3, 61, 91, 0, 0],
        ]
        
        data = np.array(data)
        X = data[:, :4]
        y = data[:, 4:5]
        
        # Normalisasi
        X[:, 0] = X[:, 0] / 4.0      # IPK (0-4)
        X[:, 1] = X[:, 1] / 100.0    # Kehadiran (0-100)
        X[:, 2] = X[:, 2] / 144.0    # SKS (0-144)
        # Organisasi sudah 0/1
        
        return X, y
    
    def create_widgets(self):
        """Buat UI components"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x')
        title_label = tk.Label(title_frame, text="🎓 PREDIKSI KELULUSAN MAHASISWA", 
                              font=('Arial', 18, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel (Input & Training)
        left_panel = tk.Frame(main_container, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Input Section
        input_frame = tk.LabelFrame(left_panel, text="📝 Input Data Mahasiswa", 
                                   font=('Arial', 12, 'bold'), bg='white', padx=15, pady=15)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        # IPK
        tk.Label(input_frame, text="IPK (0.00 - 4.00):", font=('Arial', 10), bg='white').grid(row=0, column=0, sticky='w', pady=5)
        self.ipk_entry = tk.Entry(input_frame, font=('Arial', 10), width=20)
        self.ipk_entry.grid(row=0, column=1, pady=5, padx=10)
        self.ipk_entry.insert(0, "3.5")
        
        # Kehadiran
        tk.Label(input_frame, text="Kehadiran (%):", font=('Arial', 10), bg='white').grid(row=1, column=0, sticky='w', pady=5)
        self.kehadiran_entry = tk.Entry(input_frame, font=('Arial', 10), width=20)
        self.kehadiran_entry.grid(row=1, column=1, pady=5, padx=10)
        self.kehadiran_entry.insert(0, "85")
        
        # SKS
        tk.Label(input_frame, text="SKS Lulus:", font=('Arial', 10), bg='white').grid(row=2, column=0, sticky='w', pady=5)
        self.sks_entry = tk.Entry(input_frame, font=('Arial', 10), width=20)
        self.sks_entry.grid(row=2, column=1, pady=5, padx=10)
        self.sks_entry.insert(0, "120")
        
        # Organisasi
        tk.Label(input_frame, text="Aktif Organisasi:", font=('Arial', 10), bg='white').grid(row=3, column=0, sticky='w', pady=5)
        self.org_var = tk.IntVar(value=1)
        org_frame = tk.Frame(input_frame, bg='white')
        org_frame.grid(row=3, column=1, pady=5, padx=10, sticky='w')
        tk.Radiobutton(org_frame, text="Ya", variable=self.org_var, value=1, 
                      font=('Arial', 10), bg='white').pack(side='left', padx=5)
        tk.Radiobutton(org_frame, text="Tidak", variable=self.org_var, value=0, 
                      font=('Arial', 10), bg='white').pack(side='left', padx=5)
        
        # Buttons
        button_frame = tk.Frame(left_panel, bg='white')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.train_button = tk.Button(button_frame, text="🎓 Train Model", 
                                     command=self.train_model, 
                                     font=('Arial', 11, 'bold'), 
                                     bg='#3498db', fg='white', 
                                     relief='raised', bd=3, cursor='hand2',
                                     width=15, height=2)
        self.train_button.pack(side='left', padx=5)
        
        self.predict_button = tk.Button(button_frame, text="🔮 Prediksi", 
                                       command=self.predict_graduation, 
                                       font=('Arial', 11, 'bold'), 
                                       bg='#2ecc71', fg='white', 
                                       relief='raised', bd=3, cursor='hand2',
                                       width=15, height=2)
        self.predict_button.pack(side='left', padx=5)
        
        # Training Status
        status_frame = tk.LabelFrame(left_panel, text="📊 Status Training", 
                                    font=('Arial', 12, 'bold'), bg='white', padx=15, pady=15)
        status_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.status_text = tk.Text(status_frame, height=6, font=('Courier', 9), 
                                  bg='#ecf0f1', relief='sunken', bd=2)
        self.status_text.pack(fill='both', expand=True)
        self.status_text.insert('1.0', "Model belum ditraining.\nKlik 'Train Model' untuk memulai.")
        self.status_text.config(state='disabled')
        
        # Result Section
        result_frame = tk.LabelFrame(left_panel, text="✨ Hasil Prediksi", 
                                    font=('Arial', 12, 'bold'), bg='white', padx=15, pady=15)
        result_frame.pack(fill='x', padx=10, pady=10)
        
        self.result_label = tk.Label(result_frame, text="Belum ada prediksi", 
                                     font=('Arial', 14), bg='white', fg='#7f8c8d')
        self.result_label.pack(pady=10)
        
        self.percentage_label = tk.Label(result_frame, text="", 
                                        font=('Arial', 24, 'bold'), bg='white')
        self.percentage_label.pack(pady=5)
        
        self.recommendation_label = tk.Label(result_frame, text="", 
                                            font=('Arial', 10), bg='white', 
                                            wraplength=350, justify='left')
        self.recommendation_label.pack(pady=10)
        
        # Right panel (Graph)
        right_panel = tk.Frame(main_container, bg='white', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        graph_frame = tk.LabelFrame(right_panel, text="📈 Training Progress", 
                                   font=('Arial', 12, 'bold'), bg='white', padx=10, pady=10)
        graph_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initial empty plot
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss Over Time')
        self.ax.grid(True, alpha=0.3)
        
        # Data Training Table
        table_frame = tk.LabelFrame(right_panel, text="📋 Data Training", 
                                   font=('Arial', 12, 'bold'), bg='white', padx=10, pady=10)
        table_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Treeview untuk tabel
        columns = ('IPK', 'Kehadiran', 'SKS', 'Org', 'Status')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        # Define headings
        self.tree.heading('IPK', text='IPK')
        self.tree.heading('Kehadiran', text='Kehadiran (%)')
        self.tree.heading('SKS', text='SKS')
        self.tree.heading('Org', text='Organisasi')
        self.tree.heading('Status', text='Status')
        
        # Define column widths
        self.tree.column('IPK', width=60)
        self.tree.column('Kehadiran', width=100)
        self.tree.column('SKS', width=60)
        self.tree.column('Org', width=80)
        self.tree.column('Status', width=120)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Populate table
        self.populate_training_data()
    
    def populate_training_data(self):
        """Isi tabel dengan data training"""
        for i in range(len(self.X_train)):
            ipk = self.X_train[i][0] * 4.0
            kehadiran = self.X_train[i][1] * 100
            sks = self.X_train[i][2] * 144
            org = "Ya" if self.X_train[i][3] == 1 else "Tidak"
            status = "✓ Lulus" if self.y_train[i][0] == 1 else "✗ Tidak"
            
            self.tree.insert('', 'end', values=(
                f"{ipk:.2f}", 
                f"{kehadiran:.0f}", 
                f"{sks:.0f}", 
                org, 
                status
            ))
    
    def train_model(self):
        """Training neural network"""
        self.train_button.config(state='disabled', text='Training...')
        self.root.update()
        
        # Clear previous plot
        self.ax.clear()
        
        def update_callback(epoch, loss):
            self.status_text.config(state='normal')
            self.status_text.delete('1.0', 'end')
            self.status_text.insert('1.0', 
                f"Training Progress:\n"
                f"Epoch: {epoch + 1}/5000\n"
                f"Loss: {loss:.6f}\n"
                f"Status: {'⚡ Training...' if epoch < 4900 else '✅ Complete!'}"
            )
            self.status_text.config(state='disabled')
            self.root.update()
        
        # Train
        self.nn.train(self.X_train, self.y_train, epochs=5000, callback=update_callback)
        self.is_trained = True
        
        # Plot loss
        self.ax.plot(self.nn.loss_history, color='#3498db', linewidth=2)
        self.ax.set_xlabel('Epoch', fontsize=10)
        self.ax.set_ylabel('Loss (MSE)', fontsize=10)
        self.ax.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        
        # Calculate accuracy
        predictions = self.nn.predict(self.X_train)
        predictions_binary = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions_binary == self.y_train) * 100
        
        self.status_text.config(state='normal')
        self.status_text.delete('1.0', 'end')
        self.status_text.insert('1.0', 
            f"✅ Training Selesai!\n"
            f"Total Epochs: 5000\n"
            f"Final Loss: {self.nn.loss_history[-1]:.6f}\n"
            f"Accuracy: {accuracy:.1f}%\n"
            f"Status: Model siap digunakan"
        )
        self.status_text.config(state='disabled')
        
        self.train_button.config(state='normal', text='🎓 Train Model')
        messagebox.showinfo("Training Complete", 
                           f"Model berhasil ditraining!\n\n"
                           f"Accuracy: {accuracy:.1f}%\n"
                           f"Final Loss: {self.nn.loss_history[-1]:.6f}")
    
    def predict_graduation(self):
        """Prediksi kelulusan mahasiswa"""
        if not self.is_trained:
            messagebox.showwarning("Warning", "Model belum ditraining!\nKlik 'Train Model' terlebih dahulu.")
            return
        
        try:
            # Ambil input
            ipk = float(self.ipk_entry.get())
            kehadiran = float(self.kehadiran_entry.get())
            sks = float(self.sks_entry.get())
            org = self.org_var.get()
            
            # Validasi
            if not (0 <= ipk <= 4):
                raise ValueError("IPK harus antara 0-4")
            if not (0 <= kehadiran <= 100):
                raise ValueError("Kehadiran harus antara 0-100")
            if not (0 <= sks <= 144):
                raise ValueError("SKS harus antara 0-144")
            
            # Normalisasi
            X_test = np.array([[ipk/4.0, kehadiran/100.0, sks/144.0, org]])
            
            # Prediksi
            prediction = self.nn.predict(X_test)[0][0]
            percentage = prediction * 100
            
            # Update UI
            if prediction >= 0.7:
                status = "✅ LULUS TEPAT WAKTU"
                color = '#2ecc71'
                recommendation = (
                    "Peluang lulus tepat waktu sangat tinggi!\n\n"
                    "Rekomendasi:\n"
                    "• Pertahankan performa akademik\n"
                    "• Tetap jaga kehadiran\n"
                    "• Fokus selesaikan SKS tersisa"
                )
            elif prediction >= 0.5:
                status = "⚠️ BORDERLINE"
                color = '#f39c12'
                recommendation = (
                    "Peluang lulus tepat waktu cukup baik, tapi perlu perhatian.\n\n"
                    "Rekomendasi:\n"
                    "• Tingkatkan IPK jika memungkinkan\n"
                    "• Jaga kehadiran minimal 85%\n"
                    "• Ambil SKS sesuai kemampuan\n"
                    "• Konsultasi dengan dosen pembimbing"
                )
            else:
                status = "❌ RISIKO TIDAK TEPAT WAKTU"
                color = '#e74c3c'
                recommendation = (
                    "Peluang lulus tepat waktu rendah. Perlu tindakan segera!\n\n"
                    "Rekomendasi:\n"
                    "• Fokus tingkatkan IPK\n"
                    "• Hadiri semua kuliah (target 90%+)\n"
                    "• Ambil SKS lebih banyak jika mampu\n"
                    "• Konsultasi intensif dengan pembimbing akademik\n"
                    "• Kurangi aktivitas non-akademik sementara"
                )
            
            self.result_label.config(text=status, fg=color)
            self.percentage_label.config(text=f"{percentage:.1f}%", fg=color)
            self.recommendation_label.config(text=recommendation, fg='#2c3e50')
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Error: {str(e)}")


def main():
    root = tk.Tk()
    app = StudentGraduationPredictor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
