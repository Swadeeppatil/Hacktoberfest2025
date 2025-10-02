import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import sqlite3
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import os

class CropHealthScanner:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Crop Health Scanner")
        self.root.geometry("1200x800")
        
        # Import reportlab for PDF generation
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # Initialize other components
        self.init_database()
        self.load_disease_database()
        self.model = self.load_model()
        self.setup_gui()

    def generate_report(self):
        scan_id = self.report_id_entry.get()
        if not scan_id:
            messagebox.showerror("Error", "Please enter a scan ID")
            return
            
        c = self.conn.cursor()
        c.execute("""SELECT * FROM scans WHERE id = ?""", (scan_id,))
        scan_data = c.fetchone()
        
        if not scan_data:
            messagebox.showerror("Error", "Scan ID not found")
            return
        
        # Convert bytes to strings if necessary
        scan_id = str(scan_data[0])
        crop_type = scan_data[1].decode() if isinstance(scan_data[1], bytes) else scan_data[1]
        condition = scan_data[2].decode() if isinstance(scan_data[2], bytes) else scan_data[2]
        confidence = float(scan_data[3])
        timestamp = scan_data[4]
        image_path = scan_data[5].decode() if isinstance(scan_data[5], bytes) else scan_data[5]
        
        # Create PDF
        report_path = f"scan_report_{scan_id}.pdf"
        c = canvas.Canvas(report_path, pagesize=letter)
        
        # Set up initial position
        y = 750
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Crop Health Scan Report")
        y -= 30
        
        # Add content
        c.setFont("Helvetica", 12)
        c.drawString(50, y, f"Scan ID: {scan_id}")
        y -= 20
        c.drawString(50, y, f"Date: {timestamp}")
        y -= 20
        c.drawString(50, y, f"Crop Type: {crop_type}")
        y -= 20
        c.drawString(50, y, f"Condition: {condition}")
        y -= 20
        c.drawString(50, y, f"Confidence: {confidence:.2f}%")
        y -= 30
        
        # Add image if available
        if os.path.exists(image_path):
            try:
                c.drawImage(image_path, 50, y-200, width=200, height=200)
                y -= 220
            except:
                y -= 20
                c.drawString(50, y, f"Image Path: {image_path}")
        
        # Add recommendations
        recommendations = self.get_report_recommendations(crop_type)
        c.setFont("Helvetica-Bold", 14)
        y -= 30
        c.drawString(50, y, "Recommendations:")
        y -= 20
        
        # Split recommendations into lines
        c.setFont("Helvetica", 12)
        for line in recommendations.split('\n'):
            if line.strip():
                c.drawString(50, y, line.strip())
                y -= 20
        
        # Add generation timestamp
        y -= 30
        c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save PDF
        c.save()
        
        # Open PDF automatically
        os.startfile(report_path)
        
        messagebox.showinfo("Success", f"PDF Report generated and opened: {report_path}")

    def init_database(self):
        self.conn = sqlite3.connect('crop_health.db')
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS scans
                    (id INTEGER PRIMARY KEY,
                     crop_type TEXT,
                     condition TEXT,
                     confidence REAL,
                     timestamp DATETIME,
                     image_path TEXT)''')
        self.conn.commit()

    def load_disease_database(self):
        self.disease_db = {
            "leaf": {
                "description": "Leaf health issues detected",
                "remedy": "1. Remove affected leaves\n2. Apply appropriate treatment\n3. Monitor leaf health",
                "prevention": "1. Regular inspection\n2. Proper watering\n3. Maintain plant spacing"
            },
            "plant": {
                "description": "General plant condition issues",
                "remedy": "1. Check soil condition\n2. Adjust water and nutrients\n3. Monitor environment",
                "prevention": "1. Regular maintenance\n2. Balanced fertilization\n3. Pest monitoring"
            },
            "corn": {
                "description": "Corn plant health issues",
                "remedy": "1. Check for pest damage\n2. Apply suitable fertilizer\n3. Adjust irrigation",
                "prevention": "1. Crop rotation\n2. Soil preparation\n3. Regular monitoring"
            },
            "vegetable": {
                "description": "Vegetable crop issues",
                "remedy": "1. Remove affected parts\n2. Apply organic treatments\n3. Adjust care routine",
                "prevention": "1. Companion planting\n2. Proper spacing\n3. Regular inspection"
            }
        }

    def show_recommendations(self, condition):
        # Find matching condition in database
        matched_condition = None
        for key in self.disease_db.keys():
            if key in condition.lower():
                matched_condition = key
                break
        
        if matched_condition:
            info = self.disease_db[matched_condition]
            recommendations = f"""
Analyzed Condition: {condition}

Description:
{info['description']}

Recommended Treatment:
{info['remedy']}

Prevention Guidelines:
{info['prevention']}

Additional Notes:
• Continue monitoring the affected area
• Document any changes in condition
• Consult local agricultural expert if needed
"""
        else:
            recommendations = f"""
Analyzed Condition: {condition}

General Prevention Guidelines:
1. Regular Monitoring
   • Daily plant inspection
   • Document changes
   • Photo documentation

2. Basic Care:
   • Proper watering
   • Adequate sunlight
   • Good air circulation

3. Maintenance:
   • Clean tools
   • Remove debris
   • Check soil health

4. Professional Support:
   • Consult local experts
   • Follow regional guidelines
   • Keep treatment records
"""
        
        self.recommendation_text.delete(1.0, tk.END)
        self.recommendation_text.insert(tk.END, recommendations)

    def load_model(self):
        # Using MobileNetV2 as base model
        model = MobileNetV2(weights='imagenet')
        return model

    def setup_gui(self):
        # Create main frames
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH)
        
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
        
        # Image upload section
        self.setup_image_upload()
        
        # Analysis results section
        self.setup_results_section()
        
        # History section
        self.setup_history_section()

    def setup_image_upload(self):
        upload_frame = ttk.LabelFrame(self.left_frame, text="Image Upload")
        upload_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Image preview
        self.image_label = ttk.Label(upload_frame)
        self.image_label.pack(pady=10)
        
        # Upload button
        ttk.Button(upload_frame, 
                  text="Upload Image", 
                  command=self.upload_image).pack(pady=5)
        
        # Analyze button
        ttk.Button(upload_frame, 
                  text="Analyze Image", 
                  command=self.analyze_image).pack(pady=5)

    def setup_results_section(self):
        results_frame = ttk.LabelFrame(self.right_frame, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Results text
        self.results_text = tk.Text(results_frame, 
                                  height=10, width=50, 
                                  font=('Arial', 12))
        self.results_text.pack(pady=10, padx=5)
        
        # Recommendations
        self.recommendation_text = tk.Text(results_frame, 
                                         height=10, width=50, 
                                         font=('Arial', 12))
        self.recommendation_text.pack(pady=10, padx=5)

    def setup_history_section(self):
        history_frame = ttk.LabelFrame(self.left_frame, text="Scan History")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frame for treeview and scrollbar
        tree_frame = ttk.Frame(history_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # History tree
        self.history_tree = ttk.Treeview(tree_frame, 
                                       columns=("Date", "Crop", "Condition"),
                                       show="headings")
        self.history_tree.heading("Date", text="Date")
        self.history_tree.heading("Crop", text="Crop Type")
        self.history_tree.heading("Condition", text="Condition")
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add vertical scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure treeview to use scrollbar
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

    def update_history(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        c = self.conn.cursor()
        c.execute("""SELECT timestamp, crop_type, condition 
                    FROM scans 
                    ORDER BY timestamp DESC 
                    LIMIT 10""")
        
        for scan in c.fetchall():
            self.history_tree.insert('', tk.END, values=scan)

    def get_report_recommendations(self, condition):
        matched_condition = None
        for key in self.disease_db.keys():
            if key in condition.lower():
                matched_condition = key
                break
                
        if matched_condition:
            info = self.disease_db[matched_condition]
            return f"""
Description: {info['description']}
Treatment: {info['remedy']}
Prevention: {info['prevention']}"""
        else:
            return "No specific recommendations available for this condition."

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)

    def display_image(self, path):
        image = Image.open(path)
        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def analyze_image(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("Error", "Please upload an image first")
            return
        
        # Prepare image for model
        image = tf.keras.preprocessing.image.load_img(
            self.current_image_path, 
            target_size=(224, 224)
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        
        # Get predictions
        predictions = self.model.predict(image_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Display results
        self.display_results(decoded_predictions)
        
        # Save to database
        self.save_scan_result(decoded_predictions[0])
        
        # Update history
        self.update_history()

    def display_results(self, predictions):
        self.results_text.delete(1.0, tk.END)
        self.recommendation_text.delete(1.0, tk.END)
        
        results = "Analysis Results:\n\n"
        for pred in predictions:
            results += f"{pred[1]}: {pred[2]*100:.2f}%\n"
        
        self.results_text.insert(tk.END, results)
        
        # Show recommendations
        self.show_recommendations(predictions[0][1])

    def save_scan_result(self, prediction):
        c = self.conn.cursor()
        c.execute("""INSERT INTO scans 
                    (crop_type, condition, confidence, timestamp, image_path) 
                    VALUES (?, ?, ?, ?, ?)""",
                 (prediction[1], prediction[0], prediction[2],
                  datetime.now(), self.current_image_path))
        self.conn.commit()

    def run(self):
        self.root.mainloop()
        self.conn.close()

if __name__ == "__main__":
    app = CropHealthScanner()
    app.run()