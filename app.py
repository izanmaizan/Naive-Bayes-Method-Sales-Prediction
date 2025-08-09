from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import json
import datetime

app = Flask(__name__)

class CarSalesPredictionWeb:
    def __init__(self):
        self.load_model_and_data()
    
    def load_model_and_data(self):
        """Load trained model, encoders, and training data"""
        try:
            # Load model
            with open('naive_bayes_car_sales.sav', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load encoders
            with open('encoders_car_sales.sav', 'rb') as f:
                encoders = pickle.load(f)
                self.le_gender = encoders['gender_encoder']
                self.le_satisfied = encoders['satisfied_encoder']
            
            # Load training data for visualization
            try:
                data = pd.read_excel('sale.xlsx')
                data = data.dropna()
                data.columns = data.columns.str.strip().str.replace(' ', '')
                
                # Encode the training data
                data_encoded = data.copy()
                data_encoded['Gender'] = self.le_gender.transform(data_encoded['Gender'])
                data_encoded['satisfied'] = self.le_satisfied.transform(data_encoded['satisfied'])
                
                self.training_data = data_encoded
                self.model_loaded = True
                
            except Exception as e:
                print(f"Warning: Could not load training data for visualization: {e}")
                self.training_data = None
                self.model_loaded = True
                
        except FileNotFoundError:
            print("Error: Model files not found! Run main.py first.")
            self.model_loaded = False
            self.training_data = None
        except Exception as e:
            print(f"Error: Failed to load model: {str(e)}")
            self.model_loaded = False
            self.training_data = None
    
    def predict_purchase(self, gender, age, salary, satisfied):
        """Perform purchase prediction"""
        if not self.model_loaded:
            return {"error": "Model not loaded properly!"}
        
        try:
            # Validate inputs
            age = int(age)
            salary = int(salary)
            
            if not 15 <= age <= 100:
                return {"error": "Age must be between 15-100 years"}
            
            if salary < 0:
                return {"error": "Salary cannot be negative"}
            
            # Encode categorical data
            gender_encoded = self.le_gender.transform([gender])[0]
            satisfied_encoded = self.le_satisfied.transform([satisfied])[0]
            
            # Create input array
            input_data = np.array([gender_encoded, age, salary, satisfied_encoded]).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            probabilities = self.model.predict_proba(input_data)[0]
            
            prob_no_buy = probabilities[0] * 100
            prob_buy = probabilities[1] * 100
            
            # Generate detailed analysis
            analysis = self.generate_detailed_analysis(
                gender, age, salary, satisfied, gender_encoded, satisfied_encoded, 
                input_data, prediction, probabilities
            )
            
            # Generate recommendation
            if prediction == 1:
                status = "Will Purchase"
                confidence = prob_buy
                if confidence > 80:
                    recommendation = "High priority customer\nContact immediately\nOffer exclusive deals"
                elif confidence > 60:
                    recommendation = "Potential customer\nSchedule follow-up\nSend product information"
                else:
                    recommendation = "Low priority\nAdd to nurturing campaign\nMonitor interest"
            else:
                status = "Won't Purchase"
                confidence = prob_no_buy
                if confidence > 80:
                    recommendation = "Not interested currently\nFollow up in 3-6 months\nFocus on other prospects"
                else:
                    recommendation = "Uncertain customer\nProvide more information\nAddress concerns"
            
            return {
                "prediction": int(prediction),
                "status": status,
                "confidence": round(confidence, 1),
                "prob_buy": round(prob_buy, 1),
                "prob_no_buy": round(prob_no_buy, 1),
                "recommendation": recommendation,
                "analysis": analysis
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_detailed_analysis(self, gender, age, salary, satisfied, 
                                 gender_encoded, satisfied_encoded, 
                                 input_data, prediction, probabilities):
        """Generate detailed step-by-step analysis like terminal output"""
        
        analysis = []
        
        # Header Analysis
        analysis.append({
            "title": "🚗 CAR SALES PREDICTION ANALYSIS 🚗",
            "content": [
                "=" * 60,
                "📋 Sistem prediksi pembelian mobil menggunakan Naive Bayes",
                "📊 Analisis berdasarkan: Gender, Usia, Gaji, dan Kepuasan",
                "⚡ Proses step-by-step untuk customer yang diprediksi",
                "=" * 60
            ]
        })
        
        # 1. Input Data Processing  
        analysis.append({
            "title": "=== TAHAP 1: INPUT DATA CUSTOMER ===",
            "content": [
                f"📋 RINGKASAN DATA CUSTOMER:",
                f"👤 Gender: {gender}",
                f"🎂 Age: {age} tahun",
                f"💰 Estimated Salary: ${salary:,}",
                f"😊 Satisfied: {satisfied}",
                "",
                "⏳ Memproses data customer...",
                "✅ Validasi input berhasil",
                f"   - Age range valid (15-100): {age} ✓",
                f"   - Salary valid (≥0): ${salary:,} ✓",
                f"   - Gender valid: {gender} ✓", 
                f"   - Satisfaction valid: {satisfied} ✓"
            ]
        })
        
        # 2. Encoding Process
        analysis.append({
            "title": "=== TAHAP 2: ENCODING DATA KATEGORIKAL ===",
            "content": [
                "🔄 Loading encoders yang digunakan saat training...",
                "✅ Encoders berhasil dimuat",
                "",
                "🔢 Encoding input data:",
                f"✅ Gender encoded: {gender} -> {gender_encoded}",
                f"   (Label Encoder: Female=0, Male=1)",
                f"✅ Satisfied encoded: {satisfied} -> {satisfied_encoded}",
                f"   (Label Encoder: no=0, yes=1)",
                "",
                f"🔢 Input array untuk model:",
                f"   [Gender, Age, Salary, Satisfied]",
                f"   {input_data[0]}",
                f"   [{gender_encoded}, {age}, {salary}, {satisfied_encoded}]",
                "",
                "✅ Data preprocessing selesai"
            ]
        })
        
        # 3. Model Statistics
        if self.training_data is not None:
            class_counts = np.bincount(self.training_data['Purchased'].values)
            total_samples = len(self.training_data)
            prior_no_buy = class_counts[0] / total_samples
            prior_buy = class_counts[1] / total_samples
            
            analysis.append({
                "title": "=== TAHAP 3: GAUSSIAN NAIVE BAYES CALCULATIONS ===",
                "content": [
                    "🤖 Loading model dan melakukan analisis...",
                    "✅ Model berhasil dimuat",
                    "",
                    f"📊 Prior Probabilities dari Training Data:",
                    f"   P(No Purchase) = {prior_no_buy:.4f} ({class_counts[0]}/{total_samples})",
                    f"   P(Purchase) = {prior_buy:.4f} ({class_counts[1]}/{total_samples})",
                    "",
                    "📈 Feature Statistics by Class:"
                ]
            })
            
            for class_label, class_name in [(0, "No Purchase"), (1, "Purchase")]:
                class_data = self.training_data[self.training_data['Purchased'] == class_label]
                analysis[-1]["content"].extend([
                    f"",
                    f"  📊 {class_name} Class ({len(class_data)} samples):",
                    f"    🎂 Age: mean={class_data['Age'].mean():.2f}, std={class_data['Age'].std():.2f}",
                    f"    💰 Salary: mean=${class_data['EstimatedSalary'].mean():.0f}, std=${class_data['EstimatedSalary'].std():.0f}",
                    f"    👤 Gender distribution: {dict(class_data['Gender'].value_counts())}",
                    f"    😊 Satisfaction: {dict(class_data['satisfied'].value_counts())}"
                ])
        
        # 4. Likelihood Calculations
        prob_no_buy = probabilities[0]
        prob_buy = probabilities[1]
        
        analysis.append({
            "title": "=== TAHAP 4: LIKELIHOOD CALCULATIONS ===",
            "content": [
                f"🔍 Hasil prediksi untuk input {input_data[0]}:",
                "",
                f"⚙️ Naive Bayes Calculations:",
                f"   Untuk setiap fitur, model menghitung likelihood:",
                f"   P(Gender={gender_encoded}|Class) × P(Age={age}|Class) × ", 
                f"   P(Salary={salary}|Class) × P(Satisfied={satisfied_encoded}|Class)",
                "",
                f"📊 Posterior Probabilities (setelah normalisasi):",
                f"   P(No Purchase|features) = {prob_no_buy:.6f}",
                f"   P(Purchase|features) = {prob_buy:.6f}",
                "",
                f"📈 Distribusi Probabilitas:",
                f"   🔴 Tidak Beli: {prob_no_buy*100:.2f}%",
                f"   🟢 Beli      : {prob_buy*100:.2f}%"
            ]
        })
        
        # 5. Visual Probability Bars
        bar_tidak_beli = "█" * int(prob_no_buy*100 // 5)
        bar_beli = "█" * int(prob_buy*100 // 5)
        
        analysis.append({
            "title": "=== TAHAP 5: VISUALISASI PROBABILITAS ===",
            "content": [
                "📈 Visualisasi distribusi probabilitas:",
                f"   🔴 Tidak Beli: {bar_tidak_beli} {prob_no_buy*100:.1f}%",
                f"   🟢 Beli      : {bar_beli} {prob_buy*100:.1f}%",
                "",
                "🎯 Decision Boundary:",
                f"   Threshold = 50%",
                f"   Prediksi: {'Purchase' if prob_buy > 0.5 else 'No Purchase'}"
            ]
        })
        
        # 6. Final Prediction with detailed reasoning
        prediction_text = "PURCHASE" if prediction == 1 else "NO PURCHASE"
        confidence = max(prob_buy, prob_no_buy) * 100
        
        reasoning_content = [
            "=" * 50,
            "🎯 HASIL PREDIKSI AKHIR:",
            "=" * 50
        ]
        
        if prediction == 1:
            reasoning_content.extend([
                "✅ CUSTOMER AKAN MEMBELI MOBIL",
                f"🔥 Confidence: {prob_buy*100:.2f}%",
                "",
                f"💡 Alasan model memprediksi PURCHASE:",
                f"   Purchase probability ({prob_buy:.4f}) > No Purchase probability ({prob_no_buy:.4f})"
            ])
        else:
            reasoning_content.extend([
                "❌ CUSTOMER TIDAK AKAN MEMBELI MOBIL", 
                f"📊 Confidence: {prob_no_buy*100:.2f}%",
                "",
                f"💡 Alasan model memprediksi NO PURCHASE:",
                f"   No Purchase probability ({prob_no_buy:.4f}) > Purchase probability ({prob_buy:.4f})"
            ])
        
        # Add detailed factor analysis
        reasoning_content.append("")
        reasoning_content.append("🔍 Analisis Faktor Kontribusi:")
        
        if age >= 30 and age <= 50:
            reasoning_content.append(f"  ✅ Age ({age}) dalam range optimal pembelian (30-50)")
        elif age < 25:
            reasoning_content.append(f"  ⚠️  Age ({age}) cukup muda, kemungkinan daya beli terbatas")
        elif age > 55:
            reasoning_content.append(f"  ⚠️  Age ({age}) cukup tua, prioritas berbeda")
            
        if salary >= 60000:
            reasoning_content.append(f"  ✅ Salary tinggi (${salary:,}) meningkatkan probabilitas beli")
        elif salary < 30000:
            reasoning_content.append(f"  ⚠️  Salary rendah (${salary:,}) menurunkan probabilitas beli") 
        else:
            reasoning_content.append(f"  ⚖️  Salary menengah (${salary:,}) memberikan pengaruh netral")
            
        if satisfied == 'yes':
            reasoning_content.append(f"  ✅ Customer puas, meningkatkan probabilitas pembelian")
        else:
            reasoning_content.append(f"  ⚠️  Customer tidak puas, menurunkan probabilitas pembelian")
            
        if gender == 'Male':
            reasoning_content.append(f"  👤 Gender: Male - lihat distribusi di training data")
        else:
            reasoning_content.append(f"  👤 Gender: Female - lihat distribusi di training data")
        
        # Add recommendation
        reasoning_content.append("")
        if prediction == 1:
            if confidence > 80:
                reasoning_content.append("💡 Rekomendasi: Prioritas Tinggi - Follow up segera!")
            elif confidence > 60:
                reasoning_content.append("💡 Rekomendasi: Prioritas Sedang - Customer potensial")
            else:
                reasoning_content.append("💡 Rekomendasi: Prioritas Rendah - Perlu nurturing")
        else:
            if confidence > 80:
                reasoning_content.append("💡 Rekomendasi: Customer tidak tertarik saat ini")
            else:
                reasoning_content.append("💡 Rekomendasi: Customer ragu-ragu, berikan info lebih")
        
        reasoning_content.extend([
            "",
            "=" * 50,
            f"📅 Analisis selesai: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50
        ])
        
        analysis.append({
            "title": "=== TAHAP 6: DECISION EXPLANATION ===",
            "content": reasoning_content
        })
        
        return analysis
    
    def generate_model_visualization(self):
        """Generate model visualization plots with explanations"""
        print("🔄 Starting visualization generation...")
        
        if self.training_data is None:
            error_msg = "Training data not available for visualization. Please ensure sale.xlsx file exists and run main.py first."
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        
        try:
            print(f"📊 Training data shape: {self.training_data.shape}")
            
            # Create comprehensive subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    '📈 Age Distribution by Purchase Decision', 
                    '💰 Salary Distribution by Purchase Decision',
                    '👤 Purchase Rate by Gender', 
                    '😊 Purchase Rate by Satisfaction',
                    '🎯 Age vs Salary Scatter Plot', 
                    '📊 Feature Importance Analysis'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            print("📈 Creating age distribution plots...")
            # 1. Age Distribution by Class
            for class_val, color, name in [(0, '#e74c3c', 'No Purchase'), (1, '#27ae60', 'Purchase')]:
                data = self.training_data[self.training_data['Purchased'] == class_val]['Age']
                fig.add_trace(
                    go.Histogram(x=data, name=name, opacity=0.7, 
                               marker_color=color, nbinsx=20, histnorm='probability density'),
                    row=1, col=1
                )
            
            print("💰 Creating salary distribution plots...")
            # 2. Salary Distribution by Class
            for class_val, color, name in [(0, '#e74c3c', 'No Purchase'), (1, '#27ae60', 'Purchase')]:
                data = self.training_data[self.training_data['Purchased'] == class_val]['EstimatedSalary']
                fig.add_trace(
                    go.Histogram(x=data, name=name, opacity=0.7, 
                               marker_color=color, nbinsx=20, histnorm='probability density',
                               showlegend=False),
                    row=1, col=2
                )
            
            print("👤 Creating gender distribution plots...")
            # 3. Gender Distribution
            gender_purchase = pd.crosstab(self.training_data['Gender'], self.training_data['Purchased'], normalize='index')
            fig.add_trace(
                go.Bar(x=['Female', 'Male'], y=gender_purchase[0], name='No Purchase', 
                       marker_color='#e74c3c', showlegend=False),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=['Female', 'Male'], y=gender_purchase[1], name='Purchase', 
                       marker_color='#27ae60', showlegend=False),
                row=2, col=1
            )
            
            print("😊 Creating satisfaction distribution plots...")
            # 4. Satisfaction Distribution
            satisfied_purchase = pd.crosstab(self.training_data['satisfied'], self.training_data['Purchased'], normalize='index')
            fig.add_trace(
                go.Bar(x=['No', 'Yes'], y=satisfied_purchase[0], name='No Purchase', 
                       marker_color='#e74c3c', showlegend=False),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=['No', 'Yes'], y=satisfied_purchase[1], name='Purchase', 
                       marker_color='#27ae60', showlegend=False),
                row=2, col=2
            )
            
            print("🎯 Creating scatter plot...")
            # 5. Age vs Salary Scatter Plot
            for class_val, color, name in [(0, '#e74c3c', 'No Purchase'), (1, '#27ae60', 'Purchase')]:
                data = self.training_data[self.training_data['Purchased'] == class_val]
                fig.add_trace(
                    go.Scatter(x=data['Age'], y=data['EstimatedSalary'], mode='markers',
                              name=name, marker=dict(color=color, opacity=0.6, size=8),
                              showlegend=False),
                    row=3, col=1
                )
            
            print("📊 Calculating feature importance...")
            # 6. Feature Importance
            features = ['Gender', 'Age', 'EstimatedSalary', 'satisfied']
            importance_scores = []
            
            for feature in features:
                class_0_data = self.training_data[self.training_data['Purchased'] == 0][feature]
                class_1_data = self.training_data[self.training_data['Purchased'] == 1][feature]
                
                mean_diff = abs(class_1_data.mean() - class_0_data.mean())
                combined_std = np.sqrt((class_0_data.var() + class_1_data.var()) / 2)
                separation_score = mean_diff / (combined_std + 1e-8)
                importance_scores.append(separation_score)
            
            importance_scores = np.array(importance_scores)
            importance_scores = importance_scores / importance_scores.sum()
            
            fig.add_trace(
                go.Bar(y=features, x=importance_scores, orientation='h',
                       marker_color='#3498db', showlegend=False,
                       text=[f'{score:.3f}' for score in importance_scores],
                       textposition='outside'),
                row=3, col=2
            )
            
            print("🎨 Updating layout...")
            # Update layout with explanations
            fig.update_layout(
                height=900,
                title_text="<b>🚗 Car Sales Prediction Model Analysis Dashboard</b><br><sub>📊 Comprehensive view of Gaussian Naive Bayes feature distributions and insights</sub>",
                title_x=0.5,
                title_font_size=20,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Update axes labels with detailed descriptions
            fig.update_xaxes(title_text="Age (years)", row=1, col=1, gridcolor='lightgray')
            fig.update_xaxes(title_text="Estimated Salary ($)", row=1, col=2, gridcolor='lightgray')
            fig.update_xaxes(title_text="Gender", row=2, col=1, gridcolor='lightgray')
            fig.update_xaxes(title_text="Customer Satisfaction", row=2, col=2, gridcolor='lightgray')
            fig.update_xaxes(title_text="Age (years)", row=3, col=1, gridcolor='lightgray')
            fig.update_xaxes(title_text="Importance Score", row=3, col=2, gridcolor='lightgray')
            
            fig.update_yaxes(title_text="Density", row=1, col=1, gridcolor='lightgray')
            fig.update_yaxes(title_text="Density", row=1, col=2, gridcolor='lightgray')
            fig.update_yaxes(title_text="Purchase Rate", row=2, col=1, gridcolor='lightgray')
            fig.update_yaxes(title_text="Purchase Rate", row=2, col=2, gridcolor='lightgray')
            fig.update_yaxes(title_text="Estimated Salary ($)", row=3, col=1, gridcolor='lightgray')
            fig.update_yaxes(title_text="Features", row=3, col=2, gridcolor='lightgray')
            
            print("🔄 Converting to HTML...")
            # Convert to HTML with better configuration
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'car_sales_model_analysis',
                    'height': 900,
                    'width': 1200,
                    'scale': 1
                }
            }
            
            html_div = fig.to_html(
                include_plotlyjs='cdn',
                config=config,
                div_id="model-visualization"
            )
            
            print("✅ Visualization HTML generated successfully")
            return {"visualization": html_div}
            
        except Exception as e:
            error_msg = f"Error generating visualization: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            print(f"📝 Full traceback: {traceback.format_exc()}")
            return {"error": error_msg}

# Initialize the predictor
predictor = CarSalesPredictionWeb()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        gender = data.get('gender')
        age = data.get('age')
        salary = data.get('salary')
        satisfied = data.get('satisfied')
        
        result = predictor.predict_purchase(gender, age, salary, satisfied)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/visualization')
def visualization():
    """Generate model visualization"""
    try:
        result = predictor.generate_model_visualization()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)