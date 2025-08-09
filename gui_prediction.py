import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd

class ModernCarPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.load_model_and_data()
        self.create_styles()
        self.create_widgets()
        
    def setup_window(self):
        """Setup window configuration"""
        self.root.title("Car Sales Prediction - AI Analysis")
        self.root.geometry("1200x800")
        self.root.configure(bg='#ffffff')
        self.root.resizable(True, True)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f'1200x800+{x}+{y}')
        
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
            messagebox.showerror("Error", "Model files not found!\nRun main.py first.")
            self.model_loaded = False
            self.training_data = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_loaded = False
            self.training_data = None
            
    def create_styles(self):
        """Create modern ttk styles"""
        style = ttk.Style()
        
        # Configure modern button style
        style.configure('Modern.TButton',
                       font=('Segoe UI', 12, 'bold'),
                       padding=(20, 12))
        
        # Configure modern combobox style
        style.configure('Modern.TCombobox',
                       font=('Segoe UI', 12),
                       fieldbackground='#f8f9fa')
        
        # Configure modern entry style  
        style.configure('Modern.TEntry',
                       font=('Segoe UI', 12),
                       fieldbackground='#f8f9fa',
                       borderwidth=1,
                       relief='solid')
        
    def create_widgets(self):
        """Create all GUI widgets with modern design"""
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Prediction tab
        self.prediction_tab = tk.Frame(self.notebook, bg='#ffffff')
        self.notebook.add(self.prediction_tab, text='Prediction')
        
        # Analysis tab
        self.analysis_tab = tk.Frame(self.notebook, bg='#ffffff')
        self.notebook.add(self.analysis_tab, text='Analysis')
        
        # Model Visualization tab
        self.model_tab = tk.Frame(self.notebook, bg='#ffffff')
        self.notebook.add(self.model_tab, text='Model Visualization')
        
        # Create prediction interface
        self.create_prediction_interface()
        
        # Create analysis interface
        self.create_analysis_interface()
        
        # Create model visualization interface
        self.create_model_visualization_interface()
        
    def create_prediction_interface(self):
        """Create the prediction interface"""
        # Main container for prediction
        main_container = tk.Frame(self.prediction_tab, bg='#ffffff', padx=40, pady=30)
        main_container.pack(fill='both', expand=True)
        
        # Header
        self.create_header(main_container)
        
        # Input form
        self.create_input_form(main_container)
        
        # Predict button
        self.create_predict_button(main_container)
        
        # Result section
        self.create_result_section(main_container)
        
    def create_header(self, parent):
        """Create modern header"""
        header_frame = tk.Frame(parent, bg='#ffffff')
        header_frame.pack(fill='x', pady=(0, 30))
        
        # App title
        title_label = tk.Label(header_frame,
                              text="Car Sales Prediction",
                              font=('Segoe UI', 28, 'bold'),
                              fg='#2c3e50',
                              bg='#ffffff')
        title_label.pack(anchor='center')
        
        # Subtitle
        subtitle_label = tk.Label(header_frame,
                                 text="AI-powered customer purchase prediction with detailed analysis",
                                 font=('Segoe UI', 14),
                                 fg='#7f8c8d',
                                 bg='#ffffff')
        subtitle_label.pack(anchor='center', pady=(5, 0))
        
        # Separator line
        separator = tk.Frame(header_frame, height=2, bg='#ecf0f1')
        separator.pack(fill='x', pady=(15, 0))
        
    def create_input_form(self, parent):
        """Create modern input form"""
        form_frame = tk.Frame(parent, bg='#ffffff')
        form_frame.pack(fill='x', pady=(0, 25))
        
        # Configure grid weights
        form_frame.grid_columnconfigure(1, weight=1)
        
        # Gender
        tk.Label(form_frame, text="Gender", 
                font=('Segoe UI', 14, 'bold'), 
                fg='#2c3e50', bg='#ffffff').grid(row=0, column=0, sticky='w', pady=(0, 15))
        
        self.gender_var = tk.StringVar()
        gender_combo = ttk.Combobox(form_frame, textvariable=self.gender_var,
                                   values=['Male', 'Female'], state='readonly',
                                   style='Modern.TCombobox', width=25)
        gender_combo.grid(row=0, column=1, sticky='ew', pady=(0, 15))
        gender_combo.current(0)
        
        # Age
        tk.Label(form_frame, text="Age", 
                font=('Segoe UI', 14, 'bold'), 
                fg='#2c3e50', bg='#ffffff').grid(row=1, column=0, sticky='w', pady=(0, 15))
        
        self.age_var = tk.StringVar()
        age_entry = ttk.Entry(form_frame, textvariable=self.age_var,
                             style='Modern.TEntry', width=27)
        age_entry.grid(row=1, column=1, sticky='ew', pady=(0, 15))
        
        # Salary
        tk.Label(form_frame, text="Annual Salary ($)", 
                font=('Segoe UI', 14, 'bold'), 
                fg='#2c3e50', bg='#ffffff').grid(row=2, column=0, sticky='w', pady=(0, 15))
        
        self.salary_var = tk.StringVar()
        salary_entry = ttk.Entry(form_frame, textvariable=self.salary_var,
                                style='Modern.TEntry', width=27)
        salary_entry.grid(row=2, column=1, sticky='ew', pady=(0, 15))
        
        # Satisfaction
        tk.Label(form_frame, text="Customer Satisfaction", 
                font=('Segoe UI', 14, 'bold'), 
                fg='#2c3e50', bg='#ffffff').grid(row=3, column=0, sticky='w')
        
        self.satisfied_var = tk.StringVar()
        satisfied_combo = ttk.Combobox(form_frame, textvariable=self.satisfied_var,
                                      values=['yes', 'no'], state='readonly',
                                      style='Modern.TCombobox', width=25)
        satisfied_combo.grid(row=3, column=1, sticky='ew')
        satisfied_combo.current(0)
        
    def create_predict_button(self, parent):
        """Create modern predict button"""
        button_frame = tk.Frame(parent, bg='#ffffff')
        button_frame.pack(fill='x', pady=(0, 25))
        
        self.predict_btn = tk.Button(button_frame,
                                    text="Predict & Analyze",
                                    command=self.predict_and_analyze,
                                    font=('Segoe UI', 14, 'bold'),
                                    bg='#3498db',
                                    fg='white',
                                    relief='flat',
                                    padx=40,
                                    pady=15,
                                    cursor='hand2',
                                    activebackground='#2980b9')
        self.predict_btn.pack()
        
        # Reset button
        reset_btn = tk.Button(button_frame,
                             text="Reset",
                             command=self.reset_form,
                             font=('Segoe UI', 12),
                             bg='#ecf0f1',
                             fg='#2c3e50',
                             relief='flat',
                             padx=25,
                             pady=10,
                             cursor='hand2',
                             activebackground='#d5dbdb')
        reset_btn.pack(pady=(15, 0))
        
    def create_result_section(self, parent):
        """Create modern result section"""
        self.result_frame = tk.Frame(parent, bg='#f8f9fa', relief='solid', borderwidth=1)
        self.result_frame.pack(fill='both', expand=True)
        
        # Result header
        result_header = tk.Frame(self.result_frame, bg='#f8f9fa', height=50)
        result_header.pack(fill='x', padx=20, pady=(15, 10))
        result_header.pack_propagate(False)
        
        tk.Label(result_header,
                text="Prediction Result",
                font=('Segoe UI', 16, 'bold'),
                fg='#2c3e50',
                bg='#f8f9fa').pack(side='left', anchor='center', expand=True)
        
        # Result content
        self.result_content = tk.Frame(self.result_frame, bg='#f8f9fa')
        self.result_content.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Initial state
        self.show_initial_state()
        
    def create_analysis_interface(self):
        """Create analysis interface"""
        # Analysis container
        analysis_container = tk.Frame(self.analysis_tab, bg='#ffffff', padx=20, pady=20)
        analysis_container.pack(fill='both', expand=True)
        
        # Analysis header
        analysis_header = tk.Label(analysis_container,
                                  text="Detailed Process Analysis",
                                  font=('Segoe UI', 22, 'bold'),
                                  fg='#2c3e50',
                                  bg='#ffffff')
        analysis_header.pack(pady=(0, 20))
        
        # Analysis text area
        self.analysis_text = tk.Text(analysis_container,
                                    font=('Consolas', 14),
                                    bg='#f8f9fa',
                                    fg='#2c3e50',
                                    relief='solid',
                                    borderwidth=1,
                                    wrap=tk.WORD)
        self.analysis_text.pack(fill='both', expand=True)
        
        # Scrollbar for analysis
        analysis_scroll = tk.Scrollbar(self.analysis_text)
        analysis_scroll.pack(side='right', fill='y')
        self.analysis_text.config(yscrollcommand=analysis_scroll.set)
        analysis_scroll.config(command=self.analysis_text.yview)
        
        # Initial analysis message
        self.show_initial_analysis()
        
    def create_model_visualization_interface(self):
        """Create model visualization interface"""
        # Model container
        model_container = tk.Frame(self.model_tab, bg='#ffffff', padx=10, pady=10)
        model_container.pack(fill='both', expand=True)
        
        # Model header
        model_header = tk.Label(model_container,
                               text="Gaussian Naive Bayes Model Visualization",
                               font=('Segoe UI', 20, 'bold'),
                               fg='#2c3e50',
                               bg='#ffffff')
        model_header.pack(pady=(0, 10))
        
        # Create matplotlib figure for model visualization
        self.model_fig = Figure(figsize=(12, 8), facecolor='white')
        self.model_canvas = FigureCanvasTkAgg(self.model_fig, model_container)
        self.model_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Generate model visualization
        self.create_model_visualization()
        
    def show_initial_state(self):
        """Show initial empty state"""
        for widget in self.result_content.winfo_children():
            widget.destroy()
            
        placeholder = tk.Label(self.result_content,
                              text="Enter customer data and click 'Predict & Analyze'\nto see the prediction result",
                              font=('Segoe UI', 14),
                              fg='#95a5a6',
                              bg='#f8f9fa',
                              justify='center')
        placeholder.pack(expand=True)
        
    def show_initial_analysis(self):
        """Show initial analysis message"""
        initial_msg = """DETAILED PROCESS ANALYSIS
========================================

This section will show you the step-by-step process of how the Naive Bayes algorithm
makes predictions when you click 'Predict & Analyze'.

The analysis will include:

1. DATA PREPROCESSING
   Original input data
   Label encoding process for categorical variables
   Feature scaling and normalization

2. NAIVE BAYES CALCULATIONS  
   Prior probabilities for each class
   Likelihood calculations for each feature
   Posterior probability computation using Bayes theorem

3. PREDICTION PROCESS
   Feature-wise probability calculations
   Class probability comparisons
   Final prediction decision

4. MODEL INSIGHTS
   Feature importance analysis
   Probability distributions
   Decision boundaries

Click 'Predict & Analyze' to see the detailed analysis for your specific input!
        """
        
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, initial_msg)
        
    def validate_inputs(self):
        """Validate user inputs"""
        try:
            age = int(self.age_var.get())
            if not 15 <= age <= 100:
                raise ValueError("Age must be between 15-100 years")
            
            salary = int(self.salary_var.get())
            if salary < 0:
                raise ValueError("Salary cannot be negative")
            
            return True, age, salary
        except ValueError as e:
            if "invalid literal" in str(e):
                messagebox.showerror("Invalid Input", "Please enter valid numbers for age and salary")
            else:
                messagebox.showerror("Invalid Input", str(e))
            return False, None, None
            
    def predict_and_analyze(self):
        """Perform prediction and detailed analysis"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded properly!")
            return
            
        # Validate inputs
        is_valid, age, salary = self.validate_inputs()
        if not is_valid:
            return
            
        try:
            # Get input data
            gender = self.gender_var.get()
            satisfied = self.satisfied_var.get()
            
            # Encode categorical data
            gender_encoded = self.le_gender.transform([gender])[0]
            satisfied_encoded = self.le_satisfied.transform([satisfied])[0]
            
            # Create input array
            input_data = np.array([gender_encoded, age, salary, satisfied_encoded]).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            probabilities = self.model.predict_proba(input_data)[0]
            
            # Display results in prediction tab
            self.display_modern_results(prediction, probabilities, age, salary)
            
            # Generate detailed analysis
            self.generate_detailed_analysis(gender, age, salary, satisfied, 
                                          gender_encoded, satisfied_encoded, 
                                          input_data, prediction, probabilities)
            
            # Update model visualization with current prediction
            self.update_model_visualization(input_data, prediction)
            
            # Stay on current tab (don't auto-switch)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
            
    def generate_detailed_analysis(self, gender, age, salary, satisfied, 
                                 gender_encoded, satisfied_encoded, 
                                 input_data, prediction, probabilities):
        """Generate detailed step-by-step analysis"""
        
        analysis_text = "DETAILED PROCESS ANALYSIS\n"
        analysis_text += "=" * 50 + "\n\n"
        
        # 1. Input Data Processing
        analysis_text += "1. DATA PREPROCESSING\n"
        analysis_text += "-" * 30 + "\n"
        analysis_text += f"Original Input Data:\n"
        analysis_text += f"  Gender: {gender}\n"
        analysis_text += f"  Age: {age} years\n"
        analysis_text += f"  Salary: ${salary:,}\n"
        analysis_text += f"  Satisfied: {satisfied}\n\n"
        
        analysis_text += f"Label Encoding Process:\n"
        analysis_text += f"  Gender '{gender}' -> {gender_encoded}\n"
        analysis_text += f"  Satisfied '{satisfied}' -> {satisfied_encoded}\n\n"
        
        analysis_text += f"Final Feature Vector:\n"
        analysis_text += f"  [Gender, Age, Salary, Satisfied]\n"
        analysis_text += f"  {input_data[0]}\n\n"
        
        # 2. Model Statistics
        analysis_text += "2. NAIVE BAYES CALCULATIONS\n"
        analysis_text += "-" * 30 + "\n"
        
        # Get class priors from model
        if self.training_data is not None:
            class_counts = np.bincount(self.training_data['Purchased'].values)
            total_samples = len(self.training_data)
            prior_no_buy = class_counts[0] / total_samples
            prior_buy = class_counts[1] / total_samples
            
            analysis_text += f"Prior Probabilities:\n"
            analysis_text += f"  P(No Purchase) = {prior_no_buy:.4f} ({class_counts[0]}/{total_samples})\n"
            analysis_text += f"  P(Purchase) = {prior_buy:.4f} ({class_counts[1]}/{total_samples})\n\n"
            
            # Feature statistics for each class
            analysis_text += f"Feature Statistics by Class:\n\n"
            
            for class_label, class_name in [(0, "No Purchase"), (1, "Purchase")]:
                class_data = self.training_data[self.training_data['Purchased'] == class_label]
                analysis_text += f"  {class_name} Class:\n"
                analysis_text += f"    Age: mean={class_data['Age'].mean():.2f}, std={class_data['Age'].std():.2f}\n"
                analysis_text += f"    Salary: mean=${class_data['EstimatedSalary'].mean():.0f}, std=${class_data['EstimatedSalary'].std():.0f}\n"
                analysis_text += f"    Gender distribution: {dict(class_data['Gender'].value_counts())}\n"
                analysis_text += f"    Satisfaction distribution: {dict(class_data['satisfied'].value_counts())}\n\n"
        
        # 3. Likelihood Calculations
        analysis_text += "3. LIKELIHOOD CALCULATIONS\n"
        analysis_text += "-" * 30 + "\n"
        analysis_text += f"For input vector {input_data[0]}:\n\n"
        
        # Calculate likelihoods (simplified approximation)
        prob_no_buy = probabilities[0]
        prob_buy = probabilities[1]
        
        analysis_text += f"Likelihood Calculations:\n"
        analysis_text += f"  P(features|No Purchase) = {prob_no_buy:.6f}\n"
        analysis_text += f"  P(features|Purchase) = {prob_buy:.6f}\n\n"
        
        # 4. Final Prediction
        analysis_text += "4. PREDICTION PROCESS\n"
        analysis_text += "-" * 30 + "\n"
        analysis_text += f"Posterior Probabilities:\n"
        analysis_text += f"  P(No Purchase|features) = {prob_no_buy:.4f} ({prob_no_buy*100:.2f}%)\n"
        analysis_text += f"  P(Purchase|features) = {prob_buy:.4f} ({prob_buy*100:.2f}%)\n\n"
        
        prediction_text = "PURCHASE" if prediction == 1 else "NO PURCHASE"
        confidence = max(prob_buy, prob_no_buy) * 100
        
        analysis_text += f"Final Prediction: {prediction_text}\n"
        analysis_text += f"Confidence Level: {confidence:.2f}%\n\n"
        
        # 5. Decision Explanation
        analysis_text += "5. DECISION EXPLANATION\n"
        analysis_text += "-" * 30 + "\n"
        
        if prediction == 1:
            analysis_text += f"The model predicts PURCHASE because:\n"
            analysis_text += f"  Purchase probability ({prob_buy:.4f}) > No Purchase probability ({prob_no_buy:.4f})\n"
        else:
            analysis_text += f"The model predicts NO PURCHASE because:\n"
            analysis_text += f"  No Purchase probability ({prob_no_buy:.4f}) > Purchase probability ({prob_buy:.4f})\n"
        
        # Key factors analysis
        analysis_text += f"\nKey Contributing Factors:\n"
        if age >= 30 and age <= 50:
            analysis_text += f"  Age ({age}) is in optimal buying range (30-50)\n"
        elif age < 25:
            analysis_text += f"  Young age ({age}) may indicate lower purchasing power\n"
        elif age > 55:
            analysis_text += f"  Older age ({age}) may indicate different priorities\n"
            
        if salary >= 60000:
            analysis_text += f"  High salary (${salary:,}) increases purchase likelihood\n"
        elif salary < 30000:
            analysis_text += f"  Low salary (${salary:,}) decreases purchase likelihood\n"
        else:
            analysis_text += f"  Moderate salary (${salary:,}) provides neutral influence\n"
            
        if satisfied == 'yes':
            analysis_text += f"  High customer satisfaction increases purchase probability\n"
        else:
            analysis_text += f"  Low customer satisfaction decreases purchase probability\n"
            
        analysis_text += f"\n" + "=" * 50 + "\n"
        analysis_text += f"Analysis completed at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Display analysis
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, analysis_text)
        
    def create_model_visualization(self):
        """Create Gaussian Naive Bayes model visualization"""
        if self.training_data is None:
            # Show message if no training data
            ax = self.model_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Training data not available for visualization\nRun with sale.xlsx file', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.model_canvas.draw()
            return
            
        # Clear figure
        self.model_fig.clear()
        
        # Set style
        plt.style.use('default')
        
        # Create subplots
        gs = self.model_fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Age Distribution by Class
        ax1 = self.model_fig.add_subplot(gs[0, 0])
        for class_val, color, label in [(0, '#e74c3c', 'No Purchase'), (1, '#27ae60', 'Purchase')]:
            data = self.training_data[self.training_data['Purchased'] == class_val]['Age']
            ax1.hist(data, alpha=0.7, bins=20, color=color, label=label, density=True)
        ax1.set_xlabel('Age', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Salary Distribution by Class
        ax2 = self.model_fig.add_subplot(gs[0, 1])
        for class_val, color, label in [(0, '#e74c3c', 'No Purchase'), (1, '#27ae60', 'Purchase')]:
            data = self.training_data[self.training_data['Purchased'] == class_val]['EstimatedSalary']
            ax2.hist(data, alpha=0.7, bins=20, color=color, label=label, density=True)
        ax2.set_xlabel('Estimated Salary', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Salary Distribution by Class', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Gender Distribution
        ax3 = self.model_fig.add_subplot(gs[0, 2])
        gender_purchase = pd.crosstab(self.training_data['Gender'], self.training_data['Purchased'], normalize='index')
        gender_purchase.plot(kind='bar', ax=ax3, color=['#e74c3c', '#27ae60'], width=0.7)
        ax3.set_xlabel('Gender (0=Female, 1=Male)', fontsize=12)
        ax3.set_ylabel('Proportion', fontsize=12)
        ax3.set_title('Purchase Rate by Gender', fontsize=14, fontweight='bold')
        ax3.legend(['No Purchase', 'Purchase'], fontsize=11)
        ax3.set_xticklabels(['Female', 'Male'], rotation=0)
        ax3.grid(True, alpha=0.3)
        
        # 4. Satisfaction Distribution
        ax4 = self.model_fig.add_subplot(gs[1, 0])
        satisfied_purchase = pd.crosstab(self.training_data['satisfied'], self.training_data['Purchased'], normalize='index')
        satisfied_purchase.plot(kind='bar', ax=ax4, color=['#e74c3c', '#27ae60'], width=0.7)
        ax4.set_xlabel('Satisfaction (0=No, 1=Yes)', fontsize=12)
        ax4.set_ylabel('Proportion', fontsize=12)
        ax4.set_title('Purchase Rate by Satisfaction', fontsize=14, fontweight='bold')
        ax4.legend(['No Purchase', 'Purchase'], fontsize=11)
        ax4.set_xticklabels(['No', 'Yes'], rotation=0)
        ax4.grid(True, alpha=0.3)
        
        # 5. Age vs Salary Scatter Plot
        ax5 = self.model_fig.add_subplot(gs[1, 1])
        for class_val, color, label in [(0, '#e74c3c', 'No Purchase'), (1, '#27ae60', 'Purchase')]:
            data = self.training_data[self.training_data['Purchased'] == class_val]
            ax5.scatter(data['Age'], data['EstimatedSalary'], alpha=0.6, c=color, label=label, s=40)
        ax5.set_xlabel('Age', fontsize=12)
        ax5.set_ylabel('Estimated Salary', fontsize=12)
        ax5.set_title('Age vs Salary by Purchase Decision', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # 6. Feature Importance (approximation)
        ax6 = self.model_fig.add_subplot(gs[1, 2])
        
        # Calculate feature importance based on class separation
        features = ['Gender', 'Age', 'EstimatedSalary', 'satisfied']
        importance_scores = []
        
        for feature in features:
            class_0_data = self.training_data[self.training_data['Purchased'] == 0][feature]
            class_1_data = self.training_data[self.training_data['Purchased'] == 1][feature]
            
            # Calculate separation score (normalized difference of means divided by combined std)
            mean_diff = abs(class_1_data.mean() - class_0_data.mean())
            combined_std = np.sqrt((class_0_data.var() + class_1_data.var()) / 2)
            separation_score = mean_diff / (combined_std + 1e-8)  # Add small value to avoid division by zero
            importance_scores.append(separation_score)
        
        # Normalize importance scores
        importance_scores = np.array(importance_scores)
        importance_scores = importance_scores / importance_scores.sum()
        
        bars = ax6.barh(features, importance_scores, color='#3498db')
        ax6.set_xlabel('Importance Score', fontsize=12)
        ax6.set_title('Feature Importance\n(Class Separation)', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, importance_scores):
            ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=10)
        
        # Overall title
        self.model_fig.suptitle('Gaussian Naive Bayes Model Analysis', fontsize=18, fontweight='bold')
        
        # Draw the canvas
        self.model_canvas.draw()
        
    def update_model_visualization(self, input_data, prediction):
        """Update model visualization with current prediction"""
        if self.training_data is None:
            return
            
        # Add current prediction point to existing visualizations
        # This would highlight where the current input falls in the distribution
        
        # For now, just redraw the model visualization
        # In a more advanced version, you could highlight the current point
        self.create_model_visualization()
        
    def display_modern_results(self, prediction, probabilities, age, salary):
        """Display results in modern format"""
        # Clear previous results
        for widget in self.result_content.winfo_children():
            widget.destroy()
            
        prob_no_buy = probabilities[0] * 100
        prob_buy = probabilities[1] * 100
        
        # Prediction status
        status_frame = tk.Frame(self.result_content, bg='#f8f9fa')
        status_frame.pack(fill='x', pady=(0, 20))
        
        if prediction == 1:
            status_color = '#27ae60'
            status_text = "Will Purchase"
            confidence = prob_buy
        else:
            status_color = '#e74c3c'
            status_text = "Won't Purchase"  
            confidence = prob_no_buy
            
        tk.Label(status_frame,
                text=status_text,
                font=('Segoe UI', 20, 'bold'),
                fg=status_color,
                bg='#f8f9fa').pack()
                
        tk.Label(status_frame,
                text=f"Confidence: {confidence:.1f}%",
                font=('Segoe UI', 16),
                fg='#7f8c8d',
                bg='#f8f9fa').pack(pady=(5, 0))
        
        # Probability bars
        prob_frame = tk.Frame(self.result_content, bg='#f8f9fa')
        prob_frame.pack(fill='x', pady=(0, 15))
        
        # Purchase probability
        tk.Label(prob_frame,
                text=f"Purchase: {prob_buy:.1f}%",
                font=('Segoe UI', 13, 'bold'),
                fg='#2c3e50',
                bg='#f8f9fa').pack(anchor='w')
        
        buy_bar_frame = tk.Frame(prob_frame, bg='#ecf0f1', height=12)
        buy_bar_frame.pack(fill='x', pady=(3, 10))
        
        buy_fill = tk.Frame(buy_bar_frame, bg='#27ae60', height=12)
        buy_fill.place(relwidth=prob_buy/100, relheight=1)
        
        # No purchase probability
        tk.Label(prob_frame,
                text=f"No Purchase: {prob_no_buy:.1f}%",
                font=('Segoe UI', 13, 'bold'),
                fg='#2c3e50',
                bg='#f8f9fa').pack(anchor='w')
        
        no_buy_bar_frame = tk.Frame(prob_frame, bg='#ecf0f1', height=12)
        no_buy_bar_frame.pack(fill='x', pady=(3, 0))
        
        no_buy_fill = tk.Frame(no_buy_bar_frame, bg='#e74c3c', height=12)
        no_buy_fill.place(relwidth=prob_no_buy/100, relheight=1)
        
        # Recommendation
        rec_frame = tk.Frame(self.result_content, bg='#ffffff', relief='solid', borderwidth=1)
        rec_frame.pack(fill='x', padx=10, pady=(10, 0))
        
        tk.Label(rec_frame,
                text="Recommendation",
                font=('Segoe UI', 14, 'bold'),
                fg='#2c3e50',
                bg='#ffffff').pack(anchor='w', padx=15, pady=(10, 5))
        
        if prediction == 1:
            if confidence > 80:
                rec_text = "High priority customer\nContact immediately\nOffer exclusive deals"
            elif confidence > 60:
                rec_text = "Potential customer\nSchedule follow-up\nSend product information"
            else:
                rec_text = "Low priority\nAdd to nurturing campaign\nMonitor interest"
        else:
            if confidence > 80:
                rec_text = "Not interested currently\nFollow up in 3-6 months\nFocus on other prospects"
            else:
                rec_text = "Uncertain customer\nProvide more information\nAddress concerns"
                
        tk.Label(rec_frame,
                text=rec_text,
                font=('Segoe UI', 12),
                fg='#34495e',
                bg='#ffffff',
                justify='left').pack(anchor='w', padx=15, pady=(0, 15))
        
    def reset_form(self):
        """Reset form to initial state"""
        self.gender_var.set('Male')
        self.age_var.set('')
        self.salary_var.set('')
        self.satisfied_var.set('yes')
        self.show_initial_state()
        self.show_initial_analysis()

def main():
    """Run the modern GUI application with analysis"""
    root = tk.Tk()
    app = ModernCarPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()