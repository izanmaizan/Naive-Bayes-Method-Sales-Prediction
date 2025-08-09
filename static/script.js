// Global variables
let currentPrediction = null;

// DOM Content Loaded
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
});

// Initialize Application
function initializeApp() {
  initializeTabNavigation();
  initializeForm();
  initializeModal();
  console.log("🚀 Car Sales Prediction App initialized");
}

// Tab Navigation
function initializeTabNavigation() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const targetTab = this.getAttribute("data-tab");

      // Remove active class from all buttons and contents
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      tabContents.forEach((content) => content.classList.remove("active"));

      // Add active class to clicked button and corresponding content
      this.classList.add("active");
      document.getElementById(targetTab).classList.add("active");

      // Load visualization if visualization tab is selected
      if (targetTab === "visualization") {
        loadModelVisualization();
      }
    });
  });
}

// Form Initialization
function initializeForm() {
  const form = document.getElementById("prediction-form");
  const resetBtn = document.getElementById("reset-btn");

  form.addEventListener("submit", handlePrediction);
  resetBtn.addEventListener("click", resetForm);

  // Add input validation
  const ageInput = document.getElementById("age");
  const salaryInput = document.getElementById("salary");

  ageInput.addEventListener("input", function () {
    const value = parseInt(this.value);
    if (value < 15 || value > 100) {
      this.setCustomValidity("Age must be between 15 and 100 years");
    } else {
      this.setCustomValidity("");
    }
  });

  salaryInput.addEventListener("input", function () {
    const value = parseInt(this.value);
    if (value < 0) {
      this.setCustomValidity("Salary cannot be negative");
    } else {
      this.setCustomValidity("");
    }
  });
}

// Handle Prediction
async function handlePrediction(event) {
  event.preventDefault();

  const predictBtn = document.getElementById("predict-btn");
  const btnText = predictBtn.querySelector(".btn-text");
  const spinner = predictBtn.querySelector(".loading-spinner");

  // Show loading state
  predictBtn.disabled = true;
  btnText.textContent = "🔄 Analyzing...";
  spinner.style.display = "inline-block";

  // Get form data
  const formData = {
    gender: document.getElementById("gender").value,
    age: document.getElementById("age").value,
    salary: document.getElementById("salary").value,
    satisfied: document.getElementById("satisfied").value,
  };

  console.log("📤 Sending prediction request:", formData);

  try {
    // Make prediction request
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    const result = await response.json();
    console.log("📥 Prediction response:", result);

    if (result.error) {
      showError(result.error);
    } else {
      currentPrediction = result;
      displayPredictionResults(result);
      displayAnalysisResults(result.analysis);

      // Show success message
      showSuccessMessage("✅ Prediction completed successfully!");
    }
  } catch (error) {
    showError(
      "An error occurred while making the prediction. Please try again."
    );
    console.error("❌ Prediction error:", error);
  } finally {
    // Reset button state
    predictBtn.disabled = false;
    btnText.textContent = "🚀 Predict & Analyze";
    spinner.style.display = "none";
  }
}

// Display Prediction Results
function displayPredictionResults(result) {
  const resultsContent = document.getElementById("results-content");

  const statusClass = result.prediction === 1 ? "purchase" : "no-purchase";
  const statusIcon = result.prediction === 1 ? "✅" : "❌";
  const statusColor = result.prediction === 1 ? "#27ae60" : "#e74c3c";

  resultsContent.innerHTML = `
        <div class="prediction-result">
            <div class="status-display">
                <div class="status-text ${statusClass}">${statusIcon} ${result.status}</div>
                <div class="confidence-text">🔥 Confidence: ${result.confidence}%</div>
            </div>
            
            <div class="probability-section">
                <div class="probability-item">
                    <span class="probability-label">🟢 Purchase: ${result.prob_buy}%</span>
                    <div class="probability-bar">
                        <div class="probability-fill purchase" style="width: ${result.prob_buy}%"></div>
                    </div>
                </div>
                
                <div class="probability-item">
                    <span class="probability-label">🔴 No Purchase: ${result.prob_no_buy}%</span>
                    <div class="probability-bar">
                        <div class="probability-fill no-purchase" style="width: ${result.prob_no_buy}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="recommendation-section">
                <div class="recommendation-title">💡 Recommendation</div>
                <div class="recommendation-text">${result.recommendation}</div>
            </div>
        </div>
    `;

  // Animate probability bars
  setTimeout(() => {
    const fills = resultsContent.querySelectorAll(".probability-fill");
    fills.forEach((fill) => {
      fill.style.transition = "width 1s ease-in-out";
    });
  }, 100);
}

// Display Analysis Results with Clean Theme
function displayAnalysisResults(analysisData) {
  const analysisContent = document.getElementById("analysis-content");

  // Clear content
  analysisContent.innerHTML = "";

  let analysisHTML = '<div class="analysis-results-clean">';

  analysisData.forEach((section, index) => {
    analysisHTML += `
            <div class="analysis-section-clean">
                <div class="section-header">
                    <span class="section-number">${index + 1}</span>
                    <h4 class="section-title">${section.title}</h4>
                </div>
                <div class="section-content">
                    ${section.content
                      .map((line) => `<p class="analysis-line">${line}</p>`)
                      .join("")}
                </div>
            </div>
        `;
  });

  analysisHTML +=
    '<div class="analysis-success">✅ Analysis completed successfully</div>';
  analysisHTML += "</div>";

  // Add with fade-in effect
  setTimeout(() => {
    analysisContent.innerHTML = analysisHTML;

    // Auto scroll to bottom
    analysisContent.scrollTop = analysisContent.scrollHeight;

    console.log("📊 Analysis results displayed");
  }, 500);
}

// Load Model Visualization with improved error handling
async function loadModelVisualization() {
  const vizContent = document.getElementById("visualization-content");

  // Show loading state
  vizContent.innerHTML = `
        <div class="viz-placeholder">
            <div class="loading-container">
                <div class="loading-header">
                    <h3>🔄 Loading Model Visualization</h3>
                    <p>Generating interactive Plotly charts from training data...</p>
                </div>
                <div class="loading-spinner-large"></div>
                <div class="loading-steps">
                    <div class="loading-step">📊 Reading training data from sale.xlsx</div>
                    <div class="loading-step">🧮 Computing feature distributions</div>
                    <div class="loading-step">📈 Generating interactive plots</div>
                    <div class="loading-step">🎨 Applying styling and layout</div>
                </div>
            </div>
        </div>
    `;

  console.log("📈 Loading model visualization...");

  try {
    const response = await fetch("/visualization", {
      method: "GET",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
      },
    });

    console.log("📥 Visualization response status:", response.status);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    console.log("📥 Visualization response received:", result);

    if (result.error) {
      vizContent.innerHTML = `
                <div class="viz-error">
                    <div class="error-header">
                        <h3>❌ Visualization Error</h3>
                        <p>${result.error}</p>
                    </div>
                    <div class="error-solutions">
                        <h4>🔧 Troubleshooting Steps:</h4>
                        <div class="solution-grid">
                            <div class="solution-item">
                                <span class="solution-icon">📄</span>
                                <div>
                                    <strong>Check Training Data</strong>
                                    <p>Ensure sale.xlsx file exists in project directory</p>
                                </div>
                            </div>
                            <div class="solution-item">
                                <span class="solution-icon">🏃</span>
                                <div>
                                    <strong>Run Training</strong>
                                    <p>Execute 'python main.py' to generate model files</p>
                                </div>
                            </div>
                            <div class="solution-item">
                                <span class="solution-icon">🔧</span>
                                <div>
                                    <strong>Check Dependencies</strong>
                                    <p>Verify pandas, plotly, and other packages are installed</p>
                                </div>
                            </div>
                            <div class="solution-item">
                                <span class="solution-icon">🔄</span>
                                <div>
                                    <strong>Restart Server</strong>
                                    <p>Restart Flask application and try again</p>
                                </div>
                            </div>
                        </div>
                        <button class="retry-btn" onclick="loadModelVisualization()">
                            🔄 Retry Loading
                        </button>
                    </div>
                </div>
            `;
    } else if (result.visualization) {
      // Successfully loaded visualization
      vizContent.innerHTML = result.visualization;

      // Wait for DOM to update then style the plotly container
      setTimeout(() => {
        const plotlyDiv =
          vizContent.querySelector("#model-visualization") ||
          vizContent.querySelector("div[id*='plotly']") ||
          vizContent.querySelector(".plotly-graph-div");

        if (plotlyDiv) {
          plotlyDiv.style.width = "100%";
          plotlyDiv.style.height = "900px";
          plotlyDiv.style.border = "1px solid #e9ecef";
          plotlyDiv.style.borderRadius = "8px";
          plotlyDiv.style.backgroundColor = "white";
          plotlyDiv.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
          plotlyDiv.style.marginTop = "20px";

          console.log("✅ Plotly div styled successfully");
        } else {
          console.warn("⚠️ Plotly div not found, checking for other elements");
          // Try to find and style any plotly-related divs
          const allDivs = vizContent.querySelectorAll("div");
          allDivs.forEach((div) => {
            if (
              div.innerHTML.includes("plotly") ||
              div.className.includes("plotly")
            ) {
              div.style.width = "100%";
              div.style.minHeight = "900px";
            }
          });
        }
      }, 1000);

      // Show success message
      showSuccessMessage("✅ Model visualization loaded successfully!");

      console.log("✅ Visualization loaded successfully");
    } else {
      throw new Error(
        "Invalid response format - no visualization data received"
      );
    }
  } catch (error) {
    console.error("❌ Visualization error:", error);
    vizContent.innerHTML = `
            <div class="viz-error">
                <div class="error-header">
                    <h3>🔌 Connection Error</h3>
                    <p>Unable to load model visualization. Please check your connection and server status.</p>
                </div>
                <div class="error-details">
                    <h4>🐛 Error Details:</h4>
                    <code>${error.message}</code>
                </div>
                <div class="error-solutions">
                    <h4>💡 Common Solutions:</h4>
                    <div class="solution-list">
                        <div class="solution-item">• Check if Flask server is running (python app.py)</div>
                        <div class="solution-item">• Verify you're accessing http://localhost:5000</div>
                        <div class="solution-item">• Ensure all required files are in the project directory</div>
                        <div class="solution-item">• Check browser console (F12) for additional errors</div>
                        <div class="solution-item">• Make sure sale.xlsx file exists and main.py has been run</div>
                    </div>
                    <button class="retry-btn" onclick="loadModelVisualization()">
                        🔄 Try Again
                    </button>
                </div>
            </div>
        `;
  }
}

// Reset Form
function resetForm() {
  const form = document.getElementById("prediction-form");
  const resultsContent = document.getElementById("results-content");
  const analysisContent = document.getElementById("analysis-content");

  // Reset form fields
  form.reset();

  // Set default values
  document.getElementById("gender").value = "Male";
  document.getElementById("satisfied").value = "yes";

  // Reset results
  resultsContent.innerHTML = `
        <div class="placeholder">
            <p>💡 Enter customer data and click 'Predict & Analyze' to see the AI prediction result</p>
        </div>
    `;

  // Reset analysis with clean style
  analysisContent.innerHTML = `
        <div class="analysis-placeholder-clean">
            <div class="placeholder-header">
                <h3>📊 Detailed Process Analysis</h3>
                <p>This section will show you the comprehensive step-by-step process of how the Gaussian Naive Bayes algorithm makes predictions.</p>
            </div>
            
            <div class="analysis-features-clean">
                <h4>📋 The detailed analysis will include:</h4>
                
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-icon">🔢</div>
                        <h5>TAHAP 1: INPUT DATA CUSTOMER</h5>
                        <ul>
                            <li>Customer data summary and validation</li>
                            <li>Input validation and error checking</li>
                            <li>Data type verification</li>
                        </ul>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">🔄</div>
                        <h5>TAHAP 2: ENCODING DATA KATEGORIKAL</h5>
                        <ul>
                            <li>Label encoding for Gender (Male/Female → 0/1)</li>
                            <li>Label encoding for Satisfaction (no/yes → 0/1)</li>
                            <li>Feature vector construction</li>
                        </ul>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">⚙️</div>
                        <h5>TAHAP 3: GAUSSIAN NAIVE BAYES</h5>
                        <ul>
                            <li>Prior probabilities computation</li>
                            <li>Likelihood calculations for each feature</li>
                            <li>Training data statistics analysis</li>
                        </ul>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">🧮</div>
                        <h5>TAHAP 4: LIKELIHOOD CALCULATIONS</h5>
                        <ul>
                            <li>Posterior probability computation</li>
                            <li>Class probability comparisons</li>
                            <li>Mathematical breakdown</li>
                        </ul>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">📈</div>
                        <h5>TAHAP 5: VISUALISASI PROBABILITAS</h5>
                        <ul>
                            <li>Probability distribution visualization</li>
                            <li>Decision boundary analysis</li>
                            <li>Visual probability representation</li>
                        </ul>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">🎯</div>
                        <h5>TAHAP 6: DECISION EXPLANATION</h5>
                        <ul>
                            <li>Final prediction with reasoning</li>
                            <li>Contributing factor analysis</li>
                            <li>Business recommendations</li>
                        </ul>
                    </div>
                </div>
                
                <div class="call-to-action-clean">
                    🚀 Click 'Predict & Analyze' to see the detailed step-by-step analysis for your specific customer input!
                </div>
            </div>
        </div>
    `;

  currentPrediction = null;
  console.log("🔄 Form reset completed");
}

// Modal Functions
function initializeModal() {
  const modal = document.getElementById("error-modal");
  const closeBtn = document.querySelector(".close");

  if (closeBtn) {
    closeBtn.addEventListener("click", closeModal);
  }

  window.addEventListener("click", function (event) {
    if (event.target === modal) {
      closeModal();
    }
  });
}

function showError(message) {
  const modal = document.getElementById("error-modal");
  const errorMessage = document.getElementById("error-message");

  if (errorMessage) {
    errorMessage.textContent = message;
  }
  if (modal) {
    modal.style.display = "block";
  }

  console.error("❌ Error:", message);
}

function closeModal() {
  const modal = document.getElementById("error-modal");
  if (modal) {
    modal.style.display = "none";
  }
}

// Success message function
function showSuccessMessage(message) {
  const successMsg = document.createElement("div");
  successMsg.innerHTML = message;
  successMsg.style.cssText = `
    position: fixed; 
    top: 20px; 
    right: 20px; 
    background: #27ae60; 
    color: white; 
    padding: 15px 25px; 
    border-radius: 8px; 
    z-index: 1000; 
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    animation: slideIn 0.3s ease;
  `;

  document.body.appendChild(successMsg);

  setTimeout(() => {
    if (document.body.contains(successMsg)) {
      successMsg.style.animation = "slideOut 0.3s ease";
      setTimeout(() => {
        if (document.body.contains(successMsg)) {
          document.body.removeChild(successMsg);
        }
      }, 300);
    }
  }, 3000);
}

// Utility Functions
function formatNumber(num) {
  return new Intl.NumberFormat().format(num);
}

function validateFormData(data) {
  const errors = [];

  if (!data.gender || !["Male", "Female"].includes(data.gender)) {
    errors.push("Please select a valid gender");
  }

  const age = parseInt(data.age);
  if (isNaN(age) || age < 15 || age > 100) {
    errors.push("Age must be between 15 and 100 years");
  }

  const salary = parseInt(data.salary);
  if (isNaN(salary) || salary < 0) {
    errors.push("Salary must be a positive number");
  }

  if (!data.satisfied || !["yes", "no"].includes(data.satisfied)) {
    errors.push("Please select customer satisfaction level");
  }

  return errors;
}

// Export prediction data (optional feature)
function exportPrediction() {
  if (!currentPrediction) {
    showError("No prediction data available to export");
    return;
  }

  const formData = {
    gender: document.getElementById("gender").value,
    age: document.getElementById("age").value,
    salary: document.getElementById("salary").value,
    satisfied: document.getElementById("satisfied").value,
  };

  const exportData = {
    timestamp: new Date().toISOString(),
    input: formData,
    prediction: currentPrediction,
  };

  const dataStr = JSON.stringify(exportData, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(dataBlob);

  const link = document.createElement("a");
  link.href = url;
  link.download = `car_sales_prediction_${
    new Date().toISOString().split("T")[0]
  }.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);

  showSuccessMessage("📁 Prediction data exported successfully!");
}

// Keyboard shortcuts
document.addEventListener("keydown", function (event) {
  // Ctrl/Cmd + Enter to submit form
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    const form = document.getElementById("prediction-form");
    if (form && form.checkValidity()) {
      form.dispatchEvent(new Event("submit"));
    }
  }

  // Escape to close modal
  if (event.key === "Escape") {
    closeModal();
  }
});

// Auto-load visualization when page loads if we're on that tab
window.addEventListener("load", function () {
  const currentTab = document.querySelector(".tab-btn.active");
  if (currentTab && currentTab.getAttribute("data-tab") === "visualization") {
    setTimeout(loadModelVisualization, 1000);
  }
});

console.log("📱 Car Sales Prediction JavaScript loaded successfully!");

// Add CSS styles for clean theme
const style = document.createElement("style");
style.textContent = `
  @keyframes slideIn {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  
  @keyframes slideOut {
    from {
      transform: translateX(0);
      opacity: 1;
    }
    to {
      transform: translateX(100%);
      opacity: 0;
    }
  }
  
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  
  .viz-explanation {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  }
  
  .viz-explanation p {
    margin: 0;
    font-size: 16px;
    line-height: 1.6;
  }
  
  .loading-container {
    text-align: center;
    padding: 40px;
  }
  
  .loading-header h3 {
    color: #2c3e50;
    margin-bottom: 10px;
  }
  
  .loading-steps {
    margin-top: 30px;
    text-align: left;
    max-width: 400px;
    margin-left: auto;
    margin-right: auto;
  }
  
  .loading-step {
    padding: 8px 15px;
    margin: 5px 0;
    background: #f8f9fa;
    border-radius: 5px;
    border-left: 4px solid #3498db;
    font-size: 14px;
    color: #666;
  }

  /* Clean Analysis Styles */
  .analysis-container {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 25px;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }

  .analysis-content {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 25px;
    min-height: 600px;
    border: 1px solid #e9ecef;
    color: #2c3e50;
    overflow: auto;
  }

  .analysis-placeholder-clean {
    color: #2c3e50;
    line-height: 1.6;
  }

  .placeholder-header h3 {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
  }

  .placeholder-header p {
    font-size: 16px;
    margin-bottom: 25px;
    color: #666;
  }

  .analysis-features-clean h4 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 20px;
    color: #2c3e50;
  }

  .feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
  }

  .feature-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }

  .feature-icon {
    font-size: 24px;
    margin-bottom: 10px;
  }

  .feature-card h5 {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #2c3e50;
  }

  .feature-card ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .feature-card li {
    font-size: 13px;
    color: #666;
    margin-bottom: 6px;
    padding-left: 15px;
    position: relative;
  }

  .feature-card li:before {
    content: "•";
    position: absolute;
    left: 0;
    color: #3498db;
    font-weight: bold;
  }

  .call-to-action-clean {
    font-size: 16px;
    font-weight: 600;
    color: #2c3e50;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    margin-top: 20px;
  }

  /* Clean Analysis Results */
  .analysis-results-clean {
    animation: fadeIn 0.5s ease;
  }

  .analysis-section-clean {
    margin-bottom: 30px;
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    border-left: 4px solid #3498db;
  }

  .section-header {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
  }

  .section-number {
    background: #3498db;
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    margin-right: 15px;
    font-size: 14px;
  }

  .section-title {
    font-size: 16px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0;
  }

  .section-content {
    margin-left: 45px;
  }

  .analysis-line {
    margin: 5px 0;
    font-size: 14px;
    color: #444;
    line-height: 1.5;
  }

  .analysis-success {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    margin-top: 20px;
  }

  /* Visualization Error Styles */
  .viz-error {
    padding: 30px;
    text-align: center;
    background: #fff;
    border-radius: 8px;
    border: 1px solid #e9ecef;
  }

  .error-header h3 {
    color: #e74c3c;
    margin-bottom: 15px;
  }

  .error-details {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 15px;
    border-radius: 8px;
    margin: 20px 0;
  }

  .error-details code {
    color: #721c24;
    font-family: 'Consolas', monospace;
    background: rgba(255,255,255,0.8);
    padding: 5px 8px;
    border-radius: 4px;
  }

  .solution-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 15px;
    margin: 20px 0;
  }

  .solution-item {
    display: flex;
    align-items: flex-start;
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e9ecef;
  }

  .solution-icon {
    font-size: 20px;
    margin-right: 12px;
    margin-top: 2px;
  }

  .solution-item strong {
    color: #2c3e50;
    display: block;
    margin-bottom: 5px;
  }

  .solution-item p {
    color: #666;
    font-size: 14px;
    margin: 0;
  }

  .solution-list .solution-item {
    background: none;
    border: none;
    padding: 5px 0;
    font-size: 14px;
    color: #666;
  }

  .retry-btn {
    background: #3498db;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    margin-top: 20px;
    transition: background 0.2s ease;
  }

  .retry-btn:hover {
    background: #2980b9;
  }

  @media (max-width: 768px) {
    .feature-grid {
      grid-template-columns: 1fr;
    }
    
    .solution-grid {
      grid-template-columns: 1fr;
    }
  }
`;
document.head.appendChild(style);
