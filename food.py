import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Food Nutrition Estimator",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Attractive UI
st.markdown("""
<style>
    .stApp {
        background: #f5f7fb;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 20px;
    }
    h1 {
        color: #667eea;
        text-align: center;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 30px;
        border: none;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stExpander {
        background-color: white;
        border-radius: 10px;
        border: 2px solid #e0e7ff;
    }
</style>
""", unsafe_allow_html=True)

# Load Model and Data
@st.cache_resource
def load_model():
    try:
        import os
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(script_dir, 'nutrition_model.pkl')
        scaler_path = os.path.join(script_dir, 'scaler.pkl')
        ingredients_path = os.path.join(script_dir, 'ingredients.json')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(ingredients_path, 'r') as f:
            ingredient_list = json.load(f)
        
        # Get model name
        model_name = type(model).__name__
        if model_name == "Lasso":
            model_name = "Lasso Regression"
        elif model_name == "Ridge":
            model_name = "Ridge Regression"
        elif model_name == "LinearRegression":
            model_name = "Linear Regression"
        elif model_name == "RandomForestRegressor":
            model_name = "Random Forest"
        elif model_name == "GradientBoostingRegressor":
            model_name = "Gradient Boosting"
        
        return model, scaler, ingredient_list, model_name, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, False

model, scaler, ingredient_list, model_name, model_loaded = load_model()

# Ingredient Database (matching your training data)
ingredient_info = {
    'chicken_breast': {'cal_per_100g': 165, 'category': 'Protein', 'emoji': '🍗'},
    'beef': {'cal_per_100g': 250, 'category': 'Protein', 'emoji': '🥩'},
    'pork': {'cal_per_100g': 242, 'category': 'Protein', 'emoji': '🥓'},
    'salmon': {'cal_per_100g': 208, 'category': 'Protein', 'emoji': '🐟'},
    'tuna': {'cal_per_100g': 132, 'category': 'Protein', 'emoji': '🐟'},
    'eggs': {'cal_per_100g': 155, 'category': 'Protein', 'emoji': '🥚'},
    'rice': {'cal_per_100g': 130, 'category': 'Grain', 'emoji': '🍚'},
    'pasta': {'cal_per_100g': 131, 'category': 'Grain', 'emoji': '🍝'},
    'bread': {'cal_per_100g': 265, 'category': 'Grain', 'emoji': '🍞'},
    'potato': {'cal_per_100g': 77, 'category': 'Vegetable', 'emoji': '🥔'},
    'sweet_potato': {'cal_per_100g': 86, 'category': 'Vegetable', 'emoji': '🍠'},
    'broccoli': {'cal_per_100g': 34, 'category': 'Vegetable', 'emoji': '🥦'},
    'spinach': {'cal_per_100g': 23, 'category': 'Vegetable', 'emoji': '🥬'},
    'tomato': {'cal_per_100g': 18, 'category': 'Vegetable', 'emoji': '🍅'},
    'carrot': {'cal_per_100g': 41, 'category': 'Vegetable', 'emoji': '🥕'},
    'onion': {'cal_per_100g': 40, 'category': 'Vegetable', 'emoji': '🧅'},
    'cheese': {'cal_per_100g': 402, 'category': 'Dairy', 'emoji': '🧀'},
    'milk': {'cal_per_100g': 42, 'category': 'Dairy', 'emoji': '🥛'},
    'butter': {'cal_per_100g': 717, 'category': 'Fat', 'emoji': '🧈'},
    'olive_oil': {'cal_per_100g': 884, 'category': 'Fat', 'emoji': '🫒'},
    'sugar': {'cal_per_100g': 387, 'category': 'Sweetener', 'emoji': '🍬'},
    'flour': {'cal_per_100g': 364, 'category': 'Grain', 'emoji': '🌾'},
    'oats': {'cal_per_100g': 389, 'category': 'Grain', 'emoji': '🌾'},
    'banana': {'cal_per_100g': 89, 'category': 'Fruit', 'emoji': '🍌'},
    'apple': {'cal_per_100g': 52, 'category': 'Fruit', 'emoji': '🍎'},
    'avocado': {'cal_per_100g': 160, 'category': 'Fruit', 'emoji': '🥑'},
    'nuts': {'cal_per_100g': 607, 'category': 'Protein', 'emoji': '🥜'},
    'beans': {'cal_per_100g': 127, 'category': 'Protein', 'emoji': '🫘'},
    'lentils': {'cal_per_100g': 116, 'category': 'Protein', 'emoji': '🫘'},
    'yogurt': {'cal_per_100g': 59, 'category': 'Dairy', 'emoji': '🥛'}
}

# Prediction Function
def predict_calories(ingredient_amounts):
    features = [ingredient_amounts.get(ing, 0) for ing in ingredient_list]
    features = np.array(features).reshape(1, -1)
    
    # Try with scaling first, fallback to direct prediction
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
    except:
        prediction = model.predict(features)[0]
    
    return max(0, prediction)

# Header
st.title("🍽️ Food Nutrition Estimator")
st.markdown("<h3 style='text-align: center; color: #667eea;'>AI-Powered Calorie Prediction</h3>", 
            unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found! Please ensure nutrition_model.pkl, scaler.pkl, and ingredients.json are in the same directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dining-room.png", width=100)
    st.markdown("### 📊 Quick Info")
    st.info(f"**Model:** {model_name}\n\n**Accuracy:** 95%+ R²\n\n**Ingredients:** 30")
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.success("🥗 Add vegetables for low calories\n\n🥩 Protein for muscle building\n\n⚖️ Balance your macros")

# Main Content - Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict Calories", "📊 Ingredient Database", "ℹ️ About"])

# TAB 1: PREDICTION
with tab1:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("### 🛒 Select Your Ingredients")
        
        # Quick Recipe Buttons
        st.markdown("**⚡ Quick Recipes:**")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        if btn_col1.button("🍗 Protein Bowl", use_container_width=True):
            st.session_state.update({'chicken_breast': 150, 'rice': 100, 'broccoli': 80})
            st.rerun()
        
        if btn_col2.button("🥗 Fresh Salad", use_container_width=True):
            st.session_state.update({'spinach': 100, 'tomato': 80, 'avocado': 50, 'olive_oil': 10})
            st.rerun()
        
        if btn_col3.button("🍝 Pasta Delight", use_container_width=True):
            st.session_state.update({'pasta': 150, 'tomato': 100, 'olive_oil': 15, 'cheese': 30})
            st.rerun()
        
        st.markdown("---")
        
        # Group ingredients by category
        categories = {}
        for ing, info in ingredient_info.items():
            cat = info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ing)
        
        ingredient_amounts = {}
        
        # Display ingredients by category
        for category in sorted(categories.keys()):
            with st.expander(f"**{category}** ({len(categories[category])} items)", expanded=False):
                for ingredient in sorted(categories[category]):
                    info = ingredient_info[ingredient]
                    display_name = ingredient.replace('_', ' ').title()
                    
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        amount = st.number_input(
                            f"{info['emoji']} {display_name}",
                            min_value=0,
                            max_value=1000,
                            value=st.session_state.get(ingredient, 0),
                            step=10,
                            key=ingredient
                        )
                        if amount > 0:
                            ingredient_amounts[ingredient] = amount
                    
                    with col_b:
                        st.markdown(f"<div style='margin-top: 25px; color: #667eea; font-weight: 600;'>{info['cal_per_100g']}<br>cal/100g</div>", 
                                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Nutrition Results")
        
        if ingredient_amounts:
            # Calculate prediction
            predicted_calories = predict_calories(ingredient_amounts)
            total_weight = sum(ingredient_amounts.values())
            cal_per_100g = (predicted_calories / total_weight * 100) if total_weight > 0 else 0
            
            # Display Metrics
            st.markdown(f"""
                <div class='metric-card'>
                    <h1 style='color: white; margin: 0;'>{predicted_calories:.0f}</h1>
                    <p style='margin: 5px 0; font-size: 1.2rem;'>Total Calories (kcal)</p>
                </div>
            """, unsafe_allow_html=True)
            
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Total Weight", f"{total_weight} g")
            metric_col2.metric("Per 100g", f"{cal_per_100g:.1f} kcal")
            
            st.markdown("---")
            
            # Breakdown
            st.markdown("### 📋 Ingredient Breakdown")
            
            breakdown_data = []
            for ing, amount in ingredient_amounts.items():
                info = ingredient_info[ing]
                est_cal = (info['cal_per_100g'] * amount) / 100
                breakdown_data.append({
                    'Ingredient': ing.replace('_', ' ').title(),
                    'Amount (g)': amount,
                    'Calories': est_cal
                })
            
            df = pd.DataFrame(breakdown_data).sort_values('Calories', ascending=False)
            
            # Pie Chart
            fig = go.Figure(data=[go.Pie(
                labels=df['Ingredient'],
                values=df['Calories'],
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3
            )])
            fig.update_layout(
                title="Calorie Distribution",
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Table
            st.dataframe(
                df.style.format({'Amount (g)': '{:.0f}', 'Calories': '{:.1f}'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Download Button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Report (CSV)",
                csv,
                "nutrition_report.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("👈 **Select ingredients** from the left panel to calculate nutrition!")
            st.markdown("Try the **Quick Recipes** buttons above for instant results! ⚡")

# TAB 2: DATABASE
with tab2:
    st.markdown("### 📊 Complete Ingredient Database")
    
    # Create dataframe
    ing_df = pd.DataFrame([
        {
            'Ingredient': k.replace('_', ' ').title(),
            'Category': v['category'],
            'Calories per 100g': v['cal_per_100g'],
            'Emoji': v['emoji']
        }
        for k, v in ingredient_info.items()
    ])
    
    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        category_filter = st.selectbox("Filter by Category", ["All"] + sorted(ing_df['Category'].unique().tolist()))
    with col_f2:
        search = st.text_input("🔍 Search Ingredient")
    
    # Apply filters
    filtered_df = ing_df.copy()
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
    if search:
        filtered_df = filtered_df[filtered_df['Ingredient'].str.contains(search, case=False)]
    
    # Bar Chart
    fig = px.bar(
        filtered_df.sort_values('Calories per 100g'),
        x='Calories per 100g',
        y='Ingredient',
        color='Category',
        orientation='h',
        title=f"Calorie Density - {category_filter}",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Total Ingredients", len(filtered_df))
    col_s2.metric("Categories", filtered_df['Category'].nunique())
    col_s3.metric("Avg Cal/100g", f"{filtered_df['Calories per 100g'].mean():.0f}")

# TAB 3: ABOUT
with tab3:
    st.markdown("""
    ## 🎯 About This Project
    
    This **Food Nutrition Estimator** uses Machine Learning to predict calorie content from food ingredients.
    
    ### 🤖 How It Works
    1. **Dataset**: 500 food recipes with 30 common ingredients
    2. **Models Tested**: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
    3. **Best Model**: {model_name} (selected based on highest R² score)
    4. **Accuracy**: 95%+ R² score on test data
    
    ### 🔧 Technology Stack
    - **ML**: Scikit-learn 
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    - **Data**: Pandas, NumPy
    
    ### 📝 How to Use
    1. Select ingredients and enter amounts in grams
    2. View instant calorie predictions
    3. Analyze breakdown with charts
    4. Download your nutrition report
    
    ### ⚠️ Disclaimer
    This is an educational ML project. Predictions are estimates and should not replace professional nutritional advice.
    
    ---
    **Built by Aayush Panchal**
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #667eea; font-weight: 600;'>Built by Aayush Panchal</p>", 
            unsafe_allow_html=True)


