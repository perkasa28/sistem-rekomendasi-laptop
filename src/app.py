"""
Streamlit web application for the laptop recommendation system.
"""
import streamlit as st
import pandas as pd
from data_generator import generate_laptop_dataset
from recommender import LaptopRecommender

def load_data():
    """Load laptop data from CSV or generate if not exists."""
    import os
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dummy_laptops.csv')
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        df = generate_laptop_dataset()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        return df

def format_price(price):
    """Format price in rupiah format."""
    return f"Rp {price:,.0f}"

def main():
    """Main function to run the Streamlit application."""
    st.title("Sistem Rekomendasi Laptop")
    
    # Load data and initialize recommender
    data = load_data()
    recommender = LaptopRecommender()
    recommender.fit(data)
    
    # Sidebar for user inputs
    st.sidebar.header("Preferensi Anda")
    
    # Purpose selection
    purpose = st.sidebar.selectbox(
        "Tujuan Penggunaan",
        options=['Gaming', 'Office', 'Student', 'Creator'],
        help="Pilih tujuan utama penggunaan laptop"
    )
    
    # Budget range
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    budget_range = st.sidebar.slider(
        "Rentang Budget (Juta Rupiah)",
        min_value=float(min_price/1_000_000),
        max_value=float(max_price/1_000_000),
        value=(float(min_price/1_000_000), float(max_price/1_000_000)),
        step=0.5,
        format="%.1f"
    )
    
    # Feature priorities
    st.sidebar.subheader("Prioritas Spesifikasi")
    prioritize_performance = st.sidebar.checkbox("Prioritaskan Performa (CPU, GPU, RAM)")
    prioritize_mobility = st.sidebar.checkbox("Prioritaskan Mobilitas (Berat & Baterai)")
    prioritize_display = st.sidebar.checkbox("Prioritaskan Layar")
    
    # Create preference dictionary
    preferences = {
        'purpose': purpose,
        'price': sum(budget_range) * 1_000_000 / 2,  # Use middle of budget range as preference
        'weight': 1.5 if prioritize_mobility else 2.0,
        'battery': 5000 if prioritize_mobility else 4000,
        'screen_size': 15 if prioritize_display else 14,
        'ram': 16 if prioritize_performance else 8,
        'gpu': 'NVIDIA RTX 3050' if prioritize_performance else 'Intel UHD',
        'cpu': 'Intel i7' if prioritize_performance else 'Intel i5',
        'screen_type': 'OLED' if prioritize_display else 'IPS',
        'brand': 'Asus'  # Default brand
    }
    
    # Get recommendations when user clicks the button
    if st.sidebar.button("Cari Rekomendasi"):
        # Convert budget range from millions to actual values
        min_budget = float(budget_range[0]) * 1_000_000
        max_budget = float(budget_range[1]) * 1_000_000
        
        recommendations = recommender.get_recommendations_by_budget(
            preferences=preferences,
            min_budget=min_budget,
            max_budget=max_budget,
            n_recommendations=5
        )
        
        # Debug information
        st.sidebar.write("Debug Info:")
        st.sidebar.write(f"Min Budget: {format_price(min_budget)}")
        st.sidebar.write(f"Max Budget: {format_price(max_budget)}")
        st.sidebar.write(f"Total Laptops in Dataset: {len(data)}")
        filtered_count = len(data[(data['price'] >= min_budget) & (data['price'] <= max_budget)])
        st.sidebar.write(f"Laptops in Budget Range: {filtered_count}")
        
        st.header("Rekomendasi Laptop Untuk Anda")
        
        # Check if recommendations were found
        if 'message' in recommendations.columns:
            st.warning(recommendations['message'].iloc[0])
        elif len(recommendations) == 0:
            st.warning("Tidak ada rekomendasi yang ditemukan.")
        else:
            # Display recommendations
            for i, (_, laptop) in enumerate(recommendations.iterrows(), 1):
                with st.expander(f"#{i} - {laptop['brand']} {laptop['model']} ({format_price(laptop['price'])})"):
                    col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Spesifikasi Utama:**")
                    st.write(f"- CPU: {laptop['cpu']}")
                    st.write(f"- GPU: {laptop['gpu']}")
                    st.write(f"- RAM: {laptop['ram']} GB")
                    st.write(f"- Storage: {laptop['storage']}")
                
                with col2:
                    st.write("**Spesifikasi Lainnya:**")
                    st.write(f"- Layar: {laptop['screen_size']}\" {laptop['screen_type']}")
                    st.write(f"- Berat: {laptop['weight']} kg")
                    st.write(f"- Baterai: {laptop['battery']} mAh")
                    st.write(f"- Kategori: {laptop['purpose']}")
                
                st.write(f"Skor Kemiripan: {laptop['similarity_score']:.2f}")
                
                # Feedback buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("👍 Sangat Sesuai", key=f"like_{i}")
                with col2:
                    st.button("👎 Kurang Sesuai", key=f"dislike_{i}")
                with col3:
                    st.button("💬 Beri Feedback", key=f"feedback_{i}")
    
    # Show some statistics
    with st.expander("Statistik Dataset"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Laptop", len(data))
            
        with col2:
            st.metric("Rata-rata Harga", format_price(data['price'].mean()))
            
        with col3:
            st.metric("Jumlah Brand", data['brand'].nunique())

if __name__ == "__main__":
    main()
